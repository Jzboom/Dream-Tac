"""Local inference for the 11-slot bi_flexiv world-action policy."""

from __future__ import annotations

import dataclasses
import os
import pickle
import time
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Sequence

import cv2
import numpy as np
import torch

from cosmos_policy.datasets.dataset_utils import apply_jpeg_compression_np
from cosmos_policy.experiments.robot.cosmos_utils import (
    apply_image_transforms,
    extract_action_chunk_from_latent_sequence,
    get_model,
    load_dataset_stats,
    unnormalize_actions,
)
from cosmos_policy.utils.tactile_image import merge_tactile_pair_vertical

CAMERA_KEYS = (
    "head",
    "left_wrist",
    "right_wrist",
    "left_tactile_0",
    "left_tactile_1",
    "right_tactile_0",
    "right_tactile_1",
)

# The checkpoint was trained with the numeric tactile names in CAMERA_KEYS,
# while the robot client names each sensor by arm and mounting side.  Keep the
# model-facing names stable and translate only at the inference boundary so the
# latent-slot order remains identical to training.
CLIENT_CAMERA_KEYS = (
    "head",
    "left_wrist",
    "right_wrist",
    "left_tactile_left",
    "left_tactile_right",
    "right_tactile_left",
    "right_tactile_right",
)
_CLIENT_TO_MODEL_CAMERA_KEY = dict(zip(CLIENT_CAMERA_KEYS, CAMERA_KEYS, strict=True))

STATE_DIM = 20
CHUNK_SIZE = 20
ACTION_DIM = 20
GRIPPER_START_IDX = 18
IMAGE_SIZE = 224
STATE_T = 11
NUM_CONDITIONAL_FRAMES = 7
ACTION_LATENT_IDX = 7
PIXEL_FRAMES = 41
_LATENT_INDICES = {
    "current_proprio_latent_idx": 1,
    "current_image_latent_idx": 2,
    "current_wrist_image_latent_idx": 3,
    "current_wrist_image2_latent_idx": 4,
    "action_latent_idx": 7,
    "future_proprio_latent_idx": -1,
    "future_image_latent_idx": 8,
    "future_wrist_image_latent_idx": 9,
    "future_wrist_image2_latent_idx": 10,
    "value_latent_idx": -1,
}

_MERGED_TACTILE_KEYS = ("left_tactile_merged", "right_tactile_merged")
_FUTURE_RGB_LATENT_INDICES = dict(
    zip(
        CAMERA_KEYS[:3],
        (8, 9, 10),
        strict=True,
    )
)
_NON_IMAGE_LATENT_INDICES = (0, 1, 7)
_TEMPORAL_COMPRESSION_FACTOR = 4
_ROTATION_6D_SLICES = (
    ("left", slice(3, 9)),
    ("right", slice(12, 18)),
)
_ROTATION_PROJECTION_EPS = 1e-6
_ROTATION_VALIDATION_ATOL = 1e-5


@dataclasses.dataclass(frozen=True)
class DreamTacBiFlexivPolicyConfig:
    checkpoint_path: str
    dataset_stats_path: str
    t5_embeddings_path: str
    default_prompt: str
    config_name: str = "cosmos_predict2_2b_480p_lerobot_bi_flexiv_wam_11slot__inference_only"
    config_file: str = "cosmos_policy/config/config.py"
    wan_vae_path: str | None = None
    num_denoising_steps: int = 10
    seed: int = 0
    image_size: int = IMAGE_SIZE
    center_crop: bool = True
    jpeg_quality: int | None = None
    clip_normalized_actions: bool = True
    action_output: Literal["absolute_from_state", "observation_relative"] = "absolute_from_state"
    normalization_mode: Literal["q99", "min_max"] = "q99"
    allow_prompt_fallback: bool = False
    decode_future_images: bool = False
    diffusion_step_cache: bool = True


def _as_hwc_uint8(image: Any, *, name: str, image_size: int | None) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Image {name!r} must have 3 dimensions, got {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"Image {name!r} must be HWC or CHW RGB, got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"Image {name!r} must have dtype uint8, got {array.dtype}")
    if image_size is not None and array.shape[:2] != (image_size, image_size):
        array = cv2.resize(array, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(array)


def prepare_rgb_images(
    raw_images: Mapping[str, Any],
    *,
    image_size: int = IMAGE_SIZE,
    center_crop: bool = True,
    jpeg_quality: int | None = None,
) -> dict[str, np.ndarray]:
    """Apply the model's deterministic preprocessing to the three RGB views."""
    missing = [name for name in CAMERA_KEYS[:3] if name not in raw_images]
    if missing:
        raise ValueError(f"RGB observation is missing cameras: {missing}")
    image_stack = np.stack(
        [_as_hwc_uint8(raw_images[name], name=name, image_size=image_size) for name in CAMERA_KEYS[:3]],
        axis=0,
    )
    if jpeg_quality is not None:
        if not 1 <= jpeg_quality <= 95:
            raise ValueError(f"jpeg_quality must be in [1, 95], got {jpeg_quality}")
        image_stack = apply_jpeg_compression_np(image_stack, quality=jpeg_quality)
    if center_crop:
        image_stack = apply_image_transforms(image_stack)
    return {name: np.ascontiguousarray(image_stack[index]) for index, name in enumerate(CAMERA_KEYS[:3])}


def prepare_camera_images(
    raw_images: Mapping[str, Any],
    *,
    image_size: int = IMAGE_SIZE,
    center_crop: bool = True,
    jpeg_quality: int | None = None,
) -> dict[str, np.ndarray]:
    """Prepare three RGB views and two merged tactile condition views.

    Robot-facing tactile names (for example ``left_tactile_left``) are mapped
    to the numeric names used by the training dataset.  The legacy numeric
    names remain accepted for offline evaluation and older clients.
    """
    if image_size != IMAGE_SIZE:
        raise ValueError(f"The merged-tactile policy requires image_size={IMAGE_SIZE}, got {image_size}")
    duplicate_aliases = [
        (client_name, model_name)
        for client_name, model_name in _CLIENT_TO_MODEL_CAMERA_KEY.items()
        if client_name != model_name and client_name in raw_images and model_name in raw_images
    ]
    if duplicate_aliases:
        pairs = [f"{client_name}/{model_name}" for client_name, model_name in duplicate_aliases]
        raise ValueError(f"Observation contains both client and legacy names for the same cameras: {pairs}")

    missing = [
        client_name
        for client_name, model_name in _CLIENT_TO_MODEL_CAMERA_KEY.items()
        if client_name not in raw_images and model_name not in raw_images
    ]
    if missing:
        raise ValueError(f"Observation is missing Dream-Tac cameras: {missing}")

    resolved_images = {}
    for client_name, model_name in _CLIENT_TO_MODEL_CAMERA_KEY.items():
        source_name = client_name if client_name in raw_images else model_name
        resolved_images[model_name] = _as_hwc_uint8(
            raw_images[source_name],
            name=source_name,
            image_size=None if "tactile" in model_name else image_size,
        )

    images = prepare_rgb_images(
        resolved_images,
        image_size=image_size,
        center_crop=center_crop,
        jpeg_quality=jpeg_quality,
    )

    merged_stack = np.stack(
        [
            merge_tactile_pair_vertical(
                resolved_images["left_tactile_0"],
                resolved_images["left_tactile_1"],
            ),
            merge_tactile_pair_vertical(
                resolved_images["right_tactile_0"],
                resolved_images["right_tactile_1"],
            ),
        ],
        axis=0,
    )
    if jpeg_quality is not None:
        merged_stack = apply_jpeg_compression_np(merged_stack, quality=jpeg_quality)
    if center_crop:
        # Training applies the same 90%-area resized crop augmentation to RGB
        # and merged tactile frames.  At inference use the deterministic center
        # counterpart for both modalities; otherwise tactile sees a different
        # spatial distribution between training and deployment.
        merged_stack = apply_image_transforms(merged_stack)
    images.update(
        {name: np.ascontiguousarray(merged_stack[index]) for index, name in enumerate(_MERGED_TACTILE_KEYS)}
    )
    return images


def validate_and_prepare_observation(
    observation: Mapping[str, Any],
    *,
    image_size: int = IMAGE_SIZE,
    center_crop: bool = True,
    jpeg_quality: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, str | None]:
    """Validate a robot request and apply deterministic model-side image transforms."""
    if "state" not in observation:
        raise ValueError("Observation is missing 'state'")
    state = np.asarray(observation["state"], dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"state must have shape ({STATE_DIM},), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("state contains NaN or Inf")

    raw_images = observation.get("images")
    if not isinstance(raw_images, Mapping):
        raise ValueError("Observation is missing an 'images' mapping")
    images = prepare_camera_images(
        raw_images,
        image_size=image_size,
        center_crop=center_crop,
        jpeg_quality=jpeg_quality,
    )

    # MsgPack decodes ndarray payloads as read-only buffer views.  torch.from_numpy
    # requires writable storage even though inference only reads this tensor.
    gate = np.array(observation.get("tactile_self_attn_gate"), dtype=np.float32, copy=True)
    if gate.shape != (2,):
        raise ValueError(f"tactile_self_attn_gate must have shape (2,), got {gate.shape}")
    if not np.isfinite(gate).all():
        raise ValueError("tactile_self_attn_gate contains NaN or Inf")

    prompt = observation.get("prompt")
    if prompt is not None:
        prompt = str(prompt).strip()
        if not prompt:
            prompt = None
    return state, images, gate, prompt


def decode_future_images(
    model: Any,
    generated_latent: torch.Tensor,
    orig_clean_latent_frames: torch.Tensor,
    *,
    non_image_latent_indices: Sequence[int] = _NON_IMAGE_LATENT_INDICES,
    temporal_compression_factor: int = _TEMPORAL_COMPRESSION_FACTOR,
) -> dict[str, np.ndarray]:
    """Decode the three predicted future RGB slots into RGB uint8 images.

    Proprio/action values are injected after VAE encoding. Their latent slots must
    therefore be restored from the clean pre-injection latent before decoding, or
    they introduce visible artifacts across the reconstructed video.
    """
    if generated_latent.ndim != 5 or orig_clean_latent_frames.shape != generated_latent.shape:
        raise ValueError(
            "generated_latent and orig_clean_latent_frames must have the same (B,C,T,H,W) shape, "
            f"got {tuple(generated_latent.shape)} and {tuple(orig_clean_latent_frames.shape)}"
        )
    if generated_latent.shape[0] != 1:
        raise ValueError(f"bi_flexiv inference expects batch size 1, got {generated_latent.shape[0]}")

    latent_for_decode = generated_latent.clone()
    for latent_index in non_image_latent_indices:
        if not 0 <= latent_index < latent_for_decode.shape[2]:
            raise IndexError(f"Non-image latent index {latent_index} is outside T={latent_for_decode.shape[2]}")
        latent_for_decode[:, :, latent_index] = orig_clean_latent_frames[:, :, latent_index]

    decoded = ((model.decode(latent_for_decode) + 1.0) * 127.5).clamp(0, 255)
    decoded = decoded.permute(0, 2, 3, 4, 1).to(torch.uint8).cpu().numpy()

    result: dict[str, np.ndarray] = {}
    for camera_name, latent_index in _FUTURE_RGB_LATENT_INDICES.items():
        raw_index = (latent_index - 1) * temporal_compression_factor + 1
        if not 0 <= raw_index < decoded.shape[1]:
            raise IndexError(
                f"Decoded frame index {raw_index} for {camera_name!r} is outside T={decoded.shape[1]}"
            )
        result[camera_name] = np.ascontiguousarray(decoded[0, raw_index])
    return result


def build_pixel_video(images: Mapping[str, np.ndarray]) -> np.ndarray:
    """Build the 41-frame uint8 video tensor used by the 11-slot WAN VAE."""
    blank = np.zeros_like(images["head"])
    unique_frames = [
        blank,
        blank,
        images["head"],
        images["left_wrist"],
        images["right_wrist"],
        images["left_tactile_merged"],
        images["right_tactile_merged"],
        blank,
        images["head"],
        images["left_wrist"],
        images["right_wrist"],
    ]
    repeats = np.asarray([1] + [4] * 10, dtype=np.int64)
    video_thwc = np.repeat(np.stack(unique_frames, axis=0), repeats, axis=0)
    if video_thwc.shape != (PIXEL_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected pixel video shape: {video_thwc.shape}")
    return np.ascontiguousarray(np.transpose(video_thwc, (3, 0, 1, 2))[None])


def _normalization_bounds(
    dataset_stats: Mapping[str, np.ndarray],
    data_key: str,
    normalization_mode: Literal["q99", "min_max"],
) -> tuple[np.ndarray, np.ndarray]:
    if normalization_mode == "q99":
        low_suffix, high_suffix = "q01", "q99"
    elif normalization_mode == "min_max":
        low_suffix, high_suffix = "min", "max"
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization_mode!r}")
    low = np.asarray(dataset_stats[f"{data_key}_{low_suffix}"], dtype=np.float32)
    high = np.asarray(dataset_stats[f"{data_key}_{high_suffix}"], dtype=np.float32)
    return low, high


def normalize_proprio(
    state: np.ndarray,
    dataset_stats: Mapping[str, np.ndarray],
    normalization_mode: Literal["q99", "min_max"] = "q99",
) -> np.ndarray:
    low, high = _normalization_bounds(dataset_stats, "proprio", normalization_mode)
    if low.shape != (STATE_DIM,) or high.shape != (STATE_DIM,):
        raise ValueError(f"proprio stats must have shape ({STATE_DIM},), got {low.shape} and {high.shape}")
    if normalization_mode == "q99":
        valid = (high - low) >= 1e-6
        normalized = np.zeros_like(state, dtype=np.float32)
        normalized[valid] = 2.0 * ((state[valid] - low[valid]) / (high[valid] - low[valid])) - 1.0
        normalized[~valid] = state[~valid]
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
    denom = np.where((high - low) < 1e-6, 1.0, high - low)
    return (2.0 * ((state - low) / denom) - 1.0).astype(np.float32)


def project_rotation_6d(
    rotation_6d: np.ndarray,
    *,
    eps: float = _ROTATION_PROJECTION_EPS,
    validation_atol: float = _ROTATION_VALIDATION_ATOL,
) -> np.ndarray:
    """Project two predicted rotation columns onto a valid SO(3) frame.

    Dream-Tac predicts observation-relative offsets in the ambient 6D
    representation used during training. Adding the request state can move the
    two columns away from unit and orthogonal constraints, so project them with
    guarded Gram-Schmidt before returning robot-facing absolute targets.
    """
    rotation = np.asarray(rotation_6d, dtype=np.float64)
    if rotation.shape != (6,):
        raise ValueError(f"Expected rotation_6d shape (6,), got {rotation.shape}")
    if not np.isfinite(rotation).all():
        raise ValueError("rotation_6d contains NaN or Inf")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if validation_atol <= 0:
        raise ValueError(f"validation_atol must be positive, got {validation_atol}")

    first_column = rotation[:3]
    second_column = rotation[3:]

    first_norm = float(np.linalg.norm(first_column))
    if first_norm < eps:
        raise ValueError(f"first rotation column norm {first_norm:.3e} is below {eps:.3e}")
    first_unit = first_column / first_norm

    second_orthogonal = second_column - np.dot(first_unit, second_column) * first_unit
    second_norm = float(np.linalg.norm(second_orthogonal))
    if second_norm < eps:
        raise ValueError(
            "rotation columns are degenerate or nearly collinear: "
            f"orthogonal component norm {second_norm:.3e} is below {eps:.3e}"
        )
    second_unit = second_orthogonal / second_norm
    third_unit = np.cross(first_unit, second_unit)
    rotation_matrix = np.column_stack((first_unit, second_unit, third_unit))

    if not np.isfinite(rotation_matrix).all():
        raise ValueError("projected rotation matrix contains NaN or Inf")
    if not np.allclose(
        rotation_matrix.T @ rotation_matrix,
        np.eye(3, dtype=np.float64),
        atol=validation_atol,
        rtol=0.0,
    ):
        raise ValueError("projected rotation matrix is not orthonormal")
    determinant = float(np.linalg.det(rotation_matrix))
    if not np.isclose(determinant, 1.0, atol=validation_atol, rtol=0.0):
        raise ValueError(f"projected rotation determinant must be +1, got {determinant:.8f}")

    return np.concatenate((first_unit, second_unit)).astype(np.float32)


def observation_relative_to_absolute(relative_actions: np.ndarray, base_state: np.ndarray) -> np.ndarray:
    """Convert a chunk relative to its request observation into absolute targets."""
    relative = np.asarray(relative_actions, dtype=np.float32)
    base = np.asarray(base_state, dtype=np.float32)
    if relative.shape != (CHUNK_SIZE, ACTION_DIM):
        raise ValueError(f"Expected relative actions {(CHUNK_SIZE, ACTION_DIM)}, got {relative.shape}")
    if base.shape != (STATE_DIM,):
        raise ValueError(f"Expected base state ({STATE_DIM},), got {base.shape}")
    if not np.isfinite(relative).all():
        raise ValueError("Observation-relative actions contain NaN or Inf")
    if not np.isfinite(base).all():
        raise ValueError("Base state contains NaN or Inf")
    absolute = relative.copy()
    absolute[:, :GRIPPER_START_IDX] = base[None, :GRIPPER_START_IDX] + relative[:, :GRIPPER_START_IDX]
    for step_idx in range(CHUNK_SIZE):
        for arm_name, rotation_slice in _ROTATION_6D_SLICES:
            try:
                absolute[step_idx, rotation_slice] = project_rotation_6d(absolute[step_idx, rotation_slice])
            except ValueError as exc:
                raise ValueError(f"Invalid {arm_name} 6D rotation at action step {step_idx}: {exc}") from exc
    absolute[:, GRIPPER_START_IDX:] = np.clip(relative[:, GRIPPER_START_IDX:], 0.0, 1.0)
    return absolute


class DreamTacBiFlexivPolicy:
    """Load a Dream-Tac checkpoint and expose an OpenPI-style ``infer`` API."""

    def __init__(
        self,
        config: DreamTacBiFlexivPolicyConfig,
        *,
        model: Any | None = None,
        dataset_stats: Mapping[str, np.ndarray] | None = None,
        text_embeddings: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        if config.image_size != IMAGE_SIZE:
            raise ValueError(f"The trained bi_flexiv policy requires image_size={IMAGE_SIZE}, got {config.image_size}")
        if config.diffusion_step_cache and config.num_denoising_steps not in (5, 10):
            raise ValueError("diffusion_step_cache requires num_denoising_steps to be 5 or 10")
        experiment_opts: list[str] = []
        if config.wan_vae_path:
            wan_vae_path = os.path.abspath(os.path.expanduser(config.wan_vae_path))
            if not os.path.exists(wan_vae_path):
                raise FileNotFoundError(f"WAN VAE checkpoint does not exist: {wan_vae_path}")
            os.environ["DREAMTAC_WAN_VAE"] = wan_vae_path
            experiment_opts.append(f"model.config.tokenizer.vae_pth={wan_vae_path}")

        if not torch.cuda.is_available() and model is None:
            raise RuntimeError("Dream-Tac local inference requires a CUDA GPU")

        if model is None:
            load_cfg = SimpleNamespace(
                config=config.config_name,
                ckpt_path=config.checkpoint_path,
                config_file=config.config_file,
                experiment_opts=experiment_opts,
            )
            self.model, self.cosmos_config = get_model(load_cfg)
        else:
            self.model = model
            self.cosmos_config = None
        self.model.eval()
        if config.diffusion_step_cache:
            from cosmos_policy.experiments.robot.openloop_hard_residual_cache import (
                apply_openloop_hard_residual_cache,
            )

            apply_openloop_hard_residual_cache(self.model, num_denoising_steps=config.num_denoising_steps)
        elif getattr(self.model, "_openloop_hard_residual_cache_patched", False):
            from cosmos_policy.experiments.robot.openloop_hard_residual_cache import (
                remove_openloop_hard_residual_cache,
            )

            remove_openloop_hard_residual_cache(self.model)
        self.device = self._get_model_device()

        self.dataset_stats = (
            {key: np.asarray(value, dtype=np.float32) for key, value in dataset_stats.items()}
            if dataset_stats is not None
            else {
                key: np.asarray(value, dtype=np.float32)
                for key, value in load_dataset_stats(config.dataset_stats_path).items()
            }
        )
        self._validate_stats()

        if text_embeddings is None:
            with open(config.t5_embeddings_path, "rb") as file:
                text_embeddings = pickle.load(file)
        self.text_embeddings = dict(text_embeddings)
        if not self.text_embeddings:
            raise ValueError("T5 embedding cache is empty")
        self._validate_model_contract()

    @property
    def metadata(self) -> dict[str, Any]:
        action_space = (
            "absolute_tcp18_absolute_gripper2"
            if self.config.action_output == "absolute_from_state"
            else "observation_relative_tcp18_absolute_gripper2"
        )
        return {
            "service": "dreamtac-bi_flexiv",
            "config": self.config.config_name,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "action_horizon": CHUNK_SIZE,
            "action_space": action_space,
            "normalization_mode": self.config.normalization_mode,
            "camera_keys": CLIENT_CAMERA_KEYS,
            "legacy_camera_keys": CAMERA_KEYS,
            "condition_image_keys": CAMERA_KEYS[:3] + _MERGED_TACTILE_KEYS,
            "image_shape": (self.config.image_size, self.config.image_size, 3),
            "future_images_decoded": self.config.decode_future_images,
            "state_t": STATE_T,
            "num_conditional_frames": NUM_CONDITIONAL_FRAMES,
            "action_latent_idx": ACTION_LATENT_IDX,
            "diffusion_step_cache": self.config.diffusion_step_cache,
            "future_image_horizon": CHUNK_SIZE,
            "rtc_supported": False,
        }

    def _get_model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _validate_stats(self) -> None:
        stats_chunk_size = self.dataset_stats.get("action_chunk_size")
        stats_gripper_start_idx = self.dataset_stats.get("gripper_start_idx")
        if (
            stats_chunk_size is None
            or int(np.asarray(stats_chunk_size).item()) != CHUNK_SIZE
            or stats_gripper_start_idx is None
            or int(np.asarray(stats_gripper_start_idx).item()) != GRIPPER_START_IDX
        ):
            raise ValueError(
                "Dataset statistics shape metadata do not match this policy: "
                f"expected chunk/gripper={CHUNK_SIZE}/{GRIPPER_START_IDX}, "
                f"got {stats_chunk_size!r}/{stats_gripper_start_idx!r}"
            )
        if self.config.normalization_mode == "q99":
            required = ("actions_q01", "actions_q99", "proprio_q01", "proprio_q99")
        elif self.config.normalization_mode == "min_max":
            required = ("actions_min", "actions_max", "proprio_min", "proprio_max")
        else:
            raise ValueError(f"Unsupported normalization mode: {self.config.normalization_mode!r}")
        missing = [key for key in required if key not in self.dataset_stats]
        if missing:
            raise ValueError(f"Dataset statistics missing keys: {missing}")
        for key in required:
            value = np.asarray(self.dataset_stats[key])
            if value.shape != (ACTION_DIM,):
                raise ValueError(f"Statistic {key!r} must have shape ({ACTION_DIM},), got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"Statistic {key!r} contains NaN or Inf")

    def _validate_model_contract(self) -> None:
        state_t = int(getattr(self.model.config, "state_t", -1))
        min_frames = int(getattr(self.model.config, "min_num_conditional_frames", -1))
        max_frames = int(getattr(self.model.config, "max_num_conditional_frames", -1))
        if (state_t, min_frames, max_frames) != (STATE_T, NUM_CONDITIONAL_FRAMES, NUM_CONDITIONAL_FRAMES):
            raise ValueError(
                "Checkpoint/config is not the 11-slot Dream-Tac bi_flexiv policy: "
                f"state_t={state_t}, min_conditional={min_frames}, max_conditional={max_frames}"
            )
        pixel_frames = int(self.model.tokenizer.get_pixel_num_frames(state_t))
        if pixel_frames != PIXEL_FRAMES:
            raise ValueError(f"Tokenizer expects {pixel_frames} pixel frames, expected {PIXEL_FRAMES}")

    def _get_text_embedding(self, prompt: str) -> torch.Tensor:
        if prompt in self.text_embeddings:
            value = self.text_embeddings[prompt]
        elif self.config.allow_prompt_fallback:
            value = next(iter(self.text_embeddings.values()))
        else:
            available = list(self.text_embeddings)[:5]
            raise KeyError(f"Prompt {prompt!r} is not in the T5 cache. Available examples: {available}")

        embedding = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        while embedding.ndim > 3 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        if embedding.ndim == 2:
            embedding = embedding.unsqueeze(0)
        if embedding.ndim != 3 or embedding.shape[0] != 1:
            raise ValueError(f"T5 embedding must be (1, tokens, dim) or (tokens, dim), got {tuple(embedding.shape)}")
        return embedding.to(device=self.device, dtype=torch.bfloat16)

    def _build_data_batch(
        self,
        *,
        pixel_video: np.ndarray,
        normalized_state: np.ndarray,
        tactile_gate: np.ndarray,
        prompt: str,
    ) -> dict[str, Any]:
        text_embedding = self._get_text_embedding(prompt)
        batch_size = 1
        data_batch: dict[str, Any] = {
            "dataset_name": "video_data",
            "video": torch.from_numpy(pixel_video).to(device=self.device, dtype=torch.uint8),
            "t5_text_embeddings": text_embedding,
            "t5_text_mask": torch.ones((batch_size, text_embedding.shape[1]), device=self.device, dtype=torch.int64),
            "fps": torch.tensor([16], device=self.device, dtype=torch.bfloat16),
            "padding_mask": torch.zeros(
                (batch_size, 1, self.config.image_size, self.config.image_size),
                device=self.device,
                dtype=torch.bfloat16,
            ),
            "num_conditional_frames": NUM_CONDITIONAL_FRAMES,
            "proprio": torch.from_numpy(normalized_state[None]).to(device=self.device, dtype=torch.bfloat16),
            "tactile_self_attn_gate": torch.from_numpy(tactile_gate[None]).to(device=self.device, dtype=torch.float32),
        }
        for name, index in _LATENT_INDICES.items():
            data_batch[name] = torch.tensor([index], device=self.device, dtype=torch.int64)
        return data_batch

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def infer(self, observation: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            raise NotImplementedError(
                "Dream-Tac does not support OpenPI RTC kwargs. Disable --rtc-enabled on the robot client."
            )

        infer_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        state, images, tactile_gate, request_prompt = validate_and_prepare_observation(
            observation,
            image_size=self.config.image_size,
            center_crop=self.config.center_crop,
            jpeg_quality=self.config.jpeg_quality,
        )
        prompt = request_prompt or self.config.default_prompt
        if not prompt:
            raise ValueError("No prompt was supplied and default_prompt is empty")
        normalized_state = normalize_proprio(
            state,
            self.dataset_stats,
            normalization_mode=self.config.normalization_mode,
        )
        pixel_video = build_pixel_video(images)
        data_batch = self._build_data_batch(
            pixel_video=pixel_video,
            normalized_state=normalized_state,
            tactile_gate=tactile_gate,
            prompt=prompt,
        )
        # Tensor transfers in _build_data_batch may be asynchronous.  Synchronize
        # here so their cost is attributed to preprocessing instead of vanishing
        # between the preprocessing and sampling timers.
        self._synchronize()
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        sampling_start = time.perf_counter()
        if self.config.diffusion_step_cache:
            from cosmos_policy.experiments.robot.openloop_hard_residual_cache import (
                reset_openloop_denoise_counter,
            )

            reset_openloop_denoise_counter(self.model)
        with torch.inference_mode():
            generated_result = self.model.generate_samples_from_batch(
                data_batch,
                n_sample=1,
                num_steps=self.config.num_denoising_steps,
                seed=self.config.seed,
                is_negative_prompt=False,
                use_variance_scale=False,
                return_orig_clean_latent_frames=self.config.decode_future_images,
            )
        orig_clean_latent_frames = None
        if isinstance(generated_result, tuple):
            generated, orig_clean_latent_frames = generated_result
        else:
            generated = generated_result
        self._synchronize()
        sampling_ms = (time.perf_counter() - sampling_start) * 1000.0

        postprocess_start = time.perf_counter()
        action_indices = torch.tensor([ACTION_LATENT_IDX], device=generated.device, dtype=torch.int64)
        normalized_actions = extract_action_chunk_from_latent_sequence(
            generated,
            action_shape=(CHUNK_SIZE, ACTION_DIM),
            action_indices=action_indices,
        ).to(torch.float32)
        normalized_actions_np = normalized_actions.cpu().numpy()
        clipped_fraction = float(np.mean(np.abs(normalized_actions_np) > 1.0))
        if self.config.clip_normalized_actions:
            normalized_actions_np = np.clip(normalized_actions_np, -1.0, 1.0)
        observation_relative = unnormalize_actions(
            normalized_actions_np,
            self.dataset_stats,
            normalization_mode=self.config.normalization_mode,
        )[0].astype(np.float32)
        if observation_relative.shape != (CHUNK_SIZE, ACTION_DIM) or not np.isfinite(observation_relative).all():
            raise ValueError(f"Invalid Dream-Tac action output: shape={observation_relative.shape}")

        if self.config.action_output == "absolute_from_state":
            actions = observation_relative_to_absolute(observation_relative, state)
            action_space = "absolute_tcp18_absolute_gripper2"
        else:
            actions = observation_relative
            action_space = "observation_relative_tcp18_absolute_gripper2"

        future_images: dict[str, np.ndarray] | None = None
        future_decode_ms = 0.0
        if self.config.decode_future_images:
            if orig_clean_latent_frames is None:
                raise RuntimeError("Model did not return clean latent frames required for future image decoding")
            future_decode_start = time.perf_counter()
            with torch.inference_mode():
                future_images = decode_future_images(
                    self.model,
                    generated,
                    orig_clean_latent_frames,
                )
            self._synchronize()
            future_decode_ms = (time.perf_counter() - future_decode_start) * 1000.0
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        policy_total_ms = (time.perf_counter() - infer_start) * 1000.0

        timing = {
            "preprocess_ms": preprocess_ms,
            "vae_and_denoise_ms": sampling_ms,
            "postprocess_ms": postprocess_ms,
            "policy_total_ms": policy_total_ms,
        }
        if self.config.decode_future_images:
            timing["future_decode_ms"] = future_decode_ms

        response: dict[str, Any] = {
            "actions": np.ascontiguousarray(actions, dtype=np.float32),
            "action_space": action_space,
            "server_timing": timing,
            "normalized_action_clipped_fraction": clipped_fraction,
        }
        if self.config.action_output == "absolute_from_state":
            response["observation_relative_actions"] = observation_relative
        if "observation_seq" in observation:
            response["observation_seq"] = observation["observation_seq"]
        if future_images is not None:
            response["future_images"] = future_images
        return response

    def warmup(self) -> dict[str, Any]:
        low, high = _normalization_bounds(self.dataset_stats, "proprio", self.config.normalization_mode)
        midpoint = 0.5 * (low + high)
        observation = {
            "state": midpoint,
            "images": {
                name: np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
                for name in CAMERA_KEYS
            },
            "tactile_self_attn_gate": np.asarray([0.15, 0.15], dtype=np.float32),
            "prompt": self.config.default_prompt,
        }
        return self.infer(observation)

    def reset(self) -> None:
        """The server policy is stateless between requests."""
