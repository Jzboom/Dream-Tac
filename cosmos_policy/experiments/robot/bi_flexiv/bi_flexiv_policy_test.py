from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from cosmos_policy._src.predict2.conditioner import DataType
from cosmos_policy.experiments.robot.bi_flexiv.bi_flexiv_policy import (
    ACTION_LATENT_IDX,
    CAMERA_KEYS,
    NUM_CONDITIONAL_FRAMES,
    PIXEL_FRAMES,
    STATE_T,
    DreamTacBiFlexivPolicy,
    DreamTacBiFlexivPolicyConfig,
    build_pixel_video,
    decode_future_images,
    prepare_camera_images,
)
from cosmos_policy.experiments.robot.openloop_hard_residual_cache import (
    minimal_v4_dit_forward_with_hard_block_cache,
    reset_openloop_denoise_counter,
)


CLIENT_CAMERA_KEYS = (
    "head",
    "left_wrist",
    "right_wrist",
    "left_tactile_left",
    "left_tactile_right",
    "right_tactile_left",
    "right_tactile_right",
)


def _raw_images() -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for value, name in enumerate(CLIENT_CAMERA_KEYS[:3], start=1):
        images[name] = np.full((480, 640, 3), value, dtype=np.uint8)
    for value, name in enumerate(CLIENT_CAMERA_KEYS[3:], start=4):
        images[name] = np.full((400, 700, 3), value, dtype=np.uint8)
    return images


def test_default_policy_contract_is_11_slot_10_step_cached() -> None:
    config = DreamTacBiFlexivPolicyConfig("ckpt", "stats", "t5", "prompt")

    assert STATE_T == 11
    assert NUM_CONDITIONAL_FRAMES == 7
    assert ACTION_LATENT_IDX == 7
    assert PIXEL_FRAMES == 41
    assert config.num_denoising_steps == 10
    assert config.diffusion_step_cache is True


def test_prepare_camera_images_merges_each_raw_tactile_pair() -> None:
    images = prepare_camera_images(_raw_images(), center_crop=False)

    assert tuple(images) == (
        "head",
        "left_wrist",
        "right_wrist",
        "left_tactile_merged",
        "right_tactile_merged",
    )
    assert all(image.shape == (224, 224, 3) for image in images.values())
    np.testing.assert_array_equal(images["left_tactile_merged"][:, :14], 0)
    np.testing.assert_array_equal(images["left_tactile_merged"][:, 210:], 0)
    np.testing.assert_array_equal(images["left_tactile_merged"][40, 40], np.full(3, 4, dtype=np.uint8))
    np.testing.assert_array_equal(images["left_tactile_merged"][180, 40], np.full(3, 5, dtype=np.uint8))


def test_build_pixel_video_uses_only_the_fixed_11_slot_layout() -> None:
    images = prepare_camera_images(_raw_images(), center_crop=False)

    video = build_pixel_video(images)

    assert video.shape == (1, 3, 41, 224, 224)
    # Slots 5/6 are merged tactile, slot 7 is action, and slots 8--10 are future RGB placeholders.
    np.testing.assert_array_equal(video[0, :, 17, :, :14], 0)
    np.testing.assert_array_equal(video[0, :, 21, :, :14], 0)
    np.testing.assert_array_equal(video[0, :, 25], 0)
    np.testing.assert_array_equal(video[0, :, 29], 1)
    np.testing.assert_array_equal(video[0, :, 33], 2)
    np.testing.assert_array_equal(video[0, :, 37], 3)


def test_decode_future_images_returns_only_three_rgb_views() -> None:
    class _Model:
        @staticmethod
        def decode(latent: torch.Tensor) -> torch.Tensor:
            return torch.zeros(latent.shape[0], 3, PIXEL_FRAMES, 2, 2, device=latent.device)

    generated = torch.zeros(1, 16, STATE_T, 1, 1)
    decoded = decode_future_images(_Model(), generated, torch.zeros_like(generated))

    assert tuple(decoded) == CAMERA_KEYS[:3]
    assert all(image.shape == (2, 2, 3) for image in decoded.values())


def test_diffusion_cache_rejects_unsupported_step_count_before_loading_model() -> None:
    config = DreamTacBiFlexivPolicyConfig(
        "unused",
        "unused",
        "unused",
        "unused",
        num_denoising_steps=4,
        diffusion_step_cache=True,
    )

    with pytest.raises(ValueError, match="5 or 10"):
        DreamTacBiFlexivPolicy(config)


class _TimestepEmbedding(nn.Module):
    def forward(self, timesteps: torch.Tensor) -> tuple[torch.Tensor, None]:
        return timesteps.unsqueeze(-1), None


class _CountingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor, *_args: object, **_kwargs: object) -> torch.Tensor:
        self.calls += 1
        return x + 1


class _CacheTestNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.use_crossattn_projection = False
        self.extra_image_context_dim = None
        self.use_wan_fp32_strategy = False
        self.t_embedder = _TimestepEmbedding()
        self.t_embedding_norm = nn.Identity()
        self.block = _CountingBlock()
        self.blocks = nn.ModuleList([self.block])

    @staticmethod
    def prepare_embedded_sequence(
        x: torch.Tensor, **_kwargs: object
    ) -> tuple[torch.Tensor, None, None]:
        return x.permute(0, 2, 3, 4, 1), None, None

    @staticmethod
    def _tactile_self_attn_block_kw(_x: torch.Tensor, _gate: object) -> dict[str, object]:
        return {}

    @staticmethod
    def final_layer(x: torch.Tensor, *_args: object, **_kwargs: object) -> torch.Tensor:
        return x

    @staticmethod
    def unpatchify(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 4, 1, 2, 3)


def test_ten_call_diffusion_cache_runs_full_blocks_only_on_calls_zero_and_two() -> None:
    net = _CacheTestNet()
    x = torch.zeros(1, 1, STATE_T, 1, 1)
    timesteps = torch.zeros(1, STATE_T)
    crossattn = torch.zeros(1, 1, 1)

    for call_idx in range(10):
        minimal_v4_dit_forward_with_hard_block_cache(
            net,
            x,
            timesteps,
            crossattn,
            call_idx=call_idx,
            total_calls=10,
            data_type=DataType.IMAGE,
        )

    assert net.block.calls == 2

    class _Model:
        pass

    model = _Model()
    model.net = net
    model._openloop_denoise_idx = 9
    reset_openloop_denoise_counter(model)  # type: ignore[arg-type]
    assert model._openloop_denoise_idx == 0
    assert not hasattr(net, "_openloop_residual_after_first")
    assert not hasattr(net, "_openloop_residual_after_third")
