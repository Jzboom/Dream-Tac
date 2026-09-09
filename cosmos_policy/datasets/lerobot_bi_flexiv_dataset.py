# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LeRobot v3 loader for the dual-arm bi_flexiv platform.

The sequence uses 11 latent slots. Five condition-image slots each contain
four history frames, and three prediction-image slots each contain four future
frames. With the WAN2.1 temporal compression factor of 4 this remains 41 pixel
frames.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from typing import Any, Literal

import av
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_policy.datasets.dataset_common import get_action_chunk_with_padding
from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.utils.bi_flexiv_video_layout import (
    ACTION_CHUNK_SIZE,
    ACTION_LATENT_IDX as LAYOUT_ACTION_LATENT_IDX,
    CURRENT_HEAD_IDX as LAYOUT_CURRENT_HEAD_IDX,
    CURRENT_LEFT_TACTILE_IDX as LAYOUT_CURRENT_LEFT_TACTILE_IDX,
    CURRENT_LEFT_WRIST_IDX as LAYOUT_CURRENT_LEFT_WRIST_IDX,
    CURRENT_PROPRIO_IDX as LAYOUT_CURRENT_PROPRIO_IDX,
    CURRENT_RIGHT_TACTILE_IDX as LAYOUT_CURRENT_RIGHT_TACTILE_IDX,
    CURRENT_RIGHT_WRIST_IDX as LAYOUT_CURRENT_RIGHT_WRIST_IDX,
    FUTURE_HEAD_IDX as LAYOUT_FUTURE_HEAD_IDX,
    FUTURE_IMAGE_OFFSETS,
    FUTURE_LEFT_WRIST_IDX as LAYOUT_FUTURE_LEFT_WRIST_IDX,
    FUTURE_RIGHT_WRIST_IDX as LAYOUT_FUTURE_RIGHT_WRIST_IDX,
    NUM_CONDITIONAL_FRAMES as LAYOUT_NUM_CONDITIONAL_FRAMES,
    PIXEL_FRAMES as LAYOUT_PIXEL_FRAMES,
    RGB_HISTORY_OFFSETS,
    STATE_T as LAYOUT_STATE_T,
    TACTILE_HISTORY_OFFSETS,
    build_pixel_frame_sequence,
    clamped_relative_indices,
)
from cosmos_policy.utils.tactile_image import merge_tactile_pair_vertical
from cosmos_policy.utils.tactile_self_attn_gate import scalar_gate_from_raw


# Never touch a decoder inherited across ``fork``.  libavcodec may have worker
# threads and locks that no longer exist in the child, so even calling close()
# can deadlock.  Keep the stale Python objects alive until multiprocessing
# terminates the worker process; the OS then releases their file descriptors.
_FORK_INHERITED_VIDEO_CACHES: list[OrderedDict[str, Any]] = []

# Decode nearby targets in one forward pass.  This covers the four consecutive
# tactile frames and the 10-frame-spaced future RGB targets without forcing the
# sparse 30-frame RGB history offsets into one long decode range.
_MAX_SEQUENTIAL_DECODE_GAP = max(
    later - earlier
    for earlier, later in zip(FUTURE_IMAGE_OFFSETS[:-1], FUTURE_IMAGE_OFFSETS[1:], strict=True)
)


def _group_nearby_frame_indices(
    frame_indices: tuple[int, ...],
    *,
    max_gap: int = _MAX_SEQUENTIAL_DECODE_GAP,
) -> tuple[tuple[int, ...], ...]:
    """Partition sorted unique frame indices into short sequential decode runs."""
    if max_gap < 1:
        raise ValueError(f"max_gap must be positive, got {max_gap}")
    unique_indices = sorted(set(frame_indices))
    if any(frame_idx < 0 for frame_idx in unique_indices):
        raise ValueError(f"frame indices must be non-negative, got {frame_indices}")
    if not unique_indices:
        return ()

    groups: list[list[int]] = [[unique_indices[0]]]
    for frame_idx in unique_indices[1:]:
        if frame_idx - groups[-1][-1] <= max_gap:
            groups[-1].append(frame_idx)
        else:
            groups.append([frame_idx])
    return tuple(tuple(group) for group in groups)


def build_observation_relative_action_chunk(
    raw_actions: np.ndarray,
    raw_proprio: np.ndarray,
    *,
    relative_step_idx: int,
    chunk_size: int,
    gripper_start_idx: int,
) -> np.ndarray:
    """Build one action chunk relative to the observation at its start.

    The dual-arm TCP targets use one common base for the entire chunk:
    ``action[t + k, :18] - observation.state[t, :18]``. Gripper targets remain
    absolute. End-of-episode padding matches ``get_action_chunk_with_padding``.
    """
    if raw_actions.shape != raw_proprio.shape:
        raise ValueError(f"action/proprio shapes must match, got {raw_actions.shape} and {raw_proprio.shape}")
    if raw_actions.ndim != 2:
        raise ValueError(f"action/proprio arrays must be 2D, got {raw_actions.ndim}D")
    if not 0 <= relative_step_idx < len(raw_actions):
        raise IndexError(f"relative_step_idx {relative_step_idx} is outside [0, {len(raw_actions)})")
    if not 0 <= gripper_start_idx <= raw_actions.shape[1]:
        raise ValueError(f"gripper_start_idx {gripper_start_idx} is outside [0, {raw_actions.shape[1]}]")

    chunk = get_action_chunk_with_padding(
        actions=raw_actions,
        relative_step_idx=relative_step_idx,
        chunk_size=chunk_size,
        num_steps=len(raw_actions),
    ).astype(np.float32, copy=True)
    chunk[:, :gripper_start_idx] -= raw_proprio[relative_step_idx, :gripper_start_idx]
    return chunk


@dataclass(frozen=True)
class _VideoRef:
    key: str
    chunk_index: int
    file_index: int
    from_frame: int


@dataclass(frozen=True)
class _EpisodeRef:
    episode_index: int
    length: int
    data_chunk_index: int
    data_file_index: int
    command: str
    videos: dict[str, _VideoRef]


class LeRobotBiFlexivDataset(Dataset):
    """Direct LeRobot parquet/mp4 dataset for the dual-arm bi_flexiv platform."""

    NUM_LATENT_SLOTS = LAYOUT_STATE_T
    NUM_CONDITIONAL_SLOTS = LAYOUT_NUM_CONDITIONAL_FRAMES
    PIXEL_FRAMES = LAYOUT_PIXEL_FRAMES

    VISION_KEYS = (
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )
    TACTILE_KEYS = (
        "observation.images.left_tactile_0",
        "observation.images.left_tactile_1",
        "observation.images.right_tactile_0",
        "observation.images.right_tactile_1",
    )
    VIDEO_KEYS = VISION_KEYS + TACTILE_KEYS

    # Latent slot layout.
    CURRENT_PROPRIO_IDX = LAYOUT_CURRENT_PROPRIO_IDX
    CURRENT_HEAD_IDX = LAYOUT_CURRENT_HEAD_IDX
    CURRENT_LEFT_WRIST_IDX = LAYOUT_CURRENT_LEFT_WRIST_IDX
    CURRENT_RIGHT_WRIST_IDX = LAYOUT_CURRENT_RIGHT_WRIST_IDX
    CURRENT_LEFT_TACTILE_IDX = LAYOUT_CURRENT_LEFT_TACTILE_IDX
    CURRENT_RIGHT_TACTILE_IDX = LAYOUT_CURRENT_RIGHT_TACTILE_IDX
    ACTION_IDX = LAYOUT_ACTION_LATENT_IDX
    FUTURE_HEAD_IDX = LAYOUT_FUTURE_HEAD_IDX
    FUTURE_LEFT_WRIST_IDX = LAYOUT_FUTURE_LEFT_WRIST_IDX
    FUTURE_RIGHT_WRIST_IDX = LAYOUT_FUTURE_RIGHT_WRIST_IDX

    def __init__(
        self,
        data_dir: str,
        is_train: bool = True,
        chunk_size: int = ACTION_CHUNK_SIZE,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        normalization_mode: Literal["q99", "min_max"] = "q99",
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        use_wrist_images: bool = True,
        use_third_person_images: bool = True,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        rollout_data_dir: str = "",
        demonstration_sampling_prob: float = 1.0,
        success_rollout_sampling_prob: float = 0.0,
        treat_success_rollouts_as_demos: bool = False,
        return_value_function_returns: bool = False,
        gamma: float = 0.99,
        gripper_start_idx: int = 18,
        max_open_videos: int = 16,
        max_episodes: int | None = None,
    ):
        del (
            is_train,
            use_wrist_images,
            use_third_person_images,
            rollout_data_dir,
            demonstration_sampling_prob,
            success_rollout_sampling_prob,
            treat_success_rollouts_as_demos,
            return_value_function_returns,
            gamma,
        )
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        if self.chunk_size != ACTION_CHUNK_SIZE:
            raise ValueError(f"The history layout requires chunk_size={ACTION_CHUNK_SIZE}, got {self.chunk_size}")
        self.final_image_size = final_image_size
        if self.final_image_size != 224:
            raise ValueError(f"The merged-tactile layout requires final_image_size=224, got {self.final_image_size}")
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        if normalization_mode not in ("q99", "min_max"):
            raise ValueError(f"Unsupported normalization mode: {normalization_mode!r}")
        self.normalization_mode = normalization_mode
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        if self.num_duplicates_per_image != 4:
            raise ValueError(
                "The WAN temporal layout requires num_duplicates_per_image=4, "
                f"got {self.num_duplicates_per_image}"
            )
        self.gripper_start_idx = gripper_start_idx
        self.max_open_videos = max_open_videos

        self._data_file_cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self._episode_array_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # PyAV/libdav1d is required here.  OpenCV's bundled FFmpeg can open the
        # LeRobot AV1 containers but fails on random seeks with "Missing
        # Sequence Header", which makes shuffled training crash nondeterministically.
        self._video_container_cache: OrderedDict[str, Any] = OrderedDict()
        self._video_cache_pid = os.getpid()

        self.info = self._load_info()
        self.fps = int(self.info.get("fps", 30))
        self.task_by_index = self._load_tasks()
        self.episodes = self._load_episode_refs(max_episodes=max_episodes)
        if os.environ.get("DEBUGGING", "False").lower() == "true":
            self.episodes = self.episodes[:1]
        self.num_episodes = len(self.episodes)
        self.num_steps = sum(ep.length for ep in self.episodes)
        self.epoch_length = self.num_steps
        self._episode_starts = np.cumsum([0] + [ep.length for ep in self.episodes], dtype=np.int64)

        if t5_text_embeddings_path:
            with open(t5_text_embeddings_path, "rb") as file:
                self.t5_text_embeddings = pickle.load(file)
        else:
            self.t5_text_embeddings = {}

        self.dataset_stats = (
            self._load_or_compute_dataset_statistics() if self.normalize_actions or self.normalize_proprio else {}
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # AV containers are process-local and cannot be pickled into DataLoader
        # workers.  Every worker lazily opens its own bounded cache.
        state["_video_container_cache"] = OrderedDict()
        state["_video_cache_pid"] = None
        return state

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        global_step_idx = idx % self.num_steps
        episode_list_idx = int(np.searchsorted(self._episode_starts, global_step_idx, side="right") - 1)
        relative_step_idx = int(global_step_idx - self._episode_starts[episode_list_idx])
        episode = self.episodes[episode_list_idx]
        raw_actions, raw_proprio = self._get_episode_arrays(episode)
        rgb_history_indices = clamped_relative_indices(relative_step_idx, RGB_HISTORY_OFFSETS, episode.length)
        tactile_history_indices = clamped_relative_indices(
            relative_step_idx,
            TACTILE_HISTORY_OFFSETS,
            episode.length,
        )
        future_indices = clamped_relative_indices(relative_step_idx, FUTURE_IMAGE_OFFSETS, episode.length)

        action_chunk = build_observation_relative_action_chunk(
            raw_actions,
            raw_proprio,
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            gripper_start_idx=self.gripper_start_idx,
        )
        if self.normalize_actions:
            action_chunk = self._rescale_array(
                action_chunk,
                self.dataset_stats,
                "actions",
                normalization_mode=self.normalization_mode,
            )

        normalized_proprio = raw_proprio
        if self.normalize_proprio:
            normalized_proprio = self._rescale_array(
                raw_proprio,
                self.dataset_stats,
                "proprio",
                normalization_mode=self.normalization_mode,
            )

        history_frames, future_frames = self._read_history_and_future_frames(
            episode,
            rgb_history_indices=rgb_history_indices,
            tactile_history_indices=tactile_history_indices,
            future_indices=future_indices,
        )

        left_gate, right_gate = self._compute_per_arm_tactile_gate(relative_step_idx, history_frames)
        left_tactile = np.stack(
            [
                merge_tactile_pair_vertical(first, second)
                for first, second in zip(
                    history_frames["observation.images.left_tactile_0"],
                    history_frames["observation.images.left_tactile_1"],
                    strict=True,
                )
            ],
            axis=0,
        )
        right_tactile = np.stack(
            [
                merge_tactile_pair_vertical(first, second)
                for first, second in zip(
                    history_frames["observation.images.right_tactile_0"],
                    history_frames["observation.images.right_tactile_1"],
                    strict=True,
                )
            ],
            axis=0,
        )
        condition_images = {
            "head": self._resize_sequence_for_stack(history_frames["observation.images.head"]),
            "left_wrist": self._resize_sequence_for_stack(history_frames["observation.images.left_wrist"]),
            "right_wrist": self._resize_sequence_for_stack(history_frames["observation.images.right_wrist"]),
            "left_tactile_merged": self._resize_sequence_for_stack(left_tactile),
            "right_tactile_merged": self._resize_sequence_for_stack(right_tactile),
        }
        future_images = {
            key.removeprefix("observation.images."): self._resize_sequence_for_stack(value)
            for key, value in future_frames.items()
        }
        pixel_frames = build_pixel_frame_sequence(condition_images, future_images)
        images = preprocess_image(
            pixel_frames,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )
        if images.shape[1] != self.PIXEL_FRAMES:
            raise RuntimeError(f"Expected {self.PIXEL_FRAMES} pixel frames, got {images.shape[1]}")

        if episode.command not in self.t5_text_embeddings:
            raise KeyError(
                f"Missing T5 embedding for command {episode.command!r}. "
                "Run cosmos_policy.datasets.save_lerobot_t5_text_embeddings first."
            )

        return {
            "video": images,
            "command": episode.command,
            "actions": action_chunk,
            "t5_text_embeddings": torch.squeeze(self.t5_text_embeddings[episode.command]),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": normalized_proprio[relative_step_idx]
            if self.use_proprio
            else np.zeros_like(normalized_proprio[relative_step_idx]),
            "__key__": idx,
            "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "action_latent_idx": self.ACTION_IDX,
            "value_latent_idx": -1,
            "current_proprio_latent_idx": self.CURRENT_PROPRIO_IDX if self.use_proprio else -1,
            "current_wrist_image_latent_idx": self.CURRENT_LEFT_WRIST_IDX,
            "current_wrist_image2_latent_idx": self.CURRENT_RIGHT_WRIST_IDX,
            "current_image_latent_idx": self.CURRENT_HEAD_IDX,
            "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": self.FUTURE_LEFT_WRIST_IDX,
            "future_wrist_image2_latent_idx": self.FUTURE_RIGHT_WRIST_IDX,
            "future_image_latent_idx": self.FUTURE_HEAD_IDX,
            "tactile_self_attn_gate": torch.tensor([left_gate, right_gate], dtype=torch.float32),
        }

    def get_inference_sample(self, episode_index: int, relative_step_idx: int) -> dict[str, Any]:
        """Return one raw online-style observation and its four future-frame targets.

        ``episode_index`` is the LeRobot episode id from metadata, not its list
        position. Images remain RGB uint8 at their stored resolution; the policy
        applies the same resize/crop path used by the online server.
        """
        episode = next((item for item in self.episodes if item.episode_index == episode_index), None)
        if episode is None:
            available = [item.episode_index for item in self.episodes[:20]]
            raise KeyError(f"Episode {episode_index} was not found. First available ids: {available}")
        if not 0 <= relative_step_idx < episode.length:
            raise IndexError(
                f"relative_step_idx {relative_step_idx} is outside episode {episode_index} length {episode.length}"
            )

        _, raw_proprio = self._get_episode_arrays(episode)
        rgb_history_indices = clamped_relative_indices(relative_step_idx, RGB_HISTORY_OFFSETS, episode.length)
        tactile_history_indices = clamped_relative_indices(
            relative_step_idx,
            TACTILE_HISTORY_OFFSETS,
            episode.length,
        )
        future_indices = clamped_relative_indices(relative_step_idx, FUTURE_IMAGE_OFFSETS, episode.length)
        history_frames, future_frames = self._read_history_and_future_frames(
            episode,
            rgb_history_indices=rgb_history_indices,
            tactile_history_indices=tactile_history_indices,
            future_indices=future_indices,
        )
        left_gate, right_gate = self._compute_per_arm_tactile_gate(relative_step_idx, history_frames)

        def _short_names(frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            return {key.removeprefix("observation.images."): value for key, value in frames.items()}

        return {
            "observation": {
                "observation_seq": relative_step_idx,
                "state": np.ascontiguousarray(raw_proprio[relative_step_idx], dtype=np.float32),
                "images": _short_names(history_frames),
                "tactile_self_attn_gate": np.asarray([left_gate, right_gate], dtype=np.float32),
                "prompt": episode.command,
            },
            "future_images": _short_names(future_frames),
            "episode_index": episode.episode_index,
            "start_timestep": relative_step_idx,
            "rgb_history_timesteps": rgb_history_indices,
            "tactile_history_timesteps": tactile_history_indices,
            "future_timesteps": future_indices,
            "future_timestep": future_indices[-1],
            "is_padded_future": any(
                actual != relative_step_idx + offset
                for actual, offset in zip(future_indices, FUTURE_IMAGE_OFFSETS, strict=True)
            ),
        }

    def close(self) -> None:
        """Release cached PyAV video containers."""
        for container in getattr(self, "_video_container_cache", {}).values():
            container.close()
        if hasattr(self, "_video_container_cache"):
            self._video_container_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _load_info(self) -> dict[str, Any]:
        path = os.path.join(self.data_dir, "meta", "info.json")
        with open(path) as f:
            return json.load(f)

    def _load_tasks(self) -> dict[int, str]:
        path = os.path.join(self.data_dir, "meta", "tasks.parquet")
        df = pd.read_parquet(path)
        text_col = "task" if "task" in df.columns else "tasks" if "tasks" in df.columns else None
        task_by_index = {}
        for row_index, row in df.iterrows():
            command = str(row[text_col]) if text_col is not None else str(row_index)
            task_by_index[int(row["task_index"])] = command
        return task_by_index

    def _load_episode_refs(self, max_episodes: int | None) -> list[_EpisodeRef]:
        paths = sorted(glob(os.path.join(self.data_dir, "meta", "episodes", "chunk-*", "file-*.parquet")))
        if not paths:
            raise FileNotFoundError(f"No LeRobot episode metadata parquet files under {self.data_dir}")
        episodes_df = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)
        if max_episodes is not None:
            episodes_df = episodes_df.iloc[:max_episodes]

        episodes = []
        for _, row in episodes_df.iterrows():
            if "tasks" in row and len(row["tasks"]) > 0:
                command = str(row["tasks"][0])
            else:
                command = self.task_by_index[int(row.get("stats/task_index/min", 0))]
            videos = {}
            for key in self.VIDEO_KEYS:
                prefix = f"videos/{key}"
                videos[key] = _VideoRef(
                    key=key,
                    chunk_index=int(row[f"{prefix}/chunk_index"]),
                    file_index=int(row[f"{prefix}/file_index"]),
                    from_frame=int(round(float(row[f"{prefix}/from_timestamp"]) * self.fps)),
                )
            episodes.append(
                _EpisodeRef(
                    episode_index=int(row["episode_index"]),
                    length=int(row["length"]),
                    data_chunk_index=int(row["data/chunk_index"]),
                    data_file_index=int(row["data/file_index"]),
                    command=command,
                    videos=videos,
                )
            )
        return episodes

    def _data_file_path(self, chunk_index: int, file_index: int) -> str:
        return os.path.join(self.data_dir, "data", f"chunk-{chunk_index:03d}", f"file-{file_index:03d}.parquet")

    def _load_data_file(self, chunk_index: int, file_index: int) -> dict[str, np.ndarray]:
        key = (chunk_index, file_index)
        if key in self._data_file_cache:
            return self._data_file_cache[key]
        df = pd.read_parquet(
            self._data_file_path(chunk_index, file_index),
            columns=["action", "observation.state", "episode_index"],
        )
        entry = {
            "actions": np.stack(df["action"].to_numpy()).astype(np.float32),
            "proprio": np.stack(df["observation.state"].to_numpy()).astype(np.float32),
            "episode_index": df["episode_index"].to_numpy(dtype=np.int64),
        }
        self._data_file_cache[key] = entry
        return entry

    def _get_episode_arrays(self, episode: _EpisodeRef) -> tuple[np.ndarray, np.ndarray]:
        if episode.episode_index in self._episode_array_cache:
            return self._episode_array_cache[episode.episode_index]
        data = self._load_data_file(episode.data_chunk_index, episode.data_file_index)
        mask = data["episode_index"] == episode.episode_index
        actions = data["actions"][mask]
        proprio = data["proprio"][mask]
        if len(actions) != episode.length:
            raise ValueError(
                f"Episode {episode.episode_index} length mismatch: metadata={episode.length}, data={len(actions)}"
            )
        self._episode_array_cache[episode.episode_index] = (actions, proprio)
        return actions, proprio

    def _compute_observation_relative_action_statistics(self) -> dict[str, np.ndarray]:
        """Compute exact per-dimension stats over all training action chunks."""
        first_actions, _ = self._get_episode_arrays(self.episodes[0])
        action_dim = first_actions.shape[1]
        stats = {
            name: np.empty(action_dim, dtype=np.float32)
            for name in ("min", "max", "mean", "std", "median", "q01", "q99")
        }
        horizon_offsets = np.arange(self.chunk_size, dtype=np.int64)

        for dim in tqdm(range(action_dim), desc="Computing observation-relative action statistics"):
            per_episode_values = []
            for episode in self.episodes:
                raw_actions, raw_proprio = self._get_episode_arrays(episode)
                start_indices = np.arange(episode.length, dtype=np.int64)[:, None]
                target_indices = np.minimum(start_indices + horizon_offsets[None, :], episode.length - 1)
                values = raw_actions[target_indices, dim].astype(np.float32, copy=True)
                if dim < self.gripper_start_idx:
                    values -= raw_proprio[:, None, dim]
                per_episode_values.append(values.reshape(-1))

            values = np.concatenate(per_episode_values)
            q01, median, q99 = np.quantile(values, (0.01, 0.5, 0.99))
            stats["min"][dim] = values.min()
            stats["max"][dim] = values.max()
            stats["mean"][dim] = values.mean(dtype=np.float64)
            stats["std"][dim] = values.std(dtype=np.float64)
            stats["median"][dim] = median
            stats["q01"][dim] = q01
            stats["q99"][dim] = q99

        return {f"actions_{name}": value for name, value in stats.items()}

    def _load_or_compute_dataset_statistics(self) -> dict[str, np.ndarray]:
        stats_path = os.path.join(self.data_dir, "dataset_statistics_lerobot_bi_flexiv_chunk40.json")
        stats_load_path = stats_path
        if os.path.exists(stats_load_path):
            with open(stats_load_path) as f:
                raw_stats = json.load(f)
            required_suffixes = ("q01", "q99") if self.normalization_mode == "q99" else ("min", "max")
            required_keys = {
                f"{data_key}_{suffix}" for data_key in ("actions", "proprio") for suffix in required_suffixes
            }
            missing_keys = sorted(required_keys.difference(raw_stats))
            stats_chunk_size = int(raw_stats.get("action_chunk_size", -1))
            stats_gripper_start_idx = int(raw_stats.get("gripper_start_idx", -1))
            if (
                not missing_keys
                and stats_chunk_size == self.chunk_size
                and stats_gripper_start_idx == self.gripper_start_idx
            ):
                print(f"Loaded dataset statistics from: {stats_load_path}")
                return {key: np.array(value, dtype=np.float32) for key, value in raw_stats.items()}
            print(
                f"Dataset statistics at {stats_load_path} are incompatible: missing={missing_keys}, "
                f"action_chunk_size={stats_chunk_size}, gripper_start_idx={stats_gripper_start_idx}; recomputing for "
                f"observation-relative chunks with normalization_mode={self.normalization_mode!r}."
            )

        all_proprio = []
        for episode in tqdm(self.episodes, desc="Computing LeRobot bi_flexiv statistics"):
            _, raw_proprio = self._get_episode_arrays(episode)
            all_proprio.append(raw_proprio)
        proprio = np.concatenate(all_proprio, axis=0)
        stats = self._compute_observation_relative_action_statistics()
        stats.update(
            {
                "proprio_min": proprio.min(axis=0),
                "proprio_max": proprio.max(axis=0),
                "proprio_mean": proprio.mean(axis=0),
                "proprio_std": proprio.std(axis=0),
                "proprio_median": np.median(proprio, axis=0),
                "proprio_q01": np.quantile(proprio, 0.01, axis=0).astype(np.float32),
                "proprio_q99": np.quantile(proprio, 0.99, axis=0).astype(np.float32),
                "action_chunk_size": np.asarray(self.chunk_size, dtype=np.float32),
                "gripper_start_idx": np.asarray(self.gripper_start_idx, dtype=np.float32),
            }
        )
        json_stats = {key: value.tolist() for key, value in stats.items()}
        temp_stats_path = f"{stats_path}.tmp.{os.getpid()}"
        try:
            with open(temp_stats_path, "w") as f:
                json.dump(json_stats, f, indent=4)
            os.replace(temp_stats_path, stats_path)
        finally:
            if os.path.exists(temp_stats_path):
                os.remove(temp_stats_path)
        print(f"Dataset statistics saved to: {stats_path}")
        return stats

    @staticmethod
    def _rescale_array(
        arr: np.ndarray,
        stats: dict[str, np.ndarray],
        data_key: str,
        *,
        normalization_mode: Literal["q99", "min_max"] = "q99",
    ) -> np.ndarray:
        if normalization_mode == "q99":
            low = stats[f"{data_key}_q01"]
            high = stats[f"{data_key}_q99"]
            valid = (high - low) >= 1e-6
            normalized = np.zeros_like(arr, dtype=np.float32)
            normalized[..., valid] = 2.0 * ((arr[..., valid] - low[valid]) / (high[valid] - low[valid])) - 1.0
            # Match the bundled GR00T q99 normalizer for constant dimensions.
            normalized[..., ~valid] = arr[..., ~valid]
            return np.clip(normalized, -1.0, 1.0).astype(np.float32)
        if normalization_mode == "min_max":
            low = stats[f"{data_key}_min"]
            high = stats[f"{data_key}_max"]
            denom = np.where((high - low) < 1e-6, 1.0, high - low)
            return (2.0 * ((arr - low) / denom) - 1.0).astype(np.float32)
        raise ValueError(f"Unsupported normalization mode: {normalization_mode!r}")

    def _resize_frame_for_stack(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[0] == self.final_image_size and frame.shape[1] == self.final_image_size:
            return frame
        return cv2.resize(frame, (self.final_image_size, self.final_image_size), interpolation=cv2.INTER_AREA)

    def _resize_sequence_for_stack(self, frames: np.ndarray) -> np.ndarray:
        return np.stack([self._resize_frame_for_stack(frame) for frame in frames], axis=0)

    def _video_path(self, ref: _VideoRef) -> str:
        return os.path.join(
            self.data_dir,
            "videos",
            ref.key,
            f"chunk-{ref.chunk_index:03d}",
            f"file-{ref.file_index:03d}.mp4",
        )

    def _cache_container(self, path: str, container: Any) -> Any:
        self._video_container_cache[path] = container
        if len(self._video_container_cache) > self.max_open_videos:
            _old_path, old_container = self._video_container_cache.popitem(last=False)
            old_container.close()
        return container

    def _drop_container(self, path: str) -> None:
        container = self._video_container_cache.pop(path, None)
        if container is not None:
            container.close()

    def _open_container_with_retry(self, path: str, retries: int = 5) -> Any:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                container = av.open(path, mode="r")
                if not container.streams.video:
                    container.close()
                    raise ValueError("container has no video stream")
                return container
            except Exception as error:
                last_error = error
                time.sleep(min(0.25 * (attempt + 1), 1.0))
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else -1
        raise ValueError(
            f"Could not open video file after {retries} retries: {path} "
            f"exists={exists} size={size} ({last_error})"
        ) from last_error

    def _get_container(self, path: str) -> Any:
        current_pid = os.getpid()
        if self._video_cache_pid != current_pid:
            # DataLoader normally forks workers on Linux.  A container opened
            # in the parent must never be reused by a child process because
            # libavformat decoder state and file offsets are not fork-safe.
            if self._video_container_cache:
                _FORK_INHERITED_VIDEO_CACHES.append(self._video_container_cache)
            self._video_container_cache = OrderedDict()
            self._video_cache_pid = current_pid
        container = self._video_container_cache.get(path)
        if container is not None:
            self._video_container_cache.move_to_end(path)
            return container
        return self._cache_container(path, self._open_container_with_retry(path))

    def _decode_frames(
        self,
        container: Any,
        frame_indices: tuple[int, ...],
        path: str,
    ) -> dict[int, np.ndarray]:
        """Decode sorted unique targets after one seek to the first frame."""
        if not frame_indices:
            return {}
        if frame_indices != tuple(sorted(set(frame_indices))):
            raise ValueError(f"frame_indices must be sorted and unique, got {frame_indices}")

        stream = container.streams.video[0]
        if stream.time_base is None:
            raise ValueError(f"Video stream has no time base: {path}")
        start_pts = int(stream.start_time or 0)
        time_base = float(stream.time_base)
        target_pts = start_pts + int(round((frame_indices[0] / self.fps) / time_base))
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)

        decoded_frames: dict[int, np.ndarray] = {}
        target_position = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            decoded_idx = int(round((int(frame.pts) - start_pts) * time_base * self.fps))
            target_idx = frame_indices[target_position]
            if decoded_idx < target_idx:
                continue
            if decoded_idx != target_idx:
                raise ValueError(
                    f"Decode skipped requested frame {target_idx} and reached {decoded_idx} in {path}"
                )
            decoded_frames[target_idx] = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
            target_position += 1
            if target_position == len(frame_indices):
                return decoded_frames
        missing = frame_indices[target_position:]
        raise ValueError(f"Decoder reached EOF before frames {missing} in {path}")

    def _decode_frame(self, container: Any, frame_idx: int, path: str) -> np.ndarray:
        return self._decode_frames(container, (frame_idx,), path)[frame_idx]

    def _read_frame(self, episode: _EpisodeRef, video_key: str, relative_step_idx: int) -> np.ndarray:
        ref = episode.videos[video_key]
        frame_idx = ref.from_frame + relative_step_idx
        path = self._video_path(ref)
        for attempt in range(2):
            try:
                return self._decode_frame(self._get_container(path), frame_idx, path)
            except Exception:
                self._drop_container(path)
                if attempt == 1:
                    raise
            time.sleep(0.1 * (attempt + 1))
        raise AssertionError("unreachable")

    def _read_frames(
        self,
        episode: _EpisodeRef,
        video_key: str,
        relative_step_indices: tuple[int, ...],
    ) -> np.ndarray:
        """Read targets with one seek per nearby group and preserve duplicates/order."""
        if not relative_step_indices:
            raise ValueError("relative_step_indices must not be empty")

        ref = episode.videos[video_key]
        path = self._video_path(ref)
        decoded_frames: dict[int, np.ndarray] = {}
        for relative_group in _group_nearby_frame_indices(relative_step_indices):
            absolute_group = tuple(ref.from_frame + step_idx for step_idx in relative_group)
            for attempt in range(2):
                try:
                    absolute_frames = self._decode_frames(self._get_container(path), absolute_group, path)
                    decoded_frames.update(
                        {
                            relative_idx: absolute_frames[absolute_idx]
                            for relative_idx, absolute_idx in zip(relative_group, absolute_group, strict=True)
                        }
                    )
                    break
                except Exception:
                    self._drop_container(path)
                    if attempt == 1:
                        raise
                time.sleep(0.1 * (attempt + 1))

        return np.stack([decoded_frames[step_idx] for step_idx in relative_step_indices], axis=0)

    def _read_history_and_future_frames(
        self,
        episode: _EpisodeRef,
        *,
        rgb_history_indices: tuple[int, ...],
        tactile_history_indices: tuple[int, ...],
        future_indices: tuple[int, ...],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Read each RGB stream once for its combined history/future targets."""
        num_history_frames = len(rgb_history_indices)
        combined_rgb_indices = rgb_history_indices + future_indices
        combined_rgb_frames = {
            key: self._read_frames(episode, key, combined_rgb_indices) for key in self.VISION_KEYS
        }
        history_frames = {
            key: frames[:num_history_frames] for key, frames in combined_rgb_frames.items()
        }
        history_frames.update(
            {key: self._read_frames(episode, key, tactile_history_indices) for key in self.TACTILE_KEYS}
        )
        future_frames = {
            key: frames[num_history_frames:] for key, frames in combined_rgb_frames.items()
        }
        return history_frames, future_frames

    def _compute_per_arm_tactile_gate(
        self,
        relative_step_idx: int,
        history_frames: dict[str, np.ndarray],
    ) -> tuple[float, float]:
        if relative_step_idx == 0:
            return scalar_gate_from_raw(0.0), scalar_gate_from_raw(0.0)

        def _diff(key: str) -> float:
            history = history_frames[key]
            if history.shape[0] != len(TACTILE_HISTORY_OFFSETS):
                raise ValueError(
                    f"Tactile history {key!r} must contain {len(TACTILE_HISTORY_OFFSETS)} frames, "
                    f"got {history.shape}"
                )
            prev = history[-2]
            curr = history[-1]
            return float(np.abs(curr.astype(np.float32) - prev.astype(np.float32)).mean() / 255.0)

        left_raw = max(_diff("observation.images.left_tactile_0"), _diff("observation.images.left_tactile_1"))
        right_raw = max(_diff("observation.images.right_tactile_0"), _diff("observation.images.right_tactile_1"))
        return scalar_gate_from_raw(left_raw), scalar_gate_from_raw(right_raw)
