# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LeRobot v3 loader for the dual-arm bi_flexiv platform.

The sequence uses 11 latent slots:
blank, proprio, 3 current RGB views, 2 merged current tactile views, action,
and 3 future RGB views. With the WAN2.1 temporal compression factor of 4
this becomes 41 pixel frames.
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
from cosmos_policy.utils.tactile_image import merge_tactile_pair_vertical
from cosmos_policy.utils.tactile_self_attn_gate import scalar_gate_from_raw


# Never touch a decoder inherited across ``fork``.  libavcodec may have worker
# threads and locks that no longer exist in the child, so even calling close()
# can deadlock.  Keep the stale Python objects alive until multiprocessing
# terminates the worker process; the OS then releases their file descriptors.
_FORK_INHERITED_VIDEO_CACHES: list[OrderedDict[str, Any]] = []


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

    NUM_LATENT_SLOTS = 11
    NUM_CONDITIONAL_SLOTS = 7
    PIXEL_FRAMES = 41

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
    CURRENT_PROPRIO_IDX = 1
    CURRENT_HEAD_IDX = 2
    CURRENT_LEFT_WRIST_IDX = 3
    CURRENT_RIGHT_WRIST_IDX = 4
    CURRENT_LEFT_TACTILE_IDX = 5
    CURRENT_RIGHT_TACTILE_IDX = 6
    ACTION_IDX = 7
    FUTURE_HEAD_IDX = 8
    FUTURE_LEFT_WRIST_IDX = 9
    FUTURE_RIGHT_WRIST_IDX = 10

    def __init__(
        self,
        data_dir: str,
        is_train: bool = True,
        chunk_size: int = 30,
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
        future_frame_idx = min(relative_step_idx + self.chunk_size, episode.length - 1)

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

        current_frames = {key: self._read_frame(episode, key, relative_step_idx) for key in self.VIDEO_KEYS}
        future_frames = {key: self._read_frame(episode, key, future_frame_idx) for key in self.VISION_KEYS}

        left_gate, right_gate = self._compute_per_arm_tactile_gate(episode, relative_step_idx, current_frames)
        left_tactile = merge_tactile_pair_vertical(
            current_frames["observation.images.left_tactile_0"],
            current_frames["observation.images.left_tactile_1"],
        )
        right_tactile = merge_tactile_pair_vertical(
            current_frames["observation.images.right_tactile_0"],
            current_frames["observation.images.right_tactile_1"],
        )

        blank = np.zeros_like(current_frames["observation.images.head"])
        unique_frames = [
            blank,
            blank,
            current_frames["observation.images.head"],
            current_frames["observation.images.left_wrist"],
            current_frames["observation.images.right_wrist"],
            left_tactile,
            right_tactile,
            blank,
            future_frames["observation.images.head"],
            future_frames["observation.images.left_wrist"],
            future_frames["observation.images.right_wrist"],
        ]
        if len(unique_frames) != self.NUM_LATENT_SLOTS:
            raise RuntimeError(f"Expected {self.NUM_LATENT_SLOTS} latent slots, got {len(unique_frames)}")
        repeats = [1] + [self.num_duplicates_per_image] * (len(unique_frames) - 1)
        unique_frames = [self._resize_frame_for_stack(frame) for frame in unique_frames]
        images = preprocess_image(
            np.stack(unique_frames, axis=0),
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )
        images = torch.repeat_interleave(images, torch.as_tensor(repeats, dtype=torch.long), dim=1)
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
        """Return one raw online-style observation and its future-frame GT.

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
        future_step_idx = min(relative_step_idx + self.chunk_size, episode.length - 1)
        current_frames = {key: self._read_frame(episode, key, relative_step_idx) for key in self.VIDEO_KEYS}
        future_frames = {key: self._read_frame(episode, key, future_step_idx) for key in self.VISION_KEYS}
        left_gate, right_gate = self._compute_per_arm_tactile_gate(episode, relative_step_idx, current_frames)

        def _short_names(frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            return {key.removeprefix("observation.images."): value for key, value in frames.items()}

        return {
            "observation": {
                "observation_seq": relative_step_idx,
                "state": np.ascontiguousarray(raw_proprio[relative_step_idx], dtype=np.float32),
                "images": _short_names(current_frames),
                "tactile_self_attn_gate": np.asarray([left_gate, right_gate], dtype=np.float32),
                "prompt": episode.command,
            },
            "future_images": _short_names(future_frames),
            "episode_index": episode.episode_index,
            "start_timestep": relative_step_idx,
            "future_timestep": future_step_idx,
            "is_padded_future": future_step_idx != relative_step_idx + self.chunk_size,
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
        stats_path = os.path.join(self.data_dir, "dataset_statistics_lerobot_bi_flexiv.json")
        legacy_stats_path = os.path.join(self.data_dir, "dataset_statistics_lerobot_earbud.json")
        stats_load_path = stats_path if os.path.exists(stats_path) else legacy_stats_path
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

    def _decode_frame(self, container: Any, frame_idx: int, path: str) -> np.ndarray:
        stream = container.streams.video[0]
        if stream.time_base is None:
            raise ValueError(f"Video stream has no time base: {path}")
        start_pts = int(stream.start_time or 0)
        time_base = float(stream.time_base)
        target_pts = start_pts + int(round((frame_idx / self.fps) / time_base))
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            decoded_idx = int(round((int(frame.pts) - start_pts) * time_base * self.fps))
            if decoded_idx < frame_idx:
                continue
            if decoded_idx != frame_idx:
                raise ValueError(
                    f"Seek skipped requested frame {frame_idx} and reached {decoded_idx} in {path}"
                )
            return np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
        raise ValueError(f"Decoder reached EOF before frame {frame_idx} in {path}")

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

    def _compute_per_arm_tactile_gate(
        self,
        episode: _EpisodeRef,
        relative_step_idx: int,
        current_frames: dict[str, np.ndarray],
    ) -> tuple[float, float]:
        if relative_step_idx == 0:
            return scalar_gate_from_raw(0.0), scalar_gate_from_raw(0.0)
        prev_idx = relative_step_idx - 1

        def _diff(key: str) -> float:
            prev = self._read_frame(episode, key, prev_idx)
            curr = current_frames[key]
            return float(np.abs(curr.astype(np.float32) - prev.astype(np.float32)).mean() / 255.0)

        left_raw = max(_diff("observation.images.left_tactile_0"), _diff("observation.images.left_tactile_1"))
        right_raw = max(_diff("observation.images.right_tactile_0"), _diff("observation.images.right_tactile_1"))
        return scalar_gate_from_raw(left_raw), scalar_gate_from_raw(right_raw)
