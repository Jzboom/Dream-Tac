# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LeRobot v3 loader for the dual-arm earbud insertion dataset.

The sequence uses 18 latent slots:
blank, proprio, 3 current RGB views, 4 current tactile views, action,
future proprio, 3 future RGB views, 4 future tactile views.
With the WAN2.1 temporal compression factor of 4 this becomes 69 pixel frames.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_policy.datasets.dataset_common import get_action_chunk_with_padding
from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.utils.tactile_self_attn_gate import scalar_gate_from_raw


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


class LeRobotEarbudDataset(Dataset):
    """Direct LeRobot parquet/mp4 dataset for dual-arm earbud insertion."""

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
    CURRENT_LEFT_TACTILE_0_IDX = 5
    CURRENT_LEFT_TACTILE_1_IDX = 6
    CURRENT_RIGHT_TACTILE_0_IDX = 7
    CURRENT_RIGHT_TACTILE_1_IDX = 8
    ACTION_IDX = 9
    FUTURE_PROPRIO_IDX = 10
    FUTURE_HEAD_IDX = 11
    FUTURE_LEFT_WRIST_IDX = 12
    FUTURE_RIGHT_WRIST_IDX = 13
    FUTURE_LEFT_TACTILE_0_IDX = 14
    FUTURE_LEFT_TACTILE_1_IDX = 15
    FUTURE_RIGHT_TACTILE_0_IDX = 16
    FUTURE_RIGHT_TACTILE_1_IDX = 17

    def __init__(
        self,
        data_dir: str,
        is_train: bool = True,
        chunk_size: int = 20,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
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
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.gripper_start_idx = gripper_start_idx
        self.max_open_videos = max_open_videos

        self._data_file_cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self._episode_array_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._video_capture_cache: OrderedDict[str, cv2.VideoCapture] = OrderedDict()

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
        state["_video_capture_cache"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        global_step_idx = idx % self.num_steps
        episode_list_idx = int(np.searchsorted(self._episode_starts, global_step_idx, side="right") - 1)
        relative_step_idx = int(global_step_idx - self._episode_starts[episode_list_idx])
        episode = self.episodes[episode_list_idx]
        raw_actions, raw_proprio = self._get_episode_arrays(episode)
        action_targets = self._to_action_targets(raw_actions)

        if self.normalize_actions:
            action_targets = self._rescale_array(action_targets, self.dataset_stats, "actions")
        if self.normalize_proprio:
            raw_proprio = self._rescale_array(raw_proprio, self.dataset_stats, "proprio")

        future_frame_idx = min(relative_step_idx + self.chunk_size, episode.length - 1)

        current_frames = {key: self._read_frame(episode, key, relative_step_idx) for key in self.VIDEO_KEYS}
        future_frames = {key: self._read_frame(episode, key, future_frame_idx) for key in self.VIDEO_KEYS}

        left_gate, right_gate = self._compute_per_arm_tactile_gate(episode, relative_step_idx, current_frames)

        blank = np.zeros_like(current_frames["observation.images.head"])
        unique_frames = [
            blank,
            blank,
            current_frames["observation.images.head"],
            current_frames["observation.images.left_wrist"],
            current_frames["observation.images.right_wrist"],
            current_frames["observation.images.left_tactile_0"],
            current_frames["observation.images.left_tactile_1"],
            current_frames["observation.images.right_tactile_0"],
            current_frames["observation.images.right_tactile_1"],
            blank,
            blank,
            future_frames["observation.images.head"],
            future_frames["observation.images.left_wrist"],
            future_frames["observation.images.right_wrist"],
            future_frames["observation.images.left_tactile_0"],
            future_frames["observation.images.left_tactile_1"],
            future_frames["observation.images.right_tactile_0"],
            future_frames["observation.images.right_tactile_1"],
        ]
        repeats = [1] + [self.num_duplicates_per_image] * 17
        unique_frames = [self._resize_frame_for_stack(frame) for frame in unique_frames]
        images = preprocess_image(
            np.stack(unique_frames, axis=0),
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )
        images = torch.repeat_interleave(images, torch.as_tensor(repeats, dtype=torch.long), dim=1)

        action_chunk = get_action_chunk_with_padding(
            actions=action_targets,
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode.length,
        )
        next_relative_step_idx = min(relative_step_idx + self.chunk_size, episode.length - 1)
        next_action_chunk = get_action_chunk_with_padding(
            actions=action_targets,
            relative_step_idx=next_relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode.length,
        )

        if episode.command not in self.t5_text_embeddings:
            raise KeyError(
                f"Missing T5 embedding for command {episode.command!r}. "
                "Run cosmos_policy.datasets.save_lerobot_t5_text_embeddings first."
            )

        future_tactile_indices = torch.tensor(
            [
                self.FUTURE_LEFT_TACTILE_0_IDX,
                self.FUTURE_LEFT_TACTILE_1_IDX,
                self.FUTURE_RIGHT_TACTILE_0_IDX,
                self.FUTURE_RIGHT_TACTILE_1_IDX,
            ],
            dtype=torch.long,
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
            "proprio": raw_proprio[relative_step_idx] if self.use_proprio else np.zeros_like(raw_proprio[relative_step_idx]),
            "future_proprio": raw_proprio[future_frame_idx] if self.use_proprio else np.zeros_like(raw_proprio[future_frame_idx]),
            "__key__": idx,
            "value_function_return": float("-100"),
            "next_action_chunk": next_action_chunk,
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
            "future_proprio_latent_idx": self.FUTURE_PROPRIO_IDX if self.use_proprio else -1,
            "future_wrist_image_latent_idx": self.FUTURE_LEFT_WRIST_IDX,
            "future_wrist_image2_latent_idx": self.FUTURE_RIGHT_WRIST_IDX,
            "future_image_latent_idx": self.FUTURE_HEAD_IDX,
            "future_tactile_latent_indices": future_tactile_indices,
            # Compatibility summary metrics for existing two-tactile logging code.
            "future_tactile_left_latent_idx": torch.tensor(self.FUTURE_LEFT_TACTILE_0_IDX, dtype=torch.long),
            "future_tactile_right_latent_idx": torch.tensor(self.FUTURE_RIGHT_TACTILE_0_IDX, dtype=torch.long),
            "tactile_self_attn_gate": torch.tensor([left_gate, right_gate], dtype=torch.float32),
        }

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

    def _to_action_targets(self, raw_actions: np.ndarray) -> np.ndarray:
        targets = np.zeros_like(raw_actions, dtype=np.float32)
        targets[1:, : self.gripper_start_idx] = (
            raw_actions[1:, : self.gripper_start_idx] - raw_actions[:-1, : self.gripper_start_idx]
        )
        targets[:, self.gripper_start_idx :] = raw_actions[:, self.gripper_start_idx :]
        return targets

    def _load_or_compute_dataset_statistics(self) -> dict[str, np.ndarray]:
        stats_path = os.path.join(self.data_dir, "dataset_statistics_lerobot_earbud.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                raw_stats = json.load(f)
            print(f"Loaded dataset statistics from: {stats_path}")
            return {key: np.array(value, dtype=np.float32) for key, value in raw_stats.items()}

        all_actions = []
        all_proprio = []
        for episode in tqdm(self.episodes, desc="Computing LeRobot earbud statistics"):
            raw_actions, raw_proprio = self._get_episode_arrays(episode)
            all_actions.append(self._to_action_targets(raw_actions))
            all_proprio.append(raw_proprio)
        actions = np.concatenate(all_actions, axis=0)
        proprio = np.concatenate(all_proprio, axis=0)
        stats = {
            "actions_min": actions.min(axis=0),
            "actions_max": actions.max(axis=0),
            "actions_mean": actions.mean(axis=0),
            "actions_std": actions.std(axis=0),
            "actions_median": np.median(actions, axis=0),
            "proprio_min": proprio.min(axis=0),
            "proprio_max": proprio.max(axis=0),
            "proprio_mean": proprio.mean(axis=0),
            "proprio_std": proprio.std(axis=0),
            "proprio_median": np.median(proprio, axis=0),
        }
        with open(stats_path, "w") as f:
            json.dump({key: value.tolist() for key, value in stats.items()}, f, indent=4)
        print(f"Dataset statistics saved to: {stats_path}")
        return stats

    @staticmethod
    def _rescale_array(arr: np.ndarray, stats: dict[str, np.ndarray], data_key: str) -> np.ndarray:
        curr_min = stats[f"{data_key}_min"]
        curr_max = stats[f"{data_key}_max"]
        denom = np.where((curr_max - curr_min) < 1e-6, 1.0, curr_max - curr_min)
        return (2.0 * ((arr - curr_min) / denom) - 1.0).astype(np.float32)

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

    def _cache_capture(self, path: str, cap: cv2.VideoCapture) -> cv2.VideoCapture:
        self._video_capture_cache[path] = cap
        if len(self._video_capture_cache) > self.max_open_videos:
            _old_path, old_cap = self._video_capture_cache.popitem(last=False)
            old_cap.release()
        return cap

    def _drop_capture(self, path: str) -> None:
        cap = self._video_capture_cache.pop(path, None)
        if cap is not None:
            cap.release()

    def _open_capture_with_retry(self, path: str, retries: int = 5) -> cv2.VideoCapture:
        last_error = None
        for attempt in range(retries):
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                return cap
            cap.release()
            last_error = f"attempt {attempt + 1}/{retries} failed"
            time.sleep(min(0.25 * (attempt + 1), 1.0))
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else -1
        raise ValueError(f"Could not open video file after {retries} retries: {path} exists={exists} size={size} ({last_error})")

    def _get_capture(self, path: str) -> cv2.VideoCapture:
        cap = self._video_capture_cache.get(path)
        if cap is not None and cap.isOpened():
            self._video_capture_cache.move_to_end(path)
            return cap
        self._drop_capture(path)
        return self._cache_capture(path, self._open_capture_with_retry(path))

    def _read_frame(self, episode: _EpisodeRef, video_key: str, relative_step_idx: int) -> np.ndarray:
        ref = episode.videos[video_key]
        frame_idx = ref.from_frame + relative_step_idx
        path = self._video_path(ref)
        for attempt in range(2):
            cap = self._get_capture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if ret:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._drop_capture(path)
            time.sleep(0.1 * (attempt + 1))
        raise ValueError(
            f"Could not read {video_key} frame {frame_idx} for episode {episode.episode_index} from {path}"
        )

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
