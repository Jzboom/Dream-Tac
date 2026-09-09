# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared temporal layout for the Dream-Tac bi_flexiv policy."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


STATE_T = 11
NUM_CONDITIONAL_FRAMES = 7
ACTION_LATENT_IDX = 7
PIXEL_FRAMES = 41
ACTION_CHUNK_SIZE = 40
TEMPORAL_COMPRESSION_FACTOR = 4

RGB_HISTORY_OFFSETS = (-90, -60, -30, 0)
TACTILE_HISTORY_OFFSETS = (-3, -2, -1, 0)
FUTURE_IMAGE_OFFSETS = (10, 20, 30, 40)
HISTORY_FRAMES = len(RGB_HISTORY_OFFSETS)
if len(TACTILE_HISTORY_OFFSETS) != HISTORY_FRAMES:
    raise ValueError("RGB and tactile histories must contain the same number of frames")

RGB_IMAGE_KEYS = ("head", "left_wrist", "right_wrist")
MERGED_TACTILE_KEYS = ("left_tactile_merged", "right_tactile_merged")
CONDITION_IMAGE_KEYS = RGB_IMAGE_KEYS + MERGED_TACTILE_KEYS

CURRENT_PROPRIO_IDX = 1
CURRENT_HEAD_IDX = 2
CURRENT_LEFT_WRIST_IDX = 3
CURRENT_RIGHT_WRIST_IDX = 4
CURRENT_LEFT_TACTILE_IDX = 5
CURRENT_RIGHT_TACTILE_IDX = 6
FUTURE_HEAD_IDX = 8
FUTURE_LEFT_WRIST_IDX = 9
FUTURE_RIGHT_WRIST_IDX = 10


def clamped_relative_indices(step_idx: int, offsets: tuple[int, ...], episode_length: int) -> tuple[int, ...]:
    """Resolve offsets relative to ``step_idx``, padding with episode endpoints."""
    if episode_length <= 0:
        raise ValueError(f"episode_length must be positive, got {episode_length}")
    if not 0 <= step_idx < episode_length:
        raise IndexError(f"step_idx {step_idx} is outside [0, {episode_length})")
    return tuple(min(max(step_idx + offset, 0), episode_length - 1) for offset in offsets)


def build_pixel_frame_sequence(
    history_images: Mapping[str, np.ndarray],
    future_images: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build the 41-frame THWC sequence consumed by the WAN tokenizer.

    Every condition-image value must contain four chronological frames. RGB
    histories and tactile histories may use different sampling offsets. During
    inference, omitted future RGB frames are replaced by four copies of the
    latest RGB condition frame; those slots are noised and are not conditions.
    """

    missing = [key for key in CONDITION_IMAGE_KEYS if key not in history_images]
    if missing:
        raise ValueError(f"Missing condition image histories: {missing}")

    first = np.asarray(history_images[CONDITION_IMAGE_KEYS[0]])
    if first.ndim != 4 or first.shape[0] != HISTORY_FRAMES or first.shape[-1] != 3:
        raise ValueError(
            f"Image histories must be ({HISTORY_FRAMES}, H, W, 3), got {first.shape} for "
            f"{CONDITION_IMAGE_KEYS[0]!r}"
        )
    frame_shape = first.shape[1:]

    checked_history: dict[str, np.ndarray] = {}
    for key in CONDITION_IMAGE_KEYS:
        value = np.asarray(history_images[key])
        expected = (HISTORY_FRAMES, *frame_shape)
        if value.shape != expected:
            raise ValueError(f"History {key!r} must have shape {expected}, got {value.shape}")
        checked_history[key] = value

    checked_future: dict[str, np.ndarray] = {}
    for key in RGB_IMAGE_KEYS:
        if future_images is None:
            checked_future[key] = np.repeat(checked_history[key][-1:], len(FUTURE_IMAGE_OFFSETS), axis=0)
            continue
        if key not in future_images:
            raise ValueError(f"Missing future RGB frames for {key!r}")
        value = np.asarray(future_images[key])
        expected = (len(FUTURE_IMAGE_OFFSETS), *frame_shape)
        if value.shape != expected:
            raise ValueError(f"Future images {key!r} must have shape {expected}, got {value.shape}")
        checked_future[key] = value

    blank = np.zeros(frame_shape, dtype=first.dtype)
    frames = [blank]
    frames.extend([blank] * TEMPORAL_COMPRESSION_FACTOR)
    for key in CONDITION_IMAGE_KEYS:
        frames.extend(checked_history[key])
    frames.extend([blank] * TEMPORAL_COMPRESSION_FACTOR)
    for key in RGB_IMAGE_KEYS:
        frames.extend(checked_future[key])

    result = np.stack(frames, axis=0)
    if result.shape != (PIXEL_FRAMES, *frame_shape):
        raise RuntimeError(f"Expected {(PIXEL_FRAMES, *frame_shape)}, got {result.shape}")
    return np.ascontiguousarray(result)
