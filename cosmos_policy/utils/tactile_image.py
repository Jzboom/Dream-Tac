# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared tactile-image preprocessing for the bi_flexiv policy."""

from __future__ import annotations

import cv2
import numpy as np


TACTILE_OUTPUT_HEIGHT = 224
TACTILE_CONTENT_WIDTH = 196
TACTILE_HORIZONTAL_PADDING = 14


def merge_tactile_pair_vertical(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Stack one gripper's raw tactile pair, resize it, and pad to 224 x 224.

    ``first`` is placed above ``second``. The stacked image is resized to
    224 x 196 with area interpolation, then padded with 14 black pixels on
    both horizontal sides.
    """
    first = np.asarray(first)
    second = np.asarray(second)
    for name, image in (("first", first), ("second", second)):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{name} tactile image must be HWC RGB, got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"{name} tactile image must have dtype uint8, got {image.dtype}")
    if first.shape != second.shape:
        raise ValueError(
            "Tactile images from one gripper must have identical shapes, "
            f"got {first.shape} and {second.shape}"
        )

    stacked = np.concatenate((first, second), axis=0)
    resized = cv2.resize(
        stacked,
        (TACTILE_CONTENT_WIDTH, TACTILE_OUTPUT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    padded = cv2.copyMakeBorder(
        resized,
        0,
        0,
        TACTILE_HORIZONTAL_PADDING,
        TACTILE_HORIZONTAL_PADDING,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    expected_shape = (TACTILE_OUTPUT_HEIGHT, TACTILE_OUTPUT_HEIGHT, 3)
    if padded.shape != expected_shape:
        raise RuntimeError(f"Unexpected merged tactile shape: {padded.shape}, expected {expected_shape}")
    return np.ascontiguousarray(padded)
