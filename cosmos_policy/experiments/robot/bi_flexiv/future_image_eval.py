"""Save side-by-side bi_flexiv future-image prediction comparisons."""

from __future__ import annotations

import os
import re
from typing import Any, Mapping

import cv2
import numpy as np

from cosmos_policy.utils.bi_flexiv_video_layout import FUTURE_IMAGE_OFFSETS, RGB_IMAGE_KEYS


def _safe_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not tag:
        raise ValueError("Future-image sample tag must not be empty")
    return tag


def _as_rgb_sequence_uint8(image: Any, *, camera_name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 4 or array.shape[0] != len(FUTURE_IMAGE_OFFSETS) or array.shape[-1] != 3:
        raise ValueError(
            f"Images {camera_name!r} must have shape ({len(FUTURE_IMAGE_OFFSETS)}, H, W, 3), "
            f"got {array.shape}"
        )
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _write_rgb(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write image: {path}")


def _write_comparison(path: str, gt: np.ndarray, pred: np.ndarray) -> None:
    if gt.shape != pred.shape:
        raise ValueError(f"Prediction/GT shape mismatch: pred={pred.shape}, gt={gt.shape}")
    title_height = 34
    height, width = gt.shape[:2]
    canvas = np.full((height + title_height, width * 2, 3), 255, dtype=np.uint8)
    canvas[title_height:, :width] = gt
    canvas[title_height:, width:] = pred
    cv2.putText(canvas, "Ground Truth", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(canvas, "Dream-Tac", (width + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    _write_rgb(path, canvas)


class FutureImageEvaluationWriter:
    """Write only side-by-side panels; no standalone pred/GT or JSON files."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        self.comparisons_dir = os.path.join(self.output_dir, "comparisons")
        os.makedirs(self.comparisons_dir, exist_ok=True)

    def save_comparison(
        self,
        sample_tag: str,
        predictions: Mapping[str, Any],
        ground_truth: Mapping[str, Any],
    ) -> dict[str, str]:
        tag = _safe_tag(sample_tag)
        comparison_paths: dict[str, str] = {}

        for camera_name in RGB_IMAGE_KEYS:
            if camera_name not in predictions or camera_name not in ground_truth:
                continue
            pred = _as_rgb_sequence_uint8(predictions[camera_name], camera_name=camera_name)
            gt = _as_rgb_sequence_uint8(ground_truth[camera_name], camera_name=camera_name)

            for frame_idx, offset in enumerate(FUTURE_IMAGE_OFFSETS):
                comparison_path = os.path.join(
                    self.comparisons_dir,
                    f"{tag}_{camera_name}_step{offset:03d}.png",
                )
                _write_comparison(comparison_path, gt[frame_idx], pred[frame_idx])
                comparison_paths[f"{camera_name}/t+{offset}"] = os.path.relpath(
                    comparison_path, self.output_dir
                )

        if not comparison_paths:
            raise ValueError("Prediction and GT mappings have no camera keys in common")
        return comparison_paths
