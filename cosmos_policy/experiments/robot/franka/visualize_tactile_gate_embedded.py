# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Gate visualization for RAW tactile HDF5 where tactile_rectify_left/right are stored as
# per-frame JPEG-encoded object arrays inside observations/images (no mp4 sidecars).
# Reuses the same rho->g_t computation and plotting as visualize_tactile_gate.py.
#
# Usage:
#   cd /share/project/yunfan/cosmos-policy
#   uv run --extra cu128 --group libero --python 3.10 python -m \
#     cosmos_policy.experiments.robot.franka.visualize_tactile_gate_embedded \
#     --data_dir /share/project/yunfan/datafinal/home/franka/tactile/data/insert_usb/insert_usb_20260321 \
#     --out_dir  /share/project/yunfan/datafinal/home/franka/tactile/data/insert_usb/insert_usb_20260321/gate_viz

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Tuple

import cv2
import h5py
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cosmos_policy.experiments.robot.franka.visualize_tactile_gate import (
    compute_raw_and_gate_series,
    plot_figures,
    write_summary,
)


def _decode_jpeg_series(ds, resize: int) -> np.ndarray:
    """Decode a (T,) object array of JPEG bytes to uint8 (T, resize, resize, 3)."""
    frames = []
    for i in range(len(ds)):
        buf = np.frombuffer(bytes(ds[i]), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR (H, W, 3)
        if img is None:
            raise ValueError(f"Failed to decode tactile frame {i}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA)
        frames.append(img)
    return np.stack(frames, axis=0).astype(np.uint8)


def load_tactile_rgb_pair(hdf5_path: str, resize: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load left/right tactile uint8 (T,H,W,3) from embedded JPEG object arrays."""
    with h5py.File(hdf5_path, "r") as f:
        imgs = f["observations"]["images"]
        if "tactile_rectify_left" not in imgs or "tactile_rectify_right" not in imgs:
            raise FileNotFoundError(f"No embedded tactile in {hdf5_path}")
        tl = _decode_jpeg_series(imgs["tactile_rectify_left"], resize)
        tr = _decode_jpeg_series(imgs["tactile_rectify_right"], resize)
        task = f.attrs.get("task_name", os.path.splitext(os.path.basename(hdf5_path))[0])
        if isinstance(task, bytes):
            task = task.decode("utf-8")
    t_len = min(len(tl), len(tr))
    return tl[:t_len], tr[:t_len], task


def _list_hdf5(data_dir: str) -> List[str]:
    files = glob.glob(os.path.join(data_dir, "*.hdf5"))
    # Numeric sort when filenames are like 0.hdf5, 1.hdf5, ...
    def key(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)
    return sorted(files, key=key)


def main():
    p = argparse.ArgumentParser(description="Visualize tactile contact gate for raw embedded-JPEG HDF5.")
    p.add_argument("--data_dir", type=str, required=True, help="Dir containing *.hdf5 with embedded tactile")
    p.add_argument("--out_dir", type=str, default="", help="Output directory (default: <data_dir>/gate_viz)")
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--max_episodes", type=int, default=0, help="0 = all episodes")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "gate_viz")
    files = _list_hdf5(args.data_dir)
    if args.max_episodes > 0:
        files = files[: args.max_episodes]
    if not files:
        raise SystemExit(f"No HDF5 under {args.data_dir}")

    episode_stats: List[dict] = []
    all_raw_list: List[float] = []
    all_gate_list: List[float] = []

    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        try:
            tl, tr, _task = load_tactile_rgb_pair(fp, args.resize)
        except Exception as e:
            print(f"Skip {fp}: {e}")
            continue
        raw, gate = compute_raw_and_gate_series(tl, tr)
        T = len(raw)
        raw_p75 = float(np.percentile(raw[raw > 0], 75)) if (raw > 0).any() else 0.0
        frac_hi = float(np.mean(raw > raw_p75)) if raw_p75 > 0 else 0.0
        episode_stats.append(
            {
                "name": name,
                "T": T,
                "raw": raw,
                "gate": gate,
                "raw_p75": raw_p75,
                "frac_hi": frac_hi,
                "g_range": float(gate.max() - gate.min()),
                "g_mean": float(gate.mean()),
            }
        )
        all_raw_list.extend(raw.tolist())
        all_gate_list.extend(gate.tolist())
        print(f"Processed {name}: T={T}")

    all_raw = np.array(all_raw_list, dtype=np.float64)
    all_gate = np.array(all_gate_list, dtype=np.float64)

    write_summary(out_dir, episode_stats, all_raw)
    plot_figures(out_dir, episode_stats, all_raw, all_gate)


if __name__ == "__main__":
    main()
