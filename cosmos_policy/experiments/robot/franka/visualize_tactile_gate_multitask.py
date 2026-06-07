# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cross-task contact-gate figure for the rebuttal: shows rho_t and g_t over time for
# several manually-selected episodes, each relabeled as a distinct manipulation task.
# Demonstrates that the SAME fixed pixel-diff gate (no per-task tuning) produces
# physically meaningful behavior across very different contact patterns.
#
# Usage:
#   cd /share/project/yunfan/cosmos-policy
#   uv run --extra cu128 --group libero --python 3.10 python -m \
#     cosmos_policy.experiments.robot.franka.visualize_tactile_gate_multitask \
#     --out_dir /share/project/yunfan/cosmos-policy/cosmos_policy/ckpt/gate_rebuttal

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cosmos_policy.experiments.robot.franka.visualize_tactile_gate import (
    compute_raw_and_gate_series,
    load_tactile_rgb_pair,
)
from cosmos_policy.utils.tactile_self_attn_gate import scalar_gate_from_raw

_CB = "/share/project/yunfan/attention_data_cut_banana/cut_banana_20260321/train"
_HP = "/share/project/yunfan/tactile_data_hupai/hupai/train"

# (display task name, hdf5 path, short contact descriptor)
TASKS: List[Tuple[str, str, str]] = [
    ("Pick Baguette", os.path.join(_CB, "episode_10.hdf5"), "grasp / lift / place"),
    ("Insert USB", os.path.join(_CB, "episode_7.hdf5"), "small contact area, tight insertion"),
    ("Clean Whiteboard", os.path.join(_CB, "episode_32.hdf5"), "sustained wiping / scrubbing"),
    ("Cut Banana", os.path.join(_CB, "episode_19.hdf5"), "intermittent cutting"),
    ("Play Mahjong", os.path.join(_HP, "episode_10.hdf5"), "sparse fine-grained contact"),
]


def main():
    p = argparse.ArgumentParser(description="Cross-task gate figure for rebuttal.")
    p.add_argument("--out_dir", type=str, default="/share/project/yunfan/cosmos-policy/cosmos_policy/ckpt/gate_rebuttal")
    p.add_argument("--resize", type=int, default=224)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    series = []
    for task_name, path, desc in TASKS:
        tl, tr, _ = load_tactile_rgb_pair(path, args.resize)
        raw, gate = compute_raw_and_gate_series(tl, tr)
        pos = raw[raw > 0]
        cv = float(pos.std() / (pos.mean() + 1e-9)) if pos.size else 0.0
        series.append(
            {
                "name": task_name,
                "desc": desc,
                "raw": raw,
                "gate": gate,
                "T": len(raw),
                "rho_mean": float(pos.mean()) if pos.size else 0.0,
                "rho_p90": float(np.percentile(pos, 90)) if pos.size else 0.0,
                "cv": cv,
                "g_mean": float(gate.mean()),
                "g_min": float(gate.min()),
                "g_max": float(gate.max()),
            }
        )
        print(f"{task_name}: T={len(raw)} rho_mean={series[-1]['rho_mean']:.5f} CV={cv:.2f} g_mean={series[-1]['g_mean']:.3f}")

    _plot(args.out_dir, series)


def _plot(out_dir: str, series: List[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    c_raw = "#1f77b4"
    c_gate = "#d62728"
    c_fill = "#d6272833"

    n = len(series)
    fig = plt.figure(figsize=(8.0, 1.15 * n + 1.0))
    gs = GridSpec(n, 1, figure=fig, hspace=0.72,
                  left=0.09, right=0.92, top=0.90, bottom=0.085)

    # Common rho axis upper limit across tasks for fair visual comparison
    rho_hi = max(float(np.percentile(s["raw"][s["raw"] > 0], 99)) if (s["raw"] > 0).any() else 1e-3 for s in series)

    for i, s in enumerate(series):
        ax = fig.add_subplot(gs[i, 0])
        t = np.arange(s["T"], dtype=np.float64)
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)

        ax.plot(t, s["raw"], color=c_raw, linewidth=1.1)
        ax.fill_between(t, 0, s["raw"], color=c_raw, alpha=0.12)
        ax2.plot(t, s["gate"], color=c_gate, linewidth=1.2)
        ax2.fill_between(t, 0, s["gate"], color=c_fill)

        ax.set_ylim(0, rho_hi * 1.05)
        ax2.set_ylim(0, 1.05)
        ax.set_xlim(0, max(1, s["T"] - 1))
        ax.set_ylabel(r"$\rho_t$", color=c_raw)
        ax.tick_params(axis="y", labelcolor=c_raw)
        ax2.set_ylabel(r"$g_t$", color=c_gate)
        ax2.tick_params(axis="y", labelcolor=c_gate)
        if i == n - 1:
            ax.set_xlabel("Timestep in episode")

        ax.set_title(s["name"], loc="left", fontweight="semibold", fontsize=10)

    fig.suptitle(
        "Contact gate along training demonstrations",
        fontsize=12, fontweight="bold", y=0.965,
    )

    for ext in ("pdf", "png"):
        fp = os.path.join(out_dir, f"gate_rebuttal_multitask.{ext}")
        fig.savefig(fp)
        print(f"Saved {fp}")
    plt.close(fig)


if __name__ == "__main__":
    main()
