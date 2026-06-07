# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Publication-style figures for the tactile contact gate (rho -> g_t).
# Matches training/inference: mean_abs_diff_uint8_pair + scalar_gate_from_raw (FrankaDataset).
#
# Usage:
#   cd /share/project/yunfan/cosmos-policy
#   uv run --extra cu128 --group libero --python 3.10 python -m \
#     cosmos_policy.experiments.robot.franka.visualize_tactile_gate \
#     --data_dir /share/project/yunfan/attention_data_shave_cucumber/shave_cucumber_20260321 \
#     --out_dir /share/project/yunfan/attention_data_shave_cucumber/shave_cucumber_20260321/gate_viz

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import h5py
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cosmos_policy.datasets.aloha_dataset import load_video_as_images
from cosmos_policy.datasets.dataset_utils import get_hdf5_files
from cosmos_policy.utils.tactile_self_attn_gate import mean_abs_diff_uint8_pair, scalar_gate_from_raw


def _read_path(ds) -> str:
    val = ds[()]
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def load_tactile_rgb_pair(hdf5_path: str, resize: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load left/right tactile uint8 (T,H,W,3) from Franka-style HDF5 + sidecar MP4s."""
    file_dir = os.path.dirname(hdf5_path)
    stem = os.path.splitext(os.path.basename(hdf5_path))[0]
    with h5py.File(hdf5_path, "r") as f:
        obs = f["observations"]
        vp = obs["video_paths"]
        if "tactile_rectify_left" in vp and "tactile_rectify_right" in vp:
            tl_path = os.path.join(file_dir, _read_path(vp["tactile_rectify_left"]))
            tr_path = os.path.join(file_dir, _read_path(vp["tactile_rectify_right"]))
        else:
            tl_path = os.path.join(file_dir, f"{stem}_tactile_rectify_left.mp4")
            tr_path = os.path.join(file_dir, f"{stem}_tactile_rectify_right.mp4")
        if not (os.path.isfile(tl_path) and os.path.isfile(tr_path)):
            raise FileNotFoundError(f"Missing tactile videos near {hdf5_path}")
        tl = load_video_as_images(tl_path, resize_size=resize)
        tr = load_video_as_images(tr_path, resize_size=resize)
        task = f.attrs.get("task_name", stem)
        if isinstance(task, bytes):
            task = task.decode("utf-8")
    t_len = min(len(tl), len(tr))
    return tl[:t_len].astype(np.uint8), tr[:t_len].astype(np.uint8), task


def compute_raw_and_gate_series(
    tl: np.ndarray, tr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Per timestep t: raw[0]=0, raw[t]=mean_abs_diff for t>=1; gate = scalar_gate_from_raw."""
    t = tl.shape[0]
    raw = np.zeros(t, dtype=np.float64)
    for i in range(1, t):
        raw[i] = mean_abs_diff_uint8_pair(tl[i], tr[i], tl[i - 1], tr[i - 1])
    gate = np.array([scalar_gate_from_raw(float(r)) for r in raw], dtype=np.float64)
    return raw, gate


def write_summary(out_dir: str, episode_stats: List[dict], all_raw: np.ndarray) -> None:
    lines = []
    n_diff = int((all_raw > 0).sum())
    lines.append(f"Total pairwise-diff steps (t>=1): {n_diff}")
    pos = all_raw[all_raw > 0]
    if pos.size > 0:
        lines.append(
            f"Global raw_event: min={pos.min():.6f} p50={np.percentile(pos, 50):.6f} "
            f"p90={np.percentile(pos, 90):.6f} max={pos.max():.6f}"
        )
        lines.append(
            f"Global raw_event: mean={pos.mean():.6f} std={pos.std():.6f} "
            f"CV={pos.std() / (pos.mean() + 1e-9):.3f}"
        )
    for ep in episode_stats:
        lines.append(
            f"  {ep['name']}: len={ep['T']} raw_p75={ep['raw_p75']:.6f} "
            f"frac_above_p75={ep['frac_hi']:.3f} gate_range={ep['g_range']:.4f} gate_mean={ep['g_mean']:.4f}"
        )
    lines.append("")
    lines.append("Interpretation (heuristic):")
    lines.append("  - Peaks in raw along time → contact / slip / scraping transients.")
    lines.append("  - High CV → heterogeneous dynamics; near-constant raw → gate adds little.")
    path = os.path.join(out_dir, "gate_score_summary.txt")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def plot_figures(
    out_dir: str,
    episode_stats: List[dict],
    all_raw: np.ndarray,
    all_gate: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # --- ACM-friendly typography (works with or without acmart loaded) ---
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

    # (A) Distribution of raw on positive steps
    fig_a, ax_a = plt.subplots(figsize=(3.4, 2.4))
    pos = all_raw[all_raw > 0]
    if pos.size > 0:
        ax_a.hist(pos, bins=48, color=c_raw, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax_a.axvline(np.median(pos), color="#333", linestyle="--", linewidth=1, label=f"median={np.median(pos):.4f}")
        ax_a.set_xlabel(r"Tactile change $\rho_t$ (normalized MAD)")
        ax_a.set_ylabel("Count")
        ax_a.set_title("Distribution of frame-to-frame tactile change")
        ax_a.legend(loc="upper right", frameon=False)
    fig_a.savefig(os.path.join(out_dir, "gate_raw_histogram.pdf"))
    fig_a.savefig(os.path.join(out_dir, "gate_raw_histogram.png"))
    plt.close(fig_a)

    # (B) Overview: stacked episodes, rho + gate
    n_ep = len(episode_stats)
    fig_b = plt.figure(figsize=(7.0, 0.55 + n_ep * 1.15))
    gs = GridSpec(n_ep, 1, figure=fig_b, hspace=0.55, left=0.09, right=0.97, top=0.94, bottom=0.06)

    for idx, ep in enumerate(episode_stats):
        ax = fig_b.add_subplot(gs[idx, 0])
        t = np.arange(ep["T"], dtype=np.float64)
        raw = ep["raw"]
        gate = ep["gate"]
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)

        (l1,) = ax.plot(t, raw, color=c_raw, linewidth=1.1, label=r"$\rho_t$ (raw)")
        ax.fill_between(t, 0, raw, color=c_raw, alpha=0.12)
        (l2,) = ax2.plot(t, gate, color=c_gate, linewidth=1.1, label=r"$g_t$ (gate)")
        ax2.fill_between(t, 0, gate, color=c_fill)

        thr = ep["raw_p75"]
        ax.axhline(thr, color="#666", linestyle=":", linewidth=0.8, alpha=0.9)
        ax.set_ylabel(r"$\rho_t$", color=c_raw)
        ax.tick_params(axis="y", labelcolor=c_raw)
        ax2.set_ylabel(r"$g_t$", color=c_gate)
        ax2.tick_params(axis="y", labelcolor=c_gate)
        ax.set_xlim(0, ep["T"] - 1)
        ax.set_xlabel("Timestep in episode")
        ax.set_title(f"{ep['name']}", loc="left", fontweight="semibold")

        lines = [l1, l2]
        ax.legend(lines, [l.get_label() for l in lines], loc="upper right", framealpha=0.92, fontsize=7)

    fig_b.suptitle("Contact gate along training demonstrations", fontsize=11, fontweight="bold", y=1.01)
    fig_b.savefig(os.path.join(out_dir, "gate_trajectory_overview.pdf"))
    fig_b.savefig(os.path.join(out_dir, "gate_trajectory_overview.png"))
    plt.close(fig_b)

    # (C) Scatter raw vs gate (all steps t>=1) — shows sigmoid mapping
    fig_c, ax_c = plt.subplots(figsize=(3.2, 2.6))
    mask = all_raw > 0
    if mask.sum() > 0:
        r = all_raw[mask]
        g = all_gate[mask]
        ax_c.scatter(r, g, s=4, alpha=0.25, c=c_gate, edgecolors="none", rasterized=True)
        rs = np.linspace(0, max(r.max(), 1e-6), 200)
        gs = np.array([scalar_gate_from_raw(float(x)) for x in rs])
        ax_c.plot(rs, gs, color=c_raw, linewidth=2, label="mapping (Eq.~sigmoid)")
        ax_c.set_xlabel(r"$\rho_t$")
        ax_c.set_ylabel(r"$g_t$")
        ax_c.set_title(r"Empirical $\rho_t \rightarrow g_t$")
        ax_c.legend(loc="lower right", frameon=False)
    fig_c.savefig(os.path.join(out_dir, "gate_rho_to_g_curve.pdf"))
    fig_c.savefig(os.path.join(out_dir, "gate_rho_to_g_curve.png"))
    plt.close(fig_c)

    print(f"Saved PDF/PNG under {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Visualize tactile contact gate (rho, g_t) for Franka HDF5+MP4 data.")
    p.add_argument("--data_dir", type=str, required=True, help="Dataset root (uses train/*.hdf5)")
    p.add_argument("--out_dir", type=str, default="", help="Output directory (default: <data_dir>/gate_viz)")
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--max_episodes", type=int, default=0, help="0 = all train episodes")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "gate_viz")
    files = sorted(get_hdf5_files(args.data_dir, is_train=True))
    if args.max_episodes > 0:
        files = files[: args.max_episodes]
    if not files:
        raise SystemExit(f"No HDF5 in train under {args.data_dir}")

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

    all_raw = np.array(all_raw_list, dtype=np.float64)
    all_gate = np.array(all_gate_list, dtype=np.float64)

    write_summary(out_dir, episode_stats, all_raw)
    plot_figures(out_dir, episode_stats, all_raw, all_gate)


if __name__ == "__main__":
    main()
