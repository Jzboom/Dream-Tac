# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Compare future primary + wrist predictions (Pick/Cut Banana) between two checkpoints.
# Default pairing: Dream-Tac (ours) = tactile, no tactile self-attn bias (see config_ours);
#                  Cosmos-Policy baseline = no tactile.
# Figure modes: wide | paper_rows | sixrow (default sixrow = 6×K panels, K=5 → 30 images).
# Diffusion: pass --denoise_steps_ours (default 30) and --denoise_steps_base (default 10) to get_action.
# sixrow writes tiles under out_dir/tiles/{gt,cosmos,dreamtac}_{primary,wrist}/col*.png
# and montage under out_dir/montage/compare_grid.{pdf,png}.
#
# Usage:
#   uv run ... python -m cosmos_policy.experiments.robot.franka.compare_cut_banana_future_visual

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.franka.run_franka_openloop import (
    CHUNK_SIZE,
    COSMOS_IMAGE_SIZE,
    build_franka_cfg,
    foreground_mask_iou,
    load_episode,
    _to_uint8_hw3,
)

CONFIG_FILE = os.environ.get("FRANKA_COSMOS_CONFIG_FILE", "cosmos_policy/config/config.py")
DATA_DIR_DEFAULT = "/share/project/yunfan/attention_data_cut_banana/cut_banana_20260321"


def _load_setup(config_name: str, ckpt: str, use_tactile: bool, stats_path: str, t5_path: str):
    os.environ["FRANKA_COSMOS_CONFIG"] = config_name
    os.environ["FRANKA_COSMOS_CKPT"] = ckpt
    cfg = build_franka_cfg(use_tactile=use_tactile)
    cfg.config = config_name
    cfg.ckpt_path = ckpt
    cfg.config_file = CONFIG_FILE
    model, _ = get_model(cfg)
    ds = load_dataset_stats(stats_path) if os.path.isfile(stats_path) else {}
    if not ds:
        cfg.unnormalize_actions = False
    if os.path.isfile(t5_path):
        init_t5_text_embeddings_cache(t5_path)
    return cfg, model, ds


def _infer_one(
    cfg,
    model,
    ds,
    cam_front,
    cam_high,
    qpos,
    tactile_left,
    tactile_right,
    start: int,
    instruction: str,
    num_denoising_steps_action: int,
):
    obs = {
        "primary_image": cam_front[start],
        "wrist_image": cam_high[start],
        "proprio": qpos[start],
    }
    if cfg.use_tactile and tactile_left is not None:
        obs["tactile_left_image"] = tactile_left[start]
        obs["tactile_right_image"] = tactile_right[start]
    out = get_action(
        cfg,
        model,
        ds,
        obs,
        instruction,
        seed=0,
        randomize_seed=False,
        num_denoising_steps_action=num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=True,
    )
    fut = out.get("future_image_predictions") or {}
    prim = fut.get("future_image")
    wrist = fut.get("future_wrist_image")
    if prim is None or wrist is None:
        return None, None
    return _to_uint8_hw3(prim), _to_uint8_hw3(wrist)


def _resize_to_gt(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if pred.shape[:2] == ref.shape[:2]:
        return pred
    from PIL import Image

    return np.array(
        Image.fromarray(pred).resize((ref.shape[1], ref.shape[0]), Image.BICUBIC)
    )


def _apply_publication_rc(publication: bool) -> None:
    import matplotlib as mpl

    base = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    }
    if publication:
        base.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Nimbus Sans"],
                "font.size": 8.5,
                "axes.titlesize": 8,
                "figure.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.edgecolor": "white",
            }
        )
    else:
        base["font.size"] = 8
    mpl.rcParams.update(base)


def _show_image(ax, img) -> None:
    ax.imshow(np.asarray(img).astype(np.uint8), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def _iou_overlay(ax, iou: float) -> None:
    """Small IoU annotation on prediction panels (white text on dark shadow for contrast)."""
    label = "IoU —" if (iou != iou) else f"IoU {iou:.2f}"  # NaN check
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=7,
        color="white",
        va="top",
        ha="left",
        fontweight="medium",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.45, edgecolor="none"),
    )


def _figure_wide(
    starts: list[int],
    gt_indices: list[int],
    gt_primary: list[np.ndarray],
    gt_wrist: list[np.ndarray],
    base_primary: list[np.ndarray],
    base_wrist: list[np.ndarray],
    ours_primary: list[np.ndarray],
    ours_wrist: list[np.ndarray],
    iou_base: list[dict],
    iou_ours: list[dict],
    task_name: str,
    publication: bool,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(starts)
    ncols = 2 * k
    fig_w = max(14.0, 1.85 * ncols)
    fig_h = 6.8 if publication else 6.2
    fig, axes = plt.subplots(3, ncols, figsize=(fig_w, fig_h))
    axes = np.asarray(axes)
    if ncols == 1:
        axes = axes.reshape(3, 1)

    row_names = ("Ground truth", "Cosmos-Policy", "Dream-Tac (ours)")

    for j in range(k):
        s, gti = starts[j], gt_indices[j]
        axes[0, j].set_title(f"Primary\n$s$={s}→$t$={gti}", fontsize=7)
        axes[0, k + j].set_title(f"Wrist\n$s$={s}→$t$={gti}", fontsize=7)

    for j in range(k):
        _show_image(axes[0, j], gt_primary[j])
        _show_image(axes[0, k + j], gt_wrist[j])
    for j in range(k):
        _show_image(axes[1, j], base_primary[j])
        _show_image(axes[1, k + j], base_wrist[j])
        ibp = iou_base[j].get("primary", float("nan"))
        ibw = iou_base[j].get("wrist", float("nan"))
        if publication:
            _iou_overlay(axes[1, j], ibp)
            _iou_overlay(axes[1, k + j], ibw)
        else:
            axes[1, j].set_title(f"mask-IoU\n{ibp:.3f}", fontsize=7)
            axes[1, k + j].set_title(f"mask-IoU\n{ibw:.3f}", fontsize=7)
    for j in range(k):
        _show_image(axes[2, j], ours_primary[j])
        _show_image(axes[2, k + j], ours_wrist[j])
        iop = iou_ours[j].get("primary", float("nan"))
        iow = iou_ours[j].get("wrist", float("nan"))
        if publication:
            _iou_overlay(axes[2, j], iop)
            _iou_overlay(axes[2, k + j], iow)
        else:
            axes[2, j].set_title(f"mask-IoU\n{iop:.3f}", fontsize=7)
            axes[2, k + j].set_title(f"mask-IoU\n{iow:.3f}", fontsize=7)

    for ri in range(3):
        axes[ri, 0].text(
            -0.04,
            0.5,
            row_names[ri],
            transform=axes[ri, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=10 if publication else 10,
            fontweight="bold",
        )

    st = (
        f"{task_name}\n"
        "Predicted future frames at chunk targets (Otsu foreground mask–IoU on primary / wrist)."
        if publication
        else f"{task_name} — future frames at chunk targets (joint Otsu mask-IoU on primary / wrist)"
    )
    fig.suptitle(st, fontsize=10 if publication else 11, y=1.01)
    plt.subplots_adjust(left=0.05, right=0.995, top=0.86, bottom=0.04, wspace=0.1, hspace=0.32)
    return fig


def _figure_paper_rows(
    starts: list[int],
    gt_indices: list[int],
    gt_primary: list[np.ndarray],
    gt_wrist: list[np.ndarray],
    base_primary: list[np.ndarray],
    base_wrist: list[np.ndarray],
    ours_primary: list[np.ndarray],
    ours_wrist: list[np.ndarray],
    iou_base: list[dict],
    iou_ours: list[dict],
    task_name: str,
    views: str,
):
    """
    One row per chunk start. Columns:
      both: GT_p, Base_p, Ours_p | GT_w, Base_w, Ours_w  (6 cols)
      primary|wrist: GT, Base, Ours (3 cols)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(starts)
    if views == "both":
        ncols = 6
    else:
        ncols = 3
    fig_w = 12.2 if views == "both" else 8.4
    row_h = 0.62 if views == "both" else 0.78
    fig_h = min(22.0, 1.15 + k * row_h)
    fig, axes = plt.subplots(k, ncols, figsize=(fig_w, fig_h))
    if k == 1:
        axes = np.asarray(axes).reshape(1, -1)
    col_titles_primary = ("GT", "Cosmos-Policy", "Dream-Tac")
    col_titles_wrist = ("GT", "Cosmos-Policy", "Dream-Tac")

    for c in range(ncols):
        title = None
        if views == "both":
            if c < 3:
                title = f"Primary — {col_titles_primary[c]}"
            else:
                title = f"Wrist — {col_titles_wrist[c - 3]}"
        else:
            cam = "Primary" if views == "primary" else "Wrist"
            title = f"{cam} — {col_titles_primary[c]}"
        axes[0, c].set_title(title, fontsize=8, fontweight="bold", pad=4)

    for j in range(k):
        s, gti = starts[j], gt_indices[j]
        axes[j, 0].text(
            -0.22,
            0.5,
            f"$s$={s}\n→$t$={gti}",
            transform=axes[j, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=7,
        )

        def pick(pri, wri):
            return pri if views != "wrist" else wri

        gp, gw = gt_primary[j], gt_wrist[j]
        bp, bw = base_primary[j], base_wrist[j]
        op, ow = ours_primary[j], ours_wrist[j]

        if views == "both":
            _show_image(axes[j, 0], gp)
            _show_image(axes[j, 1], bp)
            _show_image(axes[j, 2], op)
            _show_image(axes[j, 3], gw)
            _show_image(axes[j, 4], bw)
            _show_image(axes[j, 5], ow)
            ibp, ibw = iou_base[j].get("primary", float("nan")), iou_base[j].get("wrist", float("nan"))
            iop, iow = iou_ours[j].get("primary", float("nan")), iou_ours[j].get("wrist", float("nan"))
            _iou_overlay(axes[j, 1], ibp)
            _iou_overlay(axes[j, 2], iop)
            _iou_overlay(axes[j, 4], ibw)
            _iou_overlay(axes[j, 5], iow)
        else:
            g, b, o = pick(gp, gw), pick(bp, bw), pick(op, ow)
            _show_image(axes[j, 0], g)
            _show_image(axes[j, 1], b)
            _show_image(axes[j, 2], o)
            ib = iou_base[j].get("primary" if views == "primary" else "wrist", float("nan"))
            io = iou_ours[j].get("primary" if views == "primary" else "wrist", float("nan"))
            _iou_overlay(axes[j, 1], ib)
            _iou_overlay(axes[j, 2], io)

    fig.suptitle(
        f"{task_name}\n"
        "Future prediction at each chunk target (mask–IoU on overlaid predictions).",
        fontsize=10,
        y=1.002,
        va="bottom",
    )
    plt.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.01, wspace=0.06, hspace=0.25)
    return fig


def _save_figure(fig, out_pdf: str, dpi_png: int) -> None:
    import matplotlib.pyplot as plt

    out_png = out_pdf.replace(".pdf", ".png")
    fig.savefig(out_png, dpi=dpi_png, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _save_thirty_tiles(
    out_root: str,
    gt_primary: list[np.ndarray],
    gt_wrist: list[np.ndarray],
    base_primary: list[np.ndarray],
    base_wrist: list[np.ndarray],
    ours_primary: list[np.ndarray],
    ours_wrist: list[np.ndarray],
) -> None:
    from PIL import Image

    pairs = [
        ("gt_primary", gt_primary),
        ("gt_wrist", gt_wrist),
        ("cosmos_primary", base_primary),
        ("cosmos_wrist", base_wrist),
        ("dreamtac_primary", ours_primary),
        ("dreamtac_wrist", ours_wrist),
    ]
    for name, frames in pairs:
        d = os.path.join(out_root, "tiles", name)
        os.makedirs(d, exist_ok=True)
        for j, im in enumerate(frames):
            Image.fromarray(np.asarray(im).astype(np.uint8)).save(os.path.join(d, f"col{j:02d}.png"))


def _figure_sixrow(
    starts: list[int],
    gt_indices: list[int],
    gt_primary: list[np.ndarray],
    gt_wrist: list[np.ndarray],
    base_primary: list[np.ndarray],
    base_wrist: list[np.ndarray],
    ours_primary: list[np.ndarray],
    ours_wrist: list[np.ndarray],
    denoise_base: int,
    denoise_ours: int,
):
    """6 rows × K cols: GT primary/wrist, Cosmos-Policy primary/wrist, Dream-Tac primary/wrist."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(starts)
    fig_w = max(10.0, 2.05 * k)
    fig_h = 2.35 * 6
    fig, axes = plt.subplots(6, k, figsize=(fig_w, fig_h))
    if k == 1:
        axes = np.asarray(axes).reshape(6, 1)

    row_labels = (
        "GT — primary (cam_front)",
        "GT — wrist (cam_high)",
        f"Cosmos-Policy — primary ({denoise_base}-step diffusion)",
        f"Cosmos-Policy — wrist ({denoise_base}-step diffusion)",
        f"Dream-Tac (ours) — primary, cam_front ({denoise_ours}-step diffusion)",
        f"Dream-Tac (ours) — wrist, cam_high ({denoise_ours}-step diffusion)",
    )
    rows_imgs = (gt_primary, gt_wrist, base_primary, base_wrist, ours_primary, ours_wrist)

    for ri in range(6):
        for j in range(k):
            _show_image(axes[ri, j], rows_imgs[ri][j])
        axes[ri, 0].text(
            -0.11,
            0.5,
            row_labels[ri],
            transform=axes[ri, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=7.5,
            fontweight="bold",
        )

    plt.subplots_adjust(left=0.14, right=0.99, top=0.99, bottom=0.02, wspace=0.06, hspace=0.12)
    return fig


def _figure_three_rows(
    starts: list[int],
    gt_indices: list[int],
    gt_primary: list[np.ndarray],
    gt_wrist: list[np.ndarray],
    base_primary: list[np.ndarray],
    base_wrist: list[np.ndarray],
    ours_primary: list[np.ndarray],
    ours_wrist: list[np.ndarray],
    iou_base: list[dict],
    iou_ours: list[dict],
    out_path: str,
    task_name: str,
    *,
    layout: str = "wide",
    views: str = "both",
    publication: bool = False,
    dpi_png: int = 200,
    denoise_steps_base: int = 10,
    denoise_steps_ours: int = 30,
):
    import matplotlib

    matplotlib.use("Agg")

    _apply_publication_rc(publication)
    if layout == "sixrow":
        fig = _figure_sixrow(
            starts,
            gt_indices,
            gt_primary,
            gt_wrist,
            base_primary,
            base_wrist,
            ours_primary,
            ours_wrist,
            denoise_steps_base,
            denoise_steps_ours,
        )
    elif layout == "paper_rows":
        fig = _figure_paper_rows(
            starts,
            gt_indices,
            gt_primary,
            gt_wrist,
            base_primary,
            base_wrist,
            ours_primary,
            ours_wrist,
            iou_base,
            iou_ours,
            task_name,
            views,
        )
    else:
        fig = _figure_wide(
            starts,
            gt_indices,
            gt_primary,
            gt_wrist,
            base_primary,
            base_wrist,
            ours_primary,
            ours_wrist,
            iou_base,
            iou_ours,
            task_name,
            publication,
        )
    _save_figure(fig, out_path, dpi_png=dpi_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=DATA_DIR_DEFAULT)
    ap.add_argument("--episode", type=str, default="episode_0")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument(
        "--ckpt_ours",
        type=str,
        default="/share/project/yunfan/cosmos-policy/cosmos_policy/ckpt/cosmos_cut_banana_tac_no_bias/iter_000005500",
        help="Dream-Tac / ours checkpoint (default: tactile inputs, no tactile self-attn bias).",
    )
    ap.add_argument(
        "--config_ours",
        type=str,
        default="cosmos_predict2_2b_480p_franka_cut_banana_20260321_tactile_no_attn_bias",
        help="Must match the checkpoint (default pairs with cosmos_cut_banana_tac_no_bias).",
    )
    ap.add_argument(
        "--ckpt_base",
        type=str,
        default="/share/project/yunfan/cosmos-policy/cosmos_policy/ckpt/cosmos_banana_base/iter_000003350",
        help="Vision-only baseline checkpoint (no tactile).",
    )
    ap.add_argument(
        "--config_base",
        type=str,
        default="cosmos_predict2_2b_480p_franka_cut_banana_20260321_no_tactile",
        help="Experiment name: no tactile inputs / LIBERO 8-slot layout.",
    )
    ap.add_argument(
        "--num_panels",
        type=int,
        default=5,
        help="Chunk starts; with sixrow (default) this is columns (5 → 30 total images).",
    )
    ap.add_argument(
        "--denoise_steps_ours",
        type=int,
        default=30,
        help="Dream-Tac: num_denoising_steps_action for joint action+future sampling.",
    )
    ap.add_argument(
        "--denoise_steps_base",
        type=int,
        default=10,
        help="Cosmos-Policy baseline: num_denoising_steps_action.",
    )
    ap.add_argument(
        "--publication",
        action="store_true",
        help="Fonts/spacing for print + 300 dpi PNG (pdf.fonttype 42). Implies cleaner labels.",
    )
    ap.add_argument(
        "--figure_layout",
        type=str,
        choices=("auto", "wide", "paper_rows", "sixrow"),
        default="sixrow",
        help="sixrow: 6×K (GT/Cosmos/Dream-Tac × primary+wrist). auto: paper_rows if num_panels>8 else wide.",
    )
    ap.add_argument(
        "--views",
        type=str,
        choices=("both", "primary", "wrist"),
        default="both",
        help="Cameras per row in paper_rows (wide layout always shows both).",
    )
    ap.add_argument(
        "--dpi_png",
        type=int,
        default=0,
        help="PNG resolution; 0 = 300 if --publication else 200.",
    )
    args = ap.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    ep = args.episode
    hdf5 = os.path.join(train_dir, f"{ep}.hdf5")
    cam_front = os.path.join(train_dir, f"{ep}_cam_front.mp4")
    cam_high = os.path.join(train_dir, f"{ep}_cam_high.mp4")
    tl = os.path.join(train_dir, f"{ep}_tactile_rectify_left.mp4")
    tr = os.path.join(train_dir, f"{ep}_tactile_rectify_right.mp4")

    out_dir = args.out_dir or os.path.join(
        "/share/project/yunfan/cosmos-policy/cosmos_policy/ckpt",
        "eval_cut_banana_future_visual_tacbias_vs_notactile_iter3350",
    )
    os.makedirs(out_dir, exist_ok=True)

    stats_path = os.path.join(args.data_dir, "dataset_statistics_franka.json")
    t5_path = os.path.join(args.data_dir, "t5_embeddings.pkl")

    qpos, _, task_name, cam_f, cam_h, tact_l, tact_r = load_episode(
        hdf5, cam_front, cam_high, tl if os.path.isfile(tl) else None, tr if os.path.isfile(tr) else None,
    )
    T = qpos.shape[0]
    # Only the explicit no_tactile experiment disables tactile; do not use "tactile_no" substring
    # (it would wrongly exclude tactile_no_attn_bias configs).
    use_tactile_ours = tact_l is not None and tact_r is not None and "no_tactile" not in args.config_ours

    # chunk starts: spread over valid range
    max_start = max(0, T - CHUNK_SIZE - 1)
    if args.num_panels <= 1:
        starts = [0]
    else:
        starts = [int(round(i * max_start / (args.num_panels - 1))) for i in range(args.num_panels)]

    print(f"Ours: use_tactile={use_tactile_ours} config={args.config_ours}")
    print(f"Base: use_tactile=False config={args.config_base}")
    print("Loading Dream-Tac model...")
    cfg_o, model_o, ds_o = _load_setup(
        args.config_ours, args.ckpt_ours, use_tactile_ours, stats_path, t5_path
    )
    print("Loading Cosmos-Policy baseline...")
    cfg_b, model_b, ds_b = _load_setup(
        args.config_base, args.ckpt_base, False, stats_path, t5_path
    )

    records = []
    gt_ps: list[np.ndarray] = []
    gt_ws: list[np.ndarray] = []
    bps: list[np.ndarray] = []
    bws: list[np.ndarray] = []
    ops: list[np.ndarray] = []
    ows: list[np.ndarray] = []
    iou_bs: list[dict] = []
    iou_os: list[dict] = []
    gt_idxs: list[int] = []
    starts_used: list[int] = []

    for k, start in enumerate(starts):
        gt_idx = min(start + CHUNK_SIZE, T - 1)
        gt_p = np.asarray(cam_f[gt_idx], dtype=np.uint8)
        gt_w = np.asarray(cam_h[gt_idx], dtype=np.uint8)

        print(f"Chunk start={start} gt_idx={gt_idx} ...")
        op, ow = _infer_one(
            cfg_o,
            model_o,
            ds_o,
            cam_f,
            cam_h,
            qpos,
            tact_l,
            tact_r,
            start,
            task_name,
            num_denoising_steps_action=args.denoise_steps_ours,
        )
        bp, bw = _infer_one(
            cfg_b,
            model_b,
            ds_b,
            cam_f,
            cam_h,
            qpos,
            None,
            None,
            start,
            task_name,
            num_denoising_steps_action=args.denoise_steps_base,
        )
        if op is None or bp is None:
            print("  skip: missing future preds")
            continue

        op = _resize_to_gt(op, gt_p)
        ow = _resize_to_gt(ow, gt_w)
        bp = _resize_to_gt(bp, gt_p)
        bw = _resize_to_gt(bw, gt_w)

        iou_o = {
            "primary": foreground_mask_iou(op, gt_p),
            "wrist": foreground_mask_iou(ow, gt_w),
        }
        iou_b = {
            "primary": foreground_mask_iou(bp, gt_p),
            "wrist": foreground_mask_iou(bw, gt_w),
        }
        records.append(
            {
                "chunk_start": start,
                "gt_timestep": gt_idx,
                "dream_tac": iou_o,
                "cosmos_policy": iou_b,
            }
        )
        starts_used.append(start)
        gt_idxs.append(gt_idx)
        gt_ps.append(gt_p)
        gt_ws.append(gt_w)
        bps.append(bp)
        bws.append(bw)
        ops.append(op)
        ows.append(ow)
        iou_bs.append(iou_b)
        iou_os.append(iou_o)
        print(f"  IoU ours {iou_o} base {iou_b}")

    layout = args.figure_layout
    if layout == "auto":
        layout = "paper_rows" if args.num_panels > 8 else "wide"
    dpi_png = args.dpi_png if args.dpi_png > 0 else (300 if args.publication else 200)
    pub = args.publication or layout == "paper_rows"

    if layout == "sixrow":
        montage_dir = os.path.join(out_dir, "montage")
        os.makedirs(montage_dir, exist_ok=True)
        grid_pdf = os.path.join(montage_dir, "compare_grid.pdf")
    elif layout == "wide" and not args.publication:
        grid_pdf = os.path.join(out_dir, "future_vis_compare_grid.pdf")
    else:
        stem = "future_visual_compare"
        if layout == "paper_rows":
            stem += "_rows"
        if args.publication:
            stem += "_paper"
        grid_pdf = os.path.join(out_dir, f"{stem}.pdf")

    if starts_used:
        if layout == "sixrow":
            _save_thirty_tiles(out_dir, gt_ps, gt_ws, bps, bws, ops, ows)
        _figure_three_rows(
            starts_used,
            gt_idxs,
            gt_ps,
            gt_ws,
            bps,
            bws,
            ops,
            ows,
            iou_bs,
            iou_os,
            grid_pdf,
            str(task_name),
            layout=layout,
            views=args.views,
            publication=pub,
            dpi_png=dpi_png,
            denoise_steps_base=args.denoise_steps_base,
            denoise_steps_ours=args.denoise_steps_ours,
        )
        print(f"Saved figure {grid_pdf} and {grid_pdf.replace('.pdf', '.png')} (dpi={dpi_png})")
        if layout == "sixrow":
            print(f"Tiles under {os.path.join(out_dir, 'tiles')}")
    else:
        print("No successful chunks; skip grid figure.")

    def mean_key(key, sub):
        vals = [r[sub][key] for r in records]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "task_instruction": task_name,
        "episode": ep,
        "chunk_size": CHUNK_SIZE,
        "denoise_steps_dream_tac": args.denoise_steps_ours,
        "denoise_steps_cosmos_policy": args.denoise_steps_base,
        "figure_layout": layout,
        "comparison": (
            "Dream-Tac (ours): default tactile inputs without tactile self-attention bias; "
            "Cosmos-Policy (baseline): no tactile."
        ),
        "iou_definition": (
            "Joint Otsu mask IoU on grayscale (same as run_franka_openloop.foreground_mask_iou)."
        ),
        "config_ours": args.config_ours,
        "ckpt_ours": args.ckpt_ours,
        "config_base": args.config_base,
        "ckpt_base": args.ckpt_base,
        "per_panel": records,
        "mean_iou": {
            "dream_tac": {
                "primary": mean_key("primary", "dream_tac"),
                "wrist": mean_key("wrist", "dream_tac"),
            },
            "cosmos_policy": {
                "primary": mean_key("primary", "cosmos_policy"),
                "wrist": mean_key("wrist", "cosmos_policy"),
            },
        },
    }
    json_path = os.path.join(out_dir, "iou_future_visual_compare.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")
    print("Mean IoU:", summary["mean_iou"])


if __name__ == "__main__":
    main()
