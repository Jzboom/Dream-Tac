# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Encode RGB frames with the Cosmos policy Wan tokenizer (VAE), decode, and compare to GT.
# Uses the same HDF5 + sidecar mp4 layout as run_franka_openloop.load_episode.
#
# Default: tactile left + right only (--cameras tactile_both). Use front / wrist / both for RGB cams.
#
# Usage (from repo root, with CUDA):
#   uv run --extra cu128 --group libero --python 3.10 python -m \
#     cosmos_policy.experiments.robot.franka.vae_reconstruct_episode \
#     --hdf5 .../episode_0.hdf5 --out_dir ./vae_recon_tactile

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cosmos_policy.experiments.robot.cosmos_utils import COSMOS_IMAGE_SIZE, DEVICE, get_model
from cosmos_policy.experiments.robot.franka.run_franka_openloop import build_franka_cfg, load_episode

CONFIG_FILE = os.environ.get("FRANKA_COSMOS_CONFIG_FILE", "cosmos_policy/config/config.py")


def _to_minus1_1_bcthw(rgb_u8_hwc: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 -> (1,3,1,H,W) float in [-1,1]."""
    t = torch.from_numpy(np.ascontiguousarray(rgb_u8_hwc)).float()
    t = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # 1,3,1,H,W
    return t / 127.5 - 1.0


def _from_minus1_1_to_u8_hwc(x: torch.Tensor) -> np.ndarray:
    """(1,3,1,H,W) -> (H,W,3) uint8."""
    y = ((x.float().squeeze(0).squeeze(1).permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255)
    return y.round().to(torch.uint8).cpu().numpy()


def _psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(255.0**2 / mse))


def _ssim_u8_gray(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity as sk_ssim
    except ImportError:
        return None
    ag = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float64)
    bg = (0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]).astype(np.float64)
    ag = ag / 255.0
    bg = bg / 255.0
    return float(sk_ssim(ag, bg, data_range=1.0))


@torch.inference_mode()
def _vae_roundtrip(tokenizer, x_minus1_1: torch.Tensor) -> torch.Tensor:
    """x: (1,3,1,H,W) on CUDA, dtype matches tokenizer."""
    tok = tokenizer
    z = tok.encode(x_minus1_1)
    recon = tok.decode(z)
    return recon


def _concat_side_by_side(gt: np.ndarray, recon: np.ndarray) -> np.ndarray:
    return np.concatenate([gt, recon], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hdf5",
        type=str,
        default="/share/project/yunfan/attention_data_cut_banana/cut_banana_20260321/train/episode_0.hdf5",
    )
    ap.add_argument("--out_dir", type=str, default="./vae_recon_episode")
    ap.add_argument(
        "--config",
        type=str,
        default=os.environ.get(
            "FRANKA_COSMOS_CONFIG", "cosmos_predict2_2b_480p_franka_cut_banana_20260321_no_tactile"
        ),
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default=os.environ.get(
            "FRANKA_COSMOS_CKPT",
            "/share/project/yunfan/cosmos-policy/cosmos_policy/ckpt/cosmos_banana_base/iter_000003350",
        ),
    )
    ap.add_argument("--use_tactile_config", action="store_true", help="build_franka_cfg(use_tactile=True)")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument(
        "--cameras",
        type=str,
        default="tactile_both",
        choices=(
            "front",
            "wrist",
            "both",
            "tactile_left",
            "tactile_right",
            "tactile_both",
        ),
        help="tactile_* = GelSight rectify streams; need *_tactile_rectify_left/right.mp4 (or tactile in HDF5).",
    )
    args = ap.parse_args()

    train_dir = os.path.dirname(os.path.abspath(args.hdf5))
    ep = os.path.splitext(os.path.basename(args.hdf5))[0]
    cam_front = os.path.join(train_dir, f"{ep}_cam_front.mp4")
    cam_high = os.path.join(train_dir, f"{ep}_cam_high.mp4")
    tl = os.path.join(train_dir, f"{ep}_tactile_rectify_left.mp4")
    tr = os.path.join(train_dir, f"{ep}_tactile_rectify_right.mp4")

    os.environ["FRANKA_COSMOS_CONFIG"] = args.config
    os.environ["FRANKA_COSMOS_CKPT"] = args.ckpt
    cfg = build_franka_cfg(use_tactile=args.use_tactile_config)
    cfg.config = args.config
    cfg.ckpt_path = args.ckpt
    cfg.config_file = CONFIG_FILE

    print(f"Loading model (tokenizer) from {args.ckpt} …")
    model, _ = get_model(cfg)
    tokenizer = model.tokenizer
    dtype = getattr(tokenizer, "dtype", torch.bfloat16)

    qpos, _, _task, cam_f, cam_h, tact_l, tact_r = load_episode(
        args.hdf5,
        cam_front,
        cam_high,
        tl if os.path.isfile(tl) else None,
        tr if os.path.isfile(tr) else None,
    )

    if args.cameras.startswith("tactile"):
        if tact_l is None or tact_r is None:
            sys.exit(
                "Tactile frames not found. Add sidecar MP4s:\n"
                f"  {tl}\n  {tr}\n"
                "or store tactile arrays in the HDF5 (see run_franka_openloop.load_episode)."
            )

    T = len(cam_f)
    indices = list(range(0, T, args.stride))
    if args.max_frames > 0:
        indices = indices[: args.max_frames]

    os.makedirs(args.out_dir, exist_ok=True)
    sub_gt = os.path.join(args.out_dir, "gt")
    sub_recon = os.path.join(args.out_dir, "recon")
    sub_side = os.path.join(args.out_dir, "side_by_side")
    for d in (sub_gt, sub_recon, sub_side):
        os.makedirs(d, exist_ok=True)

    from PIL import Image

    records: List[dict] = []
    cams: List[Tuple[str, List[np.ndarray]]] = []
    if args.cameras in ("front", "both"):
        cams.append(("cam_front", [np.asarray(cam_f[i], dtype=np.uint8) for i in indices]))
    if args.cameras in ("wrist", "both"):
        cams.append(("cam_high", [np.asarray(cam_h[i], dtype=np.uint8) for i in indices]))
    if args.cameras in ("tactile_left", "tactile_both"):
        assert tact_l is not None
        cams.append(("tactile_left", [np.asarray(tact_l[i], dtype=np.uint8) for i in indices]))
    if args.cameras in ("tactile_right", "tactile_both"):
        assert tact_r is not None
        cams.append(("tactile_right", [np.asarray(tact_r[i], dtype=np.uint8) for i in indices]))

    for cam_name, frames in cams:
        psnrs: List[float] = []
        ssims: List[float] = []
        for j, t_idx in enumerate(indices):
            gt = frames[j]
            assert gt.shape[2] == 3 and gt.dtype == np.uint8
            if gt.shape[0] != COSMOS_IMAGE_SIZE or gt.shape[1] != COSMOS_IMAGE_SIZE:
                raise ValueError(
                    f"Expected {COSMOS_IMAGE_SIZE}x{COSMOS_IMAGE_SIZE}, got {gt.shape}; "
                    "load_episode should already resize."
                )

            x = _to_minus1_1_bcthw(gt).to(device=DEVICE, dtype=dtype)
            recon = _vae_roundtrip(tokenizer, x)
            recon_u8 = _from_minus1_1_to_u8_hwc(recon)

            psnr = _psnr_u8(gt, recon_u8)
            psnrs.append(psnr)
            sm = _ssim_u8_gray(gt, recon_u8)
            if sm is not None:
                ssims.append(sm)

            stem = f"{cam_name}_t{t_idx:04d}"
            Image.fromarray(gt).save(os.path.join(sub_gt, f"{stem}.png"))
            Image.fromarray(recon_u8).save(os.path.join(sub_recon, f"{stem}.png"))
            Image.fromarray(_concat_side_by_side(gt, recon_u8)).save(os.path.join(sub_side, f"{stem}.png"))

            rec = {"camera": cam_name, "timestep": int(t_idx), "psnr": psnr}
            if sm is not None:
                rec["ssim_gray"] = sm
            records.append(rec)
            print(f"  {stem}  PSNR={psnr:.2f} dB" + (f"  SSIM={sm:.4f}" if sm is not None else ""))

        summary_cam = {
            "camera": cam_name,
            "num_frames": len(psnrs),
            "mean_psnr": float(np.mean(psnrs)),
            "std_psnr": float(np.std(psnrs)),
        }
        if ssims:
            summary_cam["mean_ssim_gray"] = float(np.mean(ssims))
            summary_cam["std_ssim_gray"] = float(np.std(ssims))
        else:
            summary_cam["mean_ssim_gray"] = None
            summary_cam["note"] = "install scikit-image for SSIM"

    summary = {
        "hdf5": args.hdf5,
        "config": args.config,
        "ckpt": args.ckpt,
        "indices": indices,
        "per_frame": records,
    }
    json_path = os.path.join(args.out_dir, "vae_recon_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")
    print("Done. Folders: gt/, recon/, side_by_side/")


if __name__ == "__main__":
    main()
