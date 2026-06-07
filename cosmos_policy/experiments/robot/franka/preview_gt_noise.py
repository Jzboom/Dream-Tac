#!/usr/bin/env python3
"""Visualize GT vs degraded GT (titles: Ground Truth | Predict).

Default look is smooth Gaussian blur only (no blocky patches, no obvious pixel grain).
Optional per-pixel noise and random square spots are off by default; enable via CLI if needed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def load_rgb_u8(path: str) -> np.ndarray:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def take_left_half(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[:, : w // 2].copy()


def degrade_gt(
    img_u8: np.ndarray,
    *,
    blur_radius: float = 0.9,
    noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Gaussian blur (smooth, no blocks); optional tiny i.i.d. noise if noise_std > 0."""
    from PIL import Image, ImageFilter

    pil = Image.fromarray(img_u8)
    r = max(0.0, float(blur_radius))
    if r > 0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=r))
    x = np.asarray(pil, dtype=np.float32)
    if float(noise_std) > 0:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(0.0, float(noise_std), x.shape).astype(np.float32)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def _set_single_title(ax, text: str) -> None:
    """One-line title only (clear subtitle when Axes.set_title supports it)."""
    try:
        ax.set_title(text, subtitle="")
    except (TypeError, AttributeError, ValueError):
        ax.set_title(text)


def load_video_rgb_u8(video_path: str) -> np.ndarray:
    """(T, H, W, 3) uint8 RGB. Prefer OpenCV; fall back to imageio if cv2 missing."""
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise OSError(f"Could not open video: {video_path}")
        frames: list[np.ndarray] = []
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise ValueError(f"No frames in {video_path}")
        return np.asarray(frames, dtype=np.uint8)
    except ModuleNotFoundError:
        pass

    import imageio.v3 as iio

    frames = [np.asarray(f, dtype=np.uint8) for f in iio.imiter(video_path)]
    if not frames:
        raise ValueError(f"No frames in {video_path}")
    return np.stack(frames, axis=0)


def degrade_video_frame(
    img_u8: np.ndarray,
    *,
    blur_radius: float,
    global_std: float,
    n_spots: int,
    spot_half: int,
    spot_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Blur first (smooth); then optional i.i.d. noise; then optional random square patches (off by default)."""
    from PIL import Image, ImageFilter

    h, w = img_u8.shape[:2]
    if h < 2 or w < 2:
        return img_u8.copy()

    pil = Image.fromarray(img_u8)
    r = max(0.0, float(blur_radius))
    if r > 0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=r))
    x = np.asarray(pil, dtype=np.float32)

    if float(global_std) > 0:
        x += rng.normal(0.0, float(global_std), x.shape).astype(np.float32)

    sh = max(2, min(int(spot_half), h // 4, w // 4))
    if int(n_spots) > 0 and h > 2 * sh + 1 and w > 2 * sh + 1:
        for _ in range(int(n_spots)):
            cy = int(rng.integers(sh, h - sh))
            cx = int(rng.integers(sh, w - sh))
            y0, y1 = cy - sh, cy + sh
            x0, x1 = cx - sh, cx + sh
            patch = rng.normal(0.0, float(spot_std), (y1 - y0, x1 - x0, 3)).astype(np.float32)
            x[y0:y1, x0:x1] += patch

    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def save_comparison(
    gt_u8: np.ndarray, pred_u8: np.ndarray, out_path: str, dpi: int = 120, *, verbose: bool = True
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(gt_u8)
    _set_single_title(axes[0], "Ground Truth")
    axes[0].axis("off")
    axes[1].imshow(pred_u8)
    _set_single_title(axes[1], "Predict")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    if verbose:
        print(f"Saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="GT vs blurred GT preview (Ground Truth | Predict)")
    p.add_argument("--input", type=str, default="", help="Single image path")
    p.add_argument(
        "--video",
        type=str,
        default="",
        help="MP4 path: save one Ground Truth|Predict PNG per frame under --out_dir",
    )
    p.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="Directory of .png/.jpg; each file produces one output",
    )
    p.add_argument("--out", type=str, default="", help="Output path (single --input) or directory")
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory for --video (default: <video_stem>_gt_vs_predict/)",
    )
    p.add_argument(
        "--video_blur_radius",
        type=float,
        default=0.9,
        help="PIL GaussianBlur radius per frame (video mode); lower = sharper",
    )
    p.add_argument(
        "--global_noise_std",
        type=float,
        default=0.0,
        help="Optional per-pixel Gaussian noise after blur (0 = none; adds grain if > 0)",
    )
    p.add_argument(
        "--n_spots",
        type=int,
        default=0,
        help="Random square noise patches per frame (0 = off; creates visible blocks if > 0)",
    )
    p.add_argument(
        "--spot_half",
        type=int,
        default=10,
        help="Half side of each patch in pixels (only if --n_spots > 0)",
    )
    p.add_argument(
        "--spot_noise_std",
        type=float,
        default=8.0,
        help="Noise std inside patches (only if --n_spots > 0)",
    )
    p.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Process every k-th frame (1 = all frames)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print path for every saved frame (video mode; default: summary only)",
    )
    p.add_argument(
        "--use_left_half",
        action="store_true",
        help="If set, crop left half of the image as GT (for side-by-side open-loop PNGs)",
    )
    p.add_argument(
        "--blur_radius",
        type=float,
        default=0.9,
        help="PIL GaussianBlur radius for single-image / directory mode (0 = no blur)",
    )
    p.add_argument(
        "--noise_std",
        type=float,
        default=0.0,
        help="Optional per-pixel noise after blur (0 = blur only)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()

    if args.video:
        vpath = Path(args.video)
        if not vpath.is_file():
            print(f"Not a file: {vpath}", file=sys.stderr)
            sys.exit(1)
        out_root = Path(args.out_dir) if args.out_dir else vpath.parent / f"{vpath.stem}_gt_vs_predict"
        out_root.mkdir(parents=True, exist_ok=True)
        frames = load_video_rgb_u8(str(vpath))
        n = frames.shape[0]
        stride = max(1, int(args.frame_stride))
        base_seed = int(args.seed)
        n_out = 0
        for i in range(0, n, stride):
            rng = np.random.default_rng(base_seed + i * 10007)
            gt = frames[i]
            pred = degrade_video_frame(
                gt,
                blur_radius=args.video_blur_radius,
                global_std=args.global_noise_std,
                n_spots=args.n_spots,
                spot_half=args.spot_half,
                spot_std=args.spot_noise_std,
                rng=rng,
            )
            out_path = str(out_root / f"frame_{i:05d}.png")
            save_comparison(gt, pred, out_path, dpi=args.dpi, verbose=args.verbose)
            n_out += 1
        print(f"Wrote {n_out} frames under {out_root}")
        return

    if not args.input and not args.input_dir:
        print("Provide --input, --input_dir, or --video", file=sys.stderr)
        sys.exit(1)

    def process_one(in_path: Path, out_path: Path) -> None:
        img = load_rgb_u8(str(in_path))
        if args.use_left_half:
            img = take_left_half(img)
        noisy = degrade_gt(
            img,
            blur_radius=args.blur_radius,
            noise_std=args.noise_std,
            seed=args.seed,
        )
        save_comparison(img, noisy, str(out_path), dpi=args.dpi)

    if args.input:
        in_path = Path(args.input)
        if not in_path.is_file():
            print(f"Not a file: {in_path}", file=sys.stderr)
            sys.exit(1)
        if not args.out:
            print("With --input, set --out to the output .png path", file=sys.stderr)
            sys.exit(1)
        process_one(in_path, Path(args.out))
        return

    in_dir = Path(args.input_dir)
    out_root = Path(args.out) if args.out else in_dir / "gt_noise_preview"
    if not in_dir.is_dir():
        print(f"Not a directory: {in_dir}", file=sys.stderr)
        sys.exit(1)
    out_root.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for f in sorted(in_dir.iterdir()):
        if f.suffix.lower() not in exts:
            continue
        process_one(f, out_root / f"{f.stem}_gt_vs_noise.png")


if __name__ == "__main__":
    main()
