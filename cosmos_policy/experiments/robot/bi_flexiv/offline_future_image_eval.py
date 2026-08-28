"""Run bi_flexiv policy inference on a LeRobot episode and compare future images to GT."""

from __future__ import annotations

import argparse
import os
import time

from cosmos_policy.datasets.lerobot_bi_flexiv_dataset import LeRobotBiFlexivDataset
from cosmos_policy.experiments.robot.bi_flexiv.bi_flexiv_policy import (
    CHUNK_SIZE,
    DreamTacBiFlexivPolicy,
    DreamTacBiFlexivPolicyConfig,
    prepare_camera_images,
)
from cosmos_policy.experiments.robot.bi_flexiv.future_image_eval import FutureImageEvaluationWriter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="LeRobot v3 dataset root")
    parser.add_argument("--episode-index", type=int, default=0, help="LeRobot episode id")
    parser.add_argument("--start", type=int, default=0, help="First episode timestep")
    parser.add_argument("--end", type=int, default=-1, help="Exclusive end; -1 uses the last full future horizon")
    parser.add_argument("--stride", type=int, default=CHUNK_SIZE, help="Inference stride in dataset timesteps")
    parser.add_argument("--include-padded-future", action="store_true", help="Also evaluate starts whose t+20 GT is clamped")
    parser.add_argument("--output-dir", default="./bi_flexiv_future_comparisons")
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "DREAMTAC_CONFIG", "cosmos_predict2_2b_480p_lerobot_bi_flexiv_tactile__inference_only"
        ),
    )
    parser.add_argument(
        "--config-file", default=os.environ.get("DREAMTAC_CONFIG_FILE", "cosmos_policy/config/config.py")
    )
    parser.add_argument("--checkpoint", default=os.environ.get("DREAMTAC_CKPT", ""))
    parser.add_argument("--wan-vae", default=os.environ.get("DREAMTAC_WAN_VAE", ""))
    parser.add_argument("--stats", default=os.environ.get("DREAMTAC_STATS", ""))
    parser.add_argument("--t5-embeddings", default=os.environ.get("DREAMTAC_T5", ""))
    parser.add_argument("--default-prompt", default=os.environ.get("DREAMTAC_DEFAULT_PROMPT", ""))
    parser.add_argument(
        "--num-denoising-steps",
        type=int,
        default=int(os.environ.get("DREAMTAC_NUM_DENOISING_STEPS", "5")),
        help="Diffusion sampling steps; CLI overrides DREAMTAC_NUM_DENOISING_STEPS, whose fallback is 5.",
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("DREAMTAC_SEED", "0")))
    parser.add_argument(
        "--normalization-mode",
        choices=("q99", "min_max"),
        default=os.environ.get("DREAMTAC_NORMALIZATION_MODE", "q99"),
    )
    parser.add_argument("--jpeg-quality", type=int, default=None)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-normalized-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-prompt-fallback", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _require(parser: argparse.ArgumentParser, value: str, flag: str) -> str:
    if not value:
        parser.error(f"{flag} is required (or set the corresponding DREAMTAC_* environment variable)")
    return value


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.num_denoising_steps <= 0:
        parser.error("--num-denoising-steps must be positive")

    config = DreamTacBiFlexivPolicyConfig(
        checkpoint_path=_require(parser, args.checkpoint, "--checkpoint"),
        dataset_stats_path=_require(parser, args.stats, "--stats"),
        t5_embeddings_path=_require(parser, args.t5_embeddings, "--t5-embeddings"),
        default_prompt=args.default_prompt,
        config_name=args.config,
        config_file=args.config_file,
        wan_vae_path=args.wan_vae or None,
        num_denoising_steps=args.num_denoising_steps,
        seed=args.seed,
        center_crop=args.center_crop,
        jpeg_quality=args.jpeg_quality,
        clip_normalized_actions=args.clip_normalized_actions,
        action_output="observation_relative",
        normalization_mode=args.normalization_mode,
        allow_prompt_fallback=args.allow_prompt_fallback,
        decode_future_images=True,
    )

    dataset = LeRobotBiFlexivDataset(
        data_dir=os.path.abspath(os.path.expanduser(args.data_dir)),
        chunk_size=CHUNK_SIZE,
        final_image_size=config.image_size,
        t5_text_embeddings_path="",
        normalize_images=False,
        normalize_actions=False,
        normalize_proprio=False,
        use_image_aug=False,
        use_stronger_image_aug=False,
    )
    try:
        episode = next((item for item in dataset.episodes if item.episode_index == args.episode_index), None)
        if episode is None:
            parser.error(f"Episode id {args.episode_index} does not exist in {args.data_dir}")
        default_end = episode.length if args.include_padded_future else max(0, episode.length - CHUNK_SIZE)
        end = default_end if args.end < 0 else min(args.end, episode.length)
        start = max(0, args.start)
        if start >= end:
            parser.error(f"Empty timestep range [{start}, {end}) for episode length {episode.length}")

        print(
            "Offline inference settings: "
            f"denoising_steps={config.num_denoising_steps}, seed={config.seed}, stride={args.stride}",
            flush=True,
        )
        policy = DreamTacBiFlexivPolicy(config)
        writer = FutureImageEvaluationWriter(args.output_dir)
        timesteps = range(start, end, args.stride)
        completed = 0
        for timestep in timesteps:
            sample = dataset.get_inference_sample(args.episode_index, timestep)
            if sample["is_padded_future"] and not args.include_padded_future:
                continue
            started = time.perf_counter()
            response = policy.infer(sample["observation"])
            elapsed = time.perf_counter() - started
            predictions = response.get("future_images")
            if not predictions:
                raise RuntimeError("Policy did not return decoded future_images")
            gt_images = prepare_camera_images(
                sample["future_images"],
                image_size=config.image_size,
                center_crop=config.center_crop,
                jpeg_quality=config.jpeg_quality,
            )
            sample_tag = (
                f"episode{args.episode_index:06d}_start{sample['start_timestep']:06d}_"
                f"target{sample['future_timestep']:06d}_steps{config.num_denoising_steps:03d}_seed{config.seed}"
            )
            writer.save_comparison(sample_tag, predictions, gt_images)
            completed += 1
            print(f"[{completed}] saved {sample_tag} ({elapsed:.2f}s)", flush=True)
        print(f"Saved {completed} comparisons to {writer.output_dir}")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
