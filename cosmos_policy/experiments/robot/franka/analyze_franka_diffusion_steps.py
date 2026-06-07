import argparse
import os
import random
import sys
from typing import Dict, List

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
    CKPT_PATH,
    DATASET_STATS_PATH,
    DATA_ROOT,
    T5_EMBEDDINGS_PATH,
    build_franka_cfg,
    load_episode,
)


def _as_float_vector(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr.reshape(-1)


def _mean_abs(a: np.ndarray) -> float:
    return float(np.mean(np.abs(a)))


def _l1rel(ot: np.ndarray, ot_plus_1: np.ndarray, eps: float = 1e-12) -> float:
    # L1rel = L1(O_t - O_(t+1)) / L1(O_(t+1))
    num = _mean_abs(ot - ot_plus_1)
    den = max(_mean_abs(ot_plus_1), eps)
    return num / den


def _cosine_similarity(ot: np.ndarray, ot_plus_1: np.ndarray, eps: float = 1e-12) -> float:
    a = ot.astype(np.float64, copy=False)
    b = ot_plus_1.astype(np.float64, copy=False)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return float(np.dot(a, b) / denom)


def _l2_distance(ot: np.ndarray, ot_plus_1: np.ndarray) -> float:
    return float(np.linalg.norm(ot - ot_plus_1))


def _minmax_normalize_curve(curve: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    c = curve.astype(np.float32, copy=True)
    cmin = float(np.min(c))
    cmax = float(np.max(c))
    if cmax - cmin < eps:
        return np.zeros_like(c, dtype=np.float32)
    return (c - cmin) / (cmax - cmin)


def _extract_outputs(action_return: Dict) -> Dict[str, np.ndarray]:
    future = action_return.get("future_image_predictions", {}) or {}
    tactile_left = future.get("future_tactile_left", None)
    agent_image = future.get("future_image", None)
    action_latent = action_return.get("action_latent", None)
    mapped_timestep_embeddings = action_return.get("mapped_timestep_embeddings", [])

    if tactile_left is None:
        tactile_left_vec = np.zeros((1,), dtype=np.float32)
    else:
        tactile_left_vec = _as_float_vector(tactile_left)

    if agent_image is None:
        agent_image_vec = np.zeros((1,), dtype=np.float32)
    else:
        agent_image_vec = _as_float_vector(agent_image)

    if action_latent is None:
        action_latent_vec = np.zeros((1,), dtype=np.float32)
    else:
        action_latent_vec = _as_float_vector(action_latent)

    # Use mapped embeddings from Timesteps + TimestepEmbedding (+ norm) inside minimal_v4_dit.
    if not mapped_timestep_embeddings:
        time_embed_vec = np.zeros((1,), dtype=np.float32)
    else:
        # Aggregate variable-length denoise history into fixed-size vector:
        # mean over denoise calls -> (B, T_latent, D_model), then flatten.
        stacked = np.stack([np.asarray(x, dtype=np.float32) for x in mapped_timestep_embeddings], axis=0)
        mean_timestep_emb = np.mean(stacked, axis=0)
        time_embed_vec = _as_float_vector(mean_timestep_emb)

    return {
        "tactile_left_output": tactile_left_vec,
        "agent_image_output": agent_image_vec,
        "action_latent_output": action_latent_vec,
        "time_embedding_output": time_embed_vec,
    }


def _compute_pairwise_metrics(step_outputs: List[Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    keys = list(step_outputs[0].keys())
    num_points = len(step_outputs) - 1
    metrics = {
        "l1rel": {k: np.zeros((num_points,), dtype=np.float32) for k in keys},
        "cosine_similarity": {k: np.zeros((num_points,), dtype=np.float32) for k in keys},
        "l2_distance": {k: np.zeros((num_points,), dtype=np.float32) for k in keys},
    }
    for k in keys:
        for t in range(num_points):
            ot = step_outputs[t][k]
            ot_plus_1 = step_outputs[t + 1][k]
            metrics["l1rel"][k][t] = _l1rel(ot, ot_plus_1)
            metrics["cosine_similarity"][k][t] = _cosine_similarity(ot, ot_plus_1)
            metrics["l2_distance"][k][t] = _l2_distance(ot, ot_plus_1)
    return metrics


def _plot_metric(step_axis: np.ndarray, metric_curves: Dict[str, np.ndarray], out_path: str, y_label: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    plt.plot(step_axis, metric_curves["tactile_left_output"], marker="o", label="tactile latent")
    plt.plot(step_axis, metric_curves["agent_image_output"], marker="o", label="image latent")
    plt.plot(step_axis, metric_curves["action_latent_output"], marker="o", label="action latent")
    plt.plot(step_axis, metric_curves["time_embedding_output"], marker="o", label="time-step embedding")
    plt.xlabel("Diffusion Step")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze diffusion O_t vs O_(t+1) with L1rel, cosine similarity, and L2 distance."
    )
    ep = "episode_0"
    train_dir = f"{DATA_ROOT}/train"
    parser.add_argument("--hdf5", type=str, default=f"{train_dir}/{ep}.hdf5")
    parser.add_argument("--cam_front", type=str, default=f"{train_dir}/{ep}_cam_front.mp4")
    parser.add_argument("--cam_high", type=str, default=f"{train_dir}/{ep}_cam_high.mp4")
    parser.add_argument("--tactile_left", type=str, default=f"{train_dir}/{ep}_tactile_rectify_left.mp4")
    parser.add_argument("--tactile_right", type=str, default=f"{train_dir}/{ep}_tactile_rectify_right.mp4")
    parser.add_argument("--out_dir", type=str, default="./diffusion_step_analysis")
    parser.add_argument("--num_random_frames", type=int, default=50)
    parser.add_argument("--max_diffusion_step", type=int, default=10, help="Evaluate O_1..O_10 using num_steps=1..10")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for p in [args.hdf5, args.cam_front, args.cam_high]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Not a file: {p}")
    os.makedirs(args.out_dir, exist_ok=True)

    use_tactile_mp4 = (
        args.tactile_left
        and args.tactile_right
        and os.path.isfile(args.tactile_left)
        and os.path.isfile(args.tactile_right)
    )

    print("Loading episode...")
    qpos, _, task_name, cam_front, cam_high, tactile_left, tactile_right = load_episode(
        args.hdf5,
        args.cam_front,
        args.cam_high,
        args.tactile_left if use_tactile_mp4 else None,
        args.tactile_right if use_tactile_mp4 else None,
    )
    T = qpos.shape[0]
    use_tactile = tactile_left is not None and tactile_right is not None
    print(f"Episode length T={T}, task_name={task_name!r}, use_tactile={use_tactile}")

    print("Loading model...")
    cfg = build_franka_cfg(use_tactile=use_tactile)
    if not CKPT_PATH or not os.path.exists(CKPT_PATH):
        raise RuntimeError("Set FRANKA_COSMOS_CKPT to a valid checkpoint directory first.")
    cfg.ckpt_path = CKPT_PATH
    model, _ = get_model(cfg)

    dataset_stats = {}
    if DATASET_STATS_PATH and os.path.exists(DATASET_STATS_PATH):
        dataset_stats = load_dataset_stats(DATASET_STATS_PATH)
    else:
        cfg.unnormalize_actions = False
    if T5_EMBEDDINGS_PATH and os.path.exists(T5_EMBEDDINGS_PATH):
        init_t5_text_embeddings_cache(T5_EMBEDDINGS_PATH)

    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(range(T), k=min(args.num_random_frames, T)))
    print(f"Randomly selected frame indices: {chosen}")

    eval_steps = np.arange(1, args.max_diffusion_step + 1, dtype=np.int32)  # 1..10 by default
    step_axis = np.arange(1, args.max_diffusion_step, dtype=np.int32)  # t=1..9 for O_t and O_(t+1)
    per_frame_metrics: List[Dict[str, Dict[str, np.ndarray]]] = []

    for frame_idx in chosen:
        obs = {
            "primary_image": cam_front[frame_idx],
            "wrist_image": cam_high[frame_idx],
            "proprio": qpos[frame_idx],
        }
        if use_tactile:
            obs["tactile_left_image"] = tactile_left[frame_idx]
            obs["tactile_right_image"] = tactile_right[frame_idx]

        step_outputs = []
        print(f"[frame={frame_idx}] evaluating diffusion steps...")
        # Sanity check for frame condition actually changing.
        print(
            f"[frame={frame_idx}] qpos_norm={float(np.linalg.norm(qpos[frame_idx])):.6f}, "
            f"primary_mean={float(np.mean(cam_front[frame_idx])):.4f}, "
            f"wrist_mean={float(np.mean(cam_high[frame_idx])):.4f}"
        )
        for step_k in eval_steps:
            num_steps = int(step_k)
            action_return = get_action(
                cfg=cfg,
                model=model,
                dataset_stats=dataset_stats,
                obs=obs,
                task_label_or_embedding=task_name,
                seed=args.seed,
                randomize_seed=False,
                num_denoising_steps_action=num_steps,
                generate_future_state_and_value_in_parallel=True,
            )
            step_outputs.append(_extract_outputs(action_return))

        per_frame_metrics.append(_compute_pairwise_metrics(step_outputs))

    if not per_frame_metrics:
        raise RuntimeError("No frame metrics computed.")

    keys = list(per_frame_metrics[0]["l1rel"].keys())
    avg_metrics: Dict[str, Dict[str, np.ndarray]] = {}
    for metric_name in ["l1rel", "cosine_similarity", "l2_distance"]:
        avg_metrics[metric_name] = {}
        for k in keys:
            stacked = np.stack([m[metric_name][k] for m in per_frame_metrics], axis=0)  # [N_frames, N_steps-1]
            avg_metrics[metric_name][k] = np.mean(stacked, axis=0).astype(np.float32)

    # Normalize only L2 distance curves into [0, 1] for clearer cross-latent comparison.
    avg_metrics["l2_distance"] = {
        k: _minmax_normalize_curve(v) for k, v in avg_metrics["l2_distance"].items()
    }

    l1_fig_path = os.path.join(args.out_dir, "avg50_diffusion_l1rel.png")
    cos_fig_path = os.path.join(args.out_dir, "avg50_diffusion_cosine_similarity.png")
    l2_fig_path = os.path.join(args.out_dir, "avg50_diffusion_l2_distance.png")
    _plot_metric(step_axis, avg_metrics["l1rel"], l1_fig_path, "L1rel")
    _plot_metric(step_axis, avg_metrics["cosine_similarity"], cos_fig_path, "Cosine Similarity")
    _plot_metric(step_axis, avg_metrics["l2_distance"], l2_fig_path, "L2 Distance (normalized)")
    print(f"Saved figure: {l1_fig_path}")
    print(f"Saved figure: {cos_fig_path}")
    print(f"Saved figure: {l2_fig_path}")

    result_path = os.path.join(args.out_dir, "avg50_diffusion_metrics_results.json")
    import json

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_frames": chosen,
                "step_axis": step_axis.tolist(),
                "num_random_frames": len(chosen),
                "formula_l1rel": "L1rel_t = L1(O_t - O_(t+1)) / L1(O_(t+1))",
                "formula_cosine_similarity": "cos(O_t, O_(t+1))",
                "formula_l2_distance": "||O_t - O_(t+1)||_2",
                "l2_normalization": "Per-latent min-max normalization to [0,1] over diffusion steps after 50-frame averaging.",
                "avg_curves": {metric: {k: v.tolist() for k, v in curves.items()} for metric, curves in avg_metrics.items()},
            },
            f,
            indent=2,
        )
    print(f"Saved result json: {result_path}")
    print("Done.")


if __name__ == "__main__":
    main()
