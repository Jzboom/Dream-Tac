# Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation

Official implementation of **Dream-Tac**, a unified tactile world action model that jointly predicts future visual observations, future tactile signals, and robot actions for contact-rich manipulation.

**Authors:** Yunfan Lou\*, Yifan Ye\*†, Yankai Fu\*, Jun Cen, Xiaowei Chi, Yaoxu Lyu, Peidong Jia, Sirui Han, Zhihe Lu, Shanghang Zhang‡  
**Affiliations:** Peking University, HKUST, Nanjing University  
\*Equal contribution · †Project leader · ‡Corresponding author

📄 Paper: [`arxiv_tactile.pdf`](arxiv_tactile.pdf)  
🔗 Code: [https://github.com/LYFCLOUDFAN/Dream-Tac](https://github.com/LYFCLOUDFAN/Dream-Tac)

---

## Overview

World action models leverage predictive future observations to guide action generation, but vision alone often fails in contact-rich manipulation where critical cues come from physical interaction. **Dream-Tac** extends world action modeling with tactile sensing and introduces:

1. **Contact-Aware Self-Attention (CASA)** — a gated attention bias that amplifies tactile influence when contact dynamics are salient.
2. **Contact gate** — computed from frame-to-frame tactile variation (left/right fingertip sensors), suppressing noise while emphasizing contact events.
3. **Dual-level acceleration** — FlashBias-style fused attention for faster training, and diffusion-step caching for faster inference.

Built on top of [Cosmos Policy](https://arxiv.org/abs/2601.16163), Dream-Tac maps visual and tactile observations into a shared latent space and jointly denoises future images, future tactile frames, and action chunks within one diffusion transformer.

### Key results (from paper)

| Setting | Avg. success rate |
|---|---:|
| Visual WAM (Cosmos Policy baseline) | 51.7% |
| Visuo-tactile WAM | 74.2% |
| **Dream-Tac (visuo-tactile + CASA bias)** | **83.3%** |

Across six real-world Franka tasks, Dream-Tac improves over Cosmos Policy by **+31.6%** average success rate, with up to **2.9×** training speedup and **1.8×** inference speedup.

---

## Repository structure

```
cosmos_policy/
├── _src/predict2/networks/tactile_self_attn_chunked.py   # CASA + FlashBias-style SDPA backend
├── utils/tactile_self_attn_gate.py                       # contact gate g_t from tactile frame deltas
├── datasets/franka_dataset.py                            # visuo-tactile dataset loader
├── models/policy_video2world_model_openloop_residual_cache.py  # diffusion-step cache (inference)
├── experiments/robot/openloop_hard_residual_cache.py     # open-loop cache utilities
├── experiments/robot/franka/                             # preprocessing, open-loop eval, deployment
├── config/experiment/cosmos_policy_experiment_configs.py # task experiment definitions
└── scripts/train.py                                      # training entry point
```

---

## Supported tasks

Dream-Tac is evaluated on six language-conditioned contact-rich manipulation tasks on a Franka Emika Panda with two RealSense cameras and two Xense Photon tactile sensors.

| Task (paper) | Experiment config name |
|---|---|
| Pick Baguette | `cosmos_predict2_2b_480p_franka_pick_and_place_baguette` |
| Clean Whiteboard | `cosmos_predict2_2b_480p_franka_whiteboard` |
| Peel Cucumber | `cosmos_predict2_2b_480p_franka_shave_cucumber_20260321` |
| Play Mahjong | `cosmos_predict2_2b_480p_franka_hupai_tactile` |
| Cut Banana | `cosmos_predict2_2b_480p_franka_cut_banana_20260321` |
| Insert USB | *not included in this release* |

> **Note:** The Insert USB task is described in the paper but its experiment config is not shipped in this repository. The other five tasks are fully supported.

Ablation configs (no tactile / no tactile image aug) are also available, e.g. `cosmos_predict2_2b_480p_franka_cut_banana_20260321_no_tactile`.

---

## Quick start

Follow these steps in order:

1. [Environment setup](#1-environment-setup)
2. [Data preparation](#2-data-preparation)
3. [Training](#3-training)
4. [Evaluation (open-loop)](#4-evaluation-open-loop)

---

## 1. Environment Setup

Dream-Tac follows the same environment setup as the official [Cosmos Policy](https://github.com/NVlabs/cosmos-policy) release.

### 1.1 Docker-based setup (recommended)

Build the Docker image from the repository root:

```bash
docker build -t dream-tac docker
```

Launch the container:

```bash
docker run \
  -u root \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/home/cosmos/.cache \
  -v $(pwd):/workspace \
  --gpus all \
  --ipc=host \
  -it \
  --rm \
  -w /workspace \
  --entrypoint bash \
  dream-tac
```

Notes:
- `--ipc=host` is important for PyTorch data loading stability.
- If `--ipc=host` is not allowed, use `--shm-size 32g` instead.

### 1.2 Basic setup (inside container or host env)

```bash
cd /path/to/Dream-Tac
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Recommended runtime:
- Linux
- Python 3.10
- CUDA-enabled PyTorch

### 1.3 Dependency installation

From the repository root, using `uv` (recommended):

```bash
cd cosmos_policy
uv sync --extra cu128 --group franka
uv run python -c "import torch; print(torch.__version__)"
```

### 1.4 Sanity checks

```bash
python -m cosmos_policy.scripts.train --help
python -m cosmos_policy.datasets.save_franka_t5_text_embeddings --help
python -m cosmos_policy.experiments.robot.franka.run_franka_openloop --help
```

---

## 2. Data Preparation

### 2.1 Preprocess raw Franka tactile episodes

Entry point: `cosmos_policy/experiments/robot/franka/preprocess_tactile_franka_data.py`

```bash
python -m cosmos_policy.experiments.robot.franka.preprocess_tactile_franka_data \
  --input_dir /path/to/raw_data \
  --output_dir /path/to/preprocessed_data \
  --task_name your_task_name
```

### 2.2 Generate T5 instruction embeddings (required)

Entry point: `cosmos_policy/datasets/save_franka_t5_text_embeddings.py`

```bash
python -m cosmos_policy.datasets.save_franka_t5_text_embeddings \
  --data_dir /path/to/preprocessed_data
```

Expected artifacts in the processed data directory:
- `t5_embeddings.pkl`
- `dataset_statistics_franka.json`
- `preprocessing_metadata.json`
- `train/` and `val/` splits

Update the dataset paths in `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py` to point to your preprocessed data before training.

---

## 3. Training

| Component | Path |
|---|---|
| Training entry | `cosmos_policy/scripts/train.py` |
| Config router | `cosmos_policy/config/config.py` |
| Experiment definitions | `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py` |

### 3.1 Multi-GPU training

```bash
export MAGINAIRE_OUTPUT_ROOT=/path/to/checkpoints

uv run --extra cu128 --group franka --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_franka_cut_banana_20260321"
```

### 3.2 Dry-run configuration validation

```bash
uv run --extra cu128 --group franka --python 3.10 \
  python -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py --dryrun -- \
  experiment="cosmos_predict2_2b_480p_franka_cut_banana_20260321"
```

### 3.3 CASA backend selection

Contact-aware self-attention supports three backends (set via environment variable):

```bash
export COSMOS_TACTILE_SELF_ATTN_BACKEND=flashbias_sdpa  # default, recommended
# alternatives: sdpa | eager
```

Implementation: `cosmos_policy/_src/predict2/networks/tactile_self_attn_chunked.py`

---

## 4. Evaluation (Open-Loop)

Entry point: `cosmos_policy/experiments/robot/franka/run_franka_openloop.py`

### 4.1 Set runtime paths

```bash
export FRANKA_COSMOS_CONFIG=cosmos_predict2_2b_480p_franka_cut_banana_20260321
export FRANKA_COSMOS_CKPT=/path/to/checkpoints/iter_000003000
export FRANKA_DATASET_STATS_PATH=/path/to/preprocessed_data/dataset_statistics_franka.json
export FRANKA_T5_EMBEDDINGS_PATH=/path/to/preprocessed_data/t5_embeddings.pkl
```

### 4.2 Run open-loop evaluation

```bash
python -m cosmos_policy.experiments.robot.franka.run_franka_openloop \
  --hdf5 /path/to/episode_0.hdf5 \
  --cam_front /path/to/episode_0_cam_front.mp4 \
  --cam_high /path/to/episode_0_cam_high.mp4 \
  --tactile_left /path/to/episode_0_tactile_rectify_left.mp4 \
  --tactile_right /path/to/episode_0_tactile_rectify_right.mp4 \
  --out_dir ./openloop_out \
  --future_pred_eval
```

Typical outputs:
- action prediction traces
- future visual / tactile state visualizations
- JSON summaries (e.g., IoU-style future prediction metrics)

For faster inference, use the diffusion-step cache model:
- `cosmos_policy/models/policy_video2world_model_openloop_residual_cache.py`
- `cosmos_policy/experiments/robot/openloop_hard_residual_cache.py`

---

## Acknowledgements

This project builds upon [Cosmos Policy](https://arxiv.org/abs/2601.16163) (NVIDIA) and [Cosmos Predict2](https://github.com/nvidia-cosmos/cosmos-predict2). The CASA training acceleration uses a self-contained re-implementation of the FlashBias-style attention trick ([FlashBias, NeurIPS 2025](https://arxiv.org/pdf/2505.12044)).

---

## License

This repository includes code derived from Cosmos Policy, which is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) and [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for details.

---

## Citation

If you find Dream-Tac useful, please cite our paper:

```bibtex
@article{lou2026dreamtac,
  title={Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation},
  author={Lou, Yunfan and Ye, Yifan and Fu, Yankai and Cen, Jun and Chi, Xiaowei and Lyu, Yaoxu and Jia, Peidong and Han, Sirui and Lu, Zhihe and Zhang, Shanghang},
  year={2026}
}
```

Also consider citing Cosmos Policy if you use the underlying world action model framework:

```bibtex
@article{kim2026cosmos,
  title={Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning},
  author={Kim, Moo Jin and Gao, Yihuai and Lin, Tsung-Yi and Lin, Yen-Chen and Ge, Yunhao and Lam, Grace and Liang, Percy and Song, Shuran and Liu, Ming-Yu and Finn, Chelsea and Gu, Jinwei},
  journal={arXiv preprint arXiv:2601.16163},
  year={2026}
}
```
