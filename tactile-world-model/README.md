# Tactile World Model for Franka Manipulation

This repository contains training and open-loop evaluation pipelines for Franka-based tactile tasks.  
For reproducibility, the instructions below follow the exact order:

1. Environment setup  
2. Data preparation  
3. Training  
4. Evaluation

## 1. Environment Setup

This project follows the same environment setup as the official Cosmos Policy release.

### 1.1 Docker-based setup (recommended)

Build the Docker image from repository root:

```bash
docker build -t cosmos-policy docker
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
  cosmos-policy
```

Notes:
- `--ipc=host` is important for PyTorch data loading stability.  
- If `--ipc=host` is not allowed in your environment, use `--shm-size 32g` instead.

### 1.2 Basic setup (inside container or host env)

```bash
cd /path/to/tactile-world-model
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Recommended runtime:
- Linux
- Python 3.10
- CUDA-enabled PyTorch

### 1.3 Dependency installation

If you use `uv` (recommended, consistent with the original workflow):

```bash
uv sync
uv run python -c "import torch; print(torch.__version__)"
```

### 1.4 Sanity checks

```bash
python -m cosmos_policy.scripts.train --help
python -m cosmos_policy.datasets.save_franka_t5_text_embeddings --help
python -m cosmos_policy.experiments.robot.franka.run_franka_openloop --help
```

## 2. Data Preparation

### 2.1 Preprocess raw Franka tactile episodes

Preprocessing entry point:
- `cosmos_policy/experiments/robot/franka/preprocess_tactile_franka_data.py`

Example command:

```bash
python -m cosmos_policy.experiments.robot.franka.preprocess_tactile_franka_data \
  --input_dir /path/to/raw_data \
  --output_dir /path/to/preprocessed_data \
  --task_name your_task_name
```

### 2.2 Generate T5 instruction embeddings (required)

Embedding generation entry point:
- `cosmos_policy/datasets/save_franka_t5_text_embeddings.py`

```bash
python -m cosmos_policy.datasets.save_franka_t5_text_embeddings \
  --data_dir /path/to/preprocessed_data
```

Expected artifacts in your processed data directory:
- `t5_embeddings.pkl`
- `dataset_statistics_franka.json`
- `preprocessing_metadata.json`
- `train/` and `val/` splits (or equivalent split layout used by your configs)

## 3. Training

Training entry point:
- `cosmos_policy/scripts/train.py`

Config router:
- `cosmos_policy/config/config.py`

Experiment definitions:
- `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py`

### 3.1 Multi-GPU training (ALOHA-style command format)

```bash
# Optional: set if your configs reference BASE_DATASETS_DIR
export BASE_DATASETS_DIR=/PATH/TO/BASE/DATASETS/DIRECTORY

# Set nproc_per_node to your available GPU count
uv run --extra cu128 --group franka --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_franka_cut_banana_20260321"
```

Common tactile experiment names:
- `cosmos_predict2_2b_480p_franka_hupai_tactile`
- `cosmos_predict2_2b_480p_franka_shave_cucumber_20260321`
- `cosmos_predict2_2b_480p_franka_cut_banana_20260321`

### 3.2 Dry-run configuration validation

```bash
uv run --extra cu128 --group franka --python 3.10 \
  python -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py --dryrun -- \
  experiment="cosmos_predict2_2b_480p_franka_cut_banana_20260321"
```

## 4. Evaluation (Open-Loop)

Evaluation entry point:
- `cosmos_policy/experiments/robot/franka/run_franka_openloop.py`

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

Typical outputs include:
- action prediction traces
- future-state visualizations
- JSON summaries (e.g., IoU-style future prediction metrics)


