# Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation

Official implementation of **Dream-Tac**, a unified tactile world action model that jointly predicts future visual observations, future tactile signals, and robot actions for contact-rich manipulation.

**Authors:** Yunfan Lou\*, Yifan Ye\*†, Yankai Fu\*, Jun Cen, Xiaowei Chi, Yaoxu Lyu, Peidong Jia, Sirui Han, Zhihe Lu, Shanghang Zhang‡  
**Affiliations:** Peking University, HKUST, Nanjing University  
\*Equal contribution · †Project leader · ‡Corresponding author

📄 Paper: [arXiv:2606.08737](https://arxiv.org/abs/2606.08737)  
🔗 Code: [https://github.com/LYFCLOUDFAN/Dream-Tac](https://github.com/LYFCLOUDFAN/Dream-Tac)

---

## Overview

World action models leverage predictive future observations to guide action generation, but vision alone often fails in contact-rich manipulation where critical cues come from physical interaction. **Dream-Tac** extends world action modeling with tactile sensing and introduces:

1. **Contact-Aware Self-Attention (CASA)** — a gated attention bias that amplifies tactile influence when contact dynamics are salient.
2. **Contact gate** — computed from frame-to-frame tactile variation, suppressing noise while emphasizing contact events.
3. **Dual-level acceleration** — FlashBias-style fused attention for faster training and diffusion-step caching for faster inference.

Built on top of [Cosmos Policy](https://arxiv.org/abs/2601.16163), Dream-Tac maps visual and tactile observations into a shared latent space and jointly denoises future images, future tactile frames, and action chunks within one diffusion transformer.

### Key results (from paper)

| Setting | Avg. success rate |
|---|---:|
| Visual WAM (Cosmos Policy baseline) | 51.7% |
| Visuo-tactile WAM | 74.2% |
| **Dream-Tac (visuo-tactile + CASA bias)** | **83.3%** |

Across six real-world contact-rich tasks, Dream-Tac improves over Cosmos Policy by **+31.6%** average success rate, with up to **2.9×** training speedup and **1.8×** inference speedup.

---

## Repository structure

```text
cosmos_policy/
├── config/local_paths.py                              # default sibling-directory paths
├── config/experiment/lerobot_bi_flexiv_experiment_configs.py
├── datasets/lerobot_bi_flexiv_dataset.py              # LeRobot bi_flexiv dataset and statistics
├── datasets/save_lerobot_t5_text_embeddings.py        # task-text embedding generation
├── _src/predict2/networks/tactile_self_attn_chunked.py
├── models/policy_video2world_model.py
├── experiments/robot/bi_flexiv/                       # local policy server and protocol
└── scripts/train.py                                    # training entry point
```

---

## Supported LeRobot layout

The active training configuration targets a LeRobot v3-style dual-arm dataset with:

- a 20D robot state;
- a 20D action;
- three RGB views: head, left wrist, and right wrist;
- four raw tactile views: two sensors per arm, merged into one condition image per arm;
- task text stored in `meta/tasks.parquet`.

The active model uses 11 latent slots:
`blank, proprio, 3 current RGB, 2 merged current tactile, action, 3 future RGB`.
Future proprioception and future tactile prediction are not part of this policy.

The experiment name is:

```text
cosmos_predict2_2b_480p_lerobot_bi_flexiv_wam_11slot
```

The former `earbud` names remain only as import/config-name aliases. This
11-slot policy does not support checkpoints trained with the former 18-slot
layout. New statistics are written as
`dataset_statistics_lerobot_bi_flexiv.json`; if only the former statistics
filename exists, the dataset loader reads it automatically.

The dataset directory can be changed at launch time with:

```text
lerobot_dataset_path=../your_lerobot_dataset
```

---

## Quick start

Follow these steps in order:

1. [Environment setup](#1-environment-setup)
2. [Directory layout](#2-directory-layout)
3. [LeRobot data preparation](#3-lerobot-data-preparation)
4. [Training](#4-training)
5. [Local inference server](#5-local-inference-server)

All commands below are run from the `Dream-Tac` repository root unless stated otherwise.

---

## 1. Environment setup

Recommended runtime:

- Linux
- Python 3.10
- CUDA-enabled PyTorch
- CUDA 12.8 dependencies for the provided `cu128` extra

### 1.1 Install into an existing virtual environment

```bash
cd /path/to/Dream-Tac/cosmos_policy
uv pip install -e ".[cu128]"
cd ..
```

### 1.2 Create a project environment with uv

```bash
cd /path/to/Dream-Tac/cosmos_policy
uv sync --extra cu128 --python 3.10
cd ..
```

### 1.3 Docker

```bash
docker build -t dream-tac docker

docker run \
  -u root \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $(pwd)/..:/workspace \
  --gpus all \
  --ipc=host \
  -it \
  --rm \
  -w /workspace/Dream-Tac \
  --entrypoint bash \
  dream-tac
```

If `--ipc=host` is unavailable, use `--shm-size 32g`.

### 1.4 Sanity checks

```bash
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
python -m cosmos_policy.scripts.train --help
python -m cosmos_policy.datasets.save_lerobot_t5_text_embeddings --help
```

---

## 2. Directory layout

The default configuration derives paths from the repository location. The expected sibling-directory layout is:

```text
workspace/
├── Dream-Tac/                              # run commands here
├── pick_up_cube_0713/                      # example LeRobot dataset
└── checkpoints/
    ├── Cosmos-Predict2-2B-Video2World/
    │   ├── model-480p-16fps.pt
    │   └── tokenizer/
    │       └── tokenizer.pth
    ├── google-t5/
    │   └── t5-11b/
    └── <training outputs>/
```

Defaults are defined in `cosmos_policy/config/local_paths.py`. Environment variables remain optional overrides, but are not required for the layout above.

---

## 3. LeRobot data preparation

### 3.1 Expected dataset files

```text
your_lerobot_dataset/
├── data/
├── videos/
├── meta/
│   ├── info.json
│   ├── tasks.parquet
│   └── episodes/
├── t5_embeddings.pkl
└── dataset_statistics_lerobot_bi_flexiv.json
```

### 3.2 Generate T5 task embeddings

The task text is read from `meta/tasks.parquet`. Generate embeddings once for each dataset/task-text combination:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  -m cosmos_policy.datasets.save_lerobot_t5_text_embeddings \
  --data_dir ../pick_up_cube_0713 \
  --t5_model_path ../checkpoints/google-t5/t5-11b \
  --device cuda:0
```

Output:

```text
../pick_up_cube_0713/t5_embeddings.pkl
```

Regenerate this file whenever the exact task text changes.

### 3.3 Generate normalization statistics

Run statistics generation once in a single process before multi-GPU training:

```bash
python -c 'from cosmos_policy.datasets.lerobot_bi_flexiv_dataset import LeRobotBiFlexivDataset as D; D(data_dir="../pick_up_cube_0713", t5_text_embeddings_path="")'
```

Output:

```text
../pick_up_cube_0713/dataset_statistics_lerobot_bi_flexiv.json
```

The statistics use the training defaults:

```text
chunk_size=30
gripper_start_idx=18
normalization_mode=q99
```

The action chunk and future RGB target use the same horizon. At the dataset's
30 Hz sampling rate, the future RGB target is the single frame at `t+30`
(approximately one second after the current observation).

If the statistics file is missing, training can generate it automatically. Pre-generating it avoids every distributed rank computing the same statistics during startup.

### 3.4 Verify required assets

```bash
ls -lh \
  ../pick_up_cube_0713/t5_embeddings.pkl \
  ../pick_up_cube_0713/dataset_statistics_lerobot_bi_flexiv.json \
  ../checkpoints/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt \
  ../checkpoints/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth
```

---

## 4. Training

| Component | Path |
|---|---|
| Training entry | `cosmos_policy/scripts/train.py` |
| Config router | `cosmos_policy/config/config.py` |
| LeRobot experiment | `cosmos_policy/config/experiment/lerobot_bi_flexiv_experiment_configs.py` |
| Default local paths | `cosmos_policy/config/local_paths.py` |

### 4.1 Dry-run configuration validation

```bash
python -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py \
  --dryrun -- \
  experiment=cosmos_predict2_2b_480p_lerobot_bi_flexiv_wam_11slot \
  lerobot_dataset_path=../pick_up_cube_0713 \
  dataloader_train.batch_size=1 \
  job.project=cosmos_policy_lerobot_pick_up_cube \
  job.name=pick_up_cube_0713_tactile_8gpu_gbs8_v1
```

The dry run writes and prints the resolved `config.yaml` path without starting training.

### 4.2 Single-node multi-GPU training

This example uses eight GPUs and preserves the original global batch size of eight:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
  --standalone \
  --nproc_per_node=8 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_lerobot_bi_flexiv_wam_11slot \
  lerobot_dataset_path=../pick_up_cube_0713 \
  dataloader_train.batch_size=1 \
  job.project=cosmos_policy_lerobot_pick_up_cube \
  job.name=pick_up_cube_0713_tactile_8gpu_gbs8_v1
```

`--standalone` configures rendezvous automatically for single-node distributed training. The number of visible GPUs must equal `--nproc_per_node`.

### 4.3 Batch-size semantics

`dataloader_train.batch_size` is the per-process batch size:

```text
global batch size = per-process batch size × number of training processes
```

Examples:

| GPUs | Per-process batch | Global batch |
|---:|---:|---:|
| 1 | 8 | 8 |
| 4 | 2 | 8 |
| 8 | 1 | 8 |
| 8 | 8 | 64 |

If changing the global batch size substantially, review the learning rate and scheduler settings.

### 4.4 Dataset and run naming

Change datasets without editing source code:

```text
lerobot_dataset_path=../another_lerobot_dataset
```

Use a unique `job.name` for each new task or independent run. Reusing the same output root, `job.project`, `job.group`, and `job.name` allows the checkpointer to resume automatically from `latest_checkpoint.txt`.

The example above writes to:

```text
../checkpoints/
└── cosmos_policy_lerobot_pick_up_cube/
    └── wam_11slot_finetune/
        └── pick_up_cube_0713_tactile_8gpu_gbs8_v1/
            └── checkpoints/
```

### 4.5 Resume from an explicit distributed checkpoint

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
  --standalone \
  --nproc_per_node=8 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_lerobot_bi_flexiv_wam_11slot \
  lerobot_dataset_path=../pick_up_cube_0713 \
  dataloader_train.batch_size=1 \
  checkpoint.load_path=../checkpoints/path/to/checkpoints/iter_000100000 \
  checkpoint.load_training_state=true \
  trainer.max_iter=120000 \
  job.project=cosmos_policy_lerobot_pick_up_cube \
  job.name=pick_up_cube_0713_tactile_8gpu_gbs8_v1
```

`trainer.max_iter` is the final total iteration, not the number of additional iterations.
Only resume training state from a checkpoint produced by this 11-slot configuration.

### 4.6 CASA attention backend

The default and recommended backend is `flashbias_sdpa`. Override it only when debugging:

```bash
export COSMOS_TACTILE_SELF_ATTN_BACKEND=flashbias_sdpa
# alternatives: sdpa | eager
```

---

## 5. Local inference server

The trained dual-arm policy can be served through the WebSocket/MsgPack interface:

```bash
export DREAMTAC_CKPT=../checkpoints/path/to/checkpoints/iter_XXXXXXXX
export DREAMTAC_WAN_VAE=../checkpoints/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth
export DREAMTAC_STATS=../pick_up_cube_0713/dataset_statistics_lerobot_bi_flexiv.json
export DREAMTAC_T5=../pick_up_cube_0713/t5_embeddings.pkl
export DREAMTAC_DEFAULT_PROMPT='the exact training task text'

python -m cosmos_policy.experiments.robot.bi_flexiv.bi_flexiv_server \
  --config cosmos_predict2_2b_480p_lerobot_bi_flexiv_wam_11slot__inference_only \
  --host 0.0.0.0 \
  --port 8000 \
  --num-denoising-steps 10
```

Ten denoising steps and diffusion-step residual caching are the defaults. Pass
`--no-diffusion-step-cache` only for an uncached comparison. Requests contain a
20D state, three RGB views, four raw tactile views, and a two-value tactile gate.
Configure the client-side `ActionChunkBroker` with `action_horizon=30` to match
the server response.
See `cosmos_policy/experiments/robot/bi_flexiv/README.md` for the full protocol.

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
  journal={arXiv preprint arXiv:2606.08737},
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
