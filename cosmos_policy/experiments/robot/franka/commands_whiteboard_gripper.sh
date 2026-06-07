#!/bin/bash
# 数据集: /share/project/yunfan/tactile_data_preprocessed_gripper/pick_eraser_and_erase_marker_from_whiteboard
# state 6 维, action 7 维 (pose + gripper)

DATA_DIR="/share/project/yunfan/tactile_data_preprocessed_gripper/pick_eraser_and_erase_marker_from_whiteboard"

# 1) 生成 T5 文本编码（若该目录下还没有 t5_embeddings.pkl 则运行一次）
uv run --extra cu128 --group libero --python 3.10 python -m cosmos_policy.datasets.save_franka_t5_text_embeddings --data_dir "$DATA_DIR"

# 2) 训练（8 卡，grad_accum_iter=8，ckpt 保存到 cosmos_policy/ckpt）
IMAGINAIRE_OUTPUT_ROOT=/share/project/yunfan/cosmos-policy/cosmos_policy/ckpt/whiteboard \
uv run --extra cu128 --group libero --python 3.10 torchrun --nproc_per_node=8 --master_port=12341 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_franka_whiteboard" \
  trainer.grad_accum_iter=8

# 3) 推理前设置环境变量（在运行 franka_server 的终端里 export）
# export FRANKA_COSMOS_CKPT=/share/project/yunfan/cosmos-policy/cosmos_policy/ckpt/cosmos_policy_franka/cosmos_v2_finetune/cosmos_predict2_2b_480p_franka_whiteboard
# export FRANKA_DATASET_STATS_PATH="$DATA_DIR/dataset_statistics_franka.json"
# export FRANKA_T5_EMBEDDINGS_PATH="$DATA_DIR/t5_embeddings.pkl"
# uv run --extra cu128 --group libero --python 3.10 python -m cosmos_policy.experiments.robot.franka.franka_server
