# Cosmos Policy 各 Loss 分量说明

训练时 `output_batch` 中与 loss 相关的标量如下（`iter_speed` callback 会按 `logging_iter` 打印这些分量）。

## 1. 整体 latent loss（未做 mask）

| Key | 含义 |
|-----|------|
| `mse_loss` | 所有帧的 latent MSE 均值：`(x0 - pred)^2` 的 mean |
| `edm_loss` | EDM 加权后的 loss 均值：MSE × sigma 权重，与反向传播用的 kendall_loss 同尺度 |

## 2. Demo 样本上的分量（仅 demonstration，rollout_data_mask=0）

**Action（策略主 loss）**

| Key | 含义 |
|-----|------|
| `demo_sample_action_mse_loss` | Demo 上 action 帧预测的 MSE |
| `demo_sample_action_l1_loss` | Demo 上 action 帧预测的 L1 |

**Future state（未来状态辅助 loss）**

| Key | 含义 |
|-----|------|
| `demo_sample_future_proprio_mse_loss` | Demo 上 future proprio 的 MSE |
| `demo_sample_future_proprio_l1_loss` | Demo 上 future proprio 的 L1 |
| `demo_sample_future_wrist_image_mse_loss` | Demo 上 future wrist 图像的 MSE |
| `demo_sample_future_wrist_image_l1_loss` | Demo 上 future wrist 图像的 L1 |
| `demo_sample_future_image_mse_loss` | Demo 上 future 主视角图像的 MSE |
| `demo_sample_future_image_l1_loss` | Demo 上 future 主视角图像的 L1 |

**Value（无 value 目标时为 nan）**

| Key | 含义 |
|-----|------|
| `demo_sample_value_mse_loss` | Demo 上 value 预测的 MSE（Franka 无 value 时为 nan） |
| `demo_sample_value_l1_loss` | Demo 上 value 预测的 L1 |

## 3. Rollout world-model 样本（仅 rollout 且 world_model_sample_mask=1）

| Key | 含义 |
|-----|------|
| `world_model_sample_future_proprio_mse_loss` | 未来 proprio MSE |
| `world_model_sample_future_proprio_l1_loss` | 未来 proprio L1 |
| `world_model_sample_future_wrist_image_mse_loss` | 未来 wrist 图像 MSE |
| `world_model_sample_future_wrist_image_l1_loss` | 未来 wrist 图像 L1 |
| `world_model_sample_future_image_mse_loss` | 未来主视角图像 MSE |
| `world_model_sample_future_image_l1_loss` | 未来主视角图像 L1 |
| `world_model_sample_value_mse_loss` | value MSE |
| `world_model_sample_value_l1_loss` | value L1 |

## 4. Rollout value-function 样本（仅 rollout 且 value_function_sample_mask=1）

| Key | 含义 |
|-----|------|
| `value_function_sample_value_mse_loss` | value 预测 MSE |
| `value_function_sample_value_l1_loss` | value 预测 L1 |

---

**说明**

- 实际参与反向传播的是 **kendall_loss**（= `edm_loss` 按 mask 后的 mean × `loss_scale`），mask 由 `mask_value_prediction_loss_for_policy_prediction` 等配置决定。
- Franka 实验通常只有 demo，且 `return_value_function_returns=False`，因此 `demo_sample_action_*` 和 `demo_sample_future_*` 有值，`*value*` 相关项多为 nan。
