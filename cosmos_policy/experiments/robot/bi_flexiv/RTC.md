# Dream-Tac RTC（推理侧）

按 **30 action/s（30 Hz）**、每个 chunk **40 actions** 计算，当前机器建议
**prefix=14、队列剩余 14 步时发起下一次推理、delay_margin=2**。
这里的 prefix 单位是 action，不是完整 chunk。若实际需求是每秒生成 30 个完整
chunk，则要求每次推理不超过 33.3 ms，当前实测不满足。

## 实测与计算

2026-09-21，RTX 5090，`0914_continous_condition_rgb/iter_000035000`，
episode 168 的 30 个不同观测，10 次去噪、开启 diffusion-step cache、
关闭未来图像解码，均先预热，不启动训练、不连接机器人：

| 路径 | 普通推理均值 | RTC 均值 | RTC P95 | RTC 最大值 | 测试 prefix |
|---|---:|---:|---:|---:|---:|
| 本地 policy 调用 | 301.9 ms | 303.5 ms | 307.0 ms | 317.2 ms | 17 |
| 本机 WebSocket / MsgPack，初次测量 | 355.5 ms | 365.7 ms | 373.9 ms | 377.3 ms | 14 |
| 本机 WebSocket / MsgPack，本次审查复测 | 361.1 ms | 361.4 ms | 370.9 ms | 379.6 ms | 14 |

此前包含未来 RGB 解码的视频评估均值约 468.9 ms，对应 prefix=17。
当前默认动作服务不解码未来图像，因此应使用上表的当前测量。
第一次 WebSocket 测量的 RTC 最大值为 392.7 ms，也对应 prefix=14。

```text
D = ceil(max_recent_client_RTT_seconds * 30) = 12
P = D + 2 = 14
chunk 时间 = 40 / 30 = 1.333 s
prefix 时间 = 14 / 30 = 0.467 s
新生成的后缀 = 40 - 14 = 26 actions
```

客户端以队列剩余 Q=14 为触发阈值。需要满足 `Q >= D` 且
`40 - D >= Q`，本次为 `28 >= 14`。若要保留完整 2 步余量，应令
`Q >= D + 2`。不要把过大的延迟简单截断为 39；若这些条件不满足，
需要降低推理延迟或控制频率。采样中的 prefix 始终不能超过实际剩余动作数。

**本机 WebSocket 不代表真实机器人网络 RTT。** 部署时用客户端
`rtc_metrics.infer_round_trip_ms` 和 `real_delay_steps` 更新估计；网络更慢时
同时增大初始 prefix 和队列触发阈值，并检查上述条件。当前 14 包含 2 步余量，
覆盖到 400 ms 的估计延迟；额外网络开销仍需在机器人侧实测。

原始结果（仓库根目录下）保存于：

- `artifacts/rtc/benchmark.json`
- `artifacts/rtc/websocket_benchmark.json`
- `artifacts/rtc/websocket_prefix14.json`
- `artifacts/rtc/review_websocket_prefix14.json`

所有真实模型 RTC 请求的返回 prefix 逐元素相等，最大误差 0；相同观测、
相同随机种子的后缀与普通推理不同，证明约束进入了采样过程。
这些测试验证推理和协议正确性，尚未验证实体机器人的任务成功率。

## 与参考实现的对应关系

参考 [Jzboom/xense-openpi](https://github.com/Jzboom/xense-openpi)，固定检查版本
`45054b5869f73503c133e5d765a3f9e06534266c`：

- [policy.py](https://github.com/Jzboom/xense-openpi/blob/45054b5869f73503c133e5d765a3f9e06534266c/src/openpi/policies/policy.py)：将旧绝对动作变换回当前观测下的模型动作空间。
- [pi0_pytorch.py](https://github.com/Jzboom/xense-openpi/blob/45054b5869f73503c133e5d765a3f9e06534266c/src/openpi/models_pytorch/pi0_pytorch.py)：每步去噪约束 prefix，结束后再次固定。
- [rtc_action_chunk_broker.py](https://github.com/Jzboom/xense-openpi/blob/45054b5869f73503c133e5d765a3f9e06534266c/packages/xense-client/src/xense_client/rtc_action_chunk_broker.py)：异步推理，以实际消费的动作数截去响应前部，滚动最大延迟加余量。

Dream-Tac 的整个动作 chunk 重复存放在一个 latent 槽中，使用槽级时间编码，
不能直接使用 π₀ 的逐动作零时间编码。本实现是**无需重训的 EDM prefix
inpainting**，不声称现有 checkpoint 已经过 π₀ training-time RTC 训练。
每次调用 denoiser 前，prefix 坐标设为 `normalized_prefix + sigma * epsilon`，
其中 epsilon 来自本次采样的固定初始噪声；x0 输出和最终结果中固定 clean
prefix。完整重复和末尾不足一个 chunk 的重复都会约束，其余动作和未来图像
继续联合去噪。没有修改模型参数、训练 loss 或 checkpoint 格式。

旧动作以绝对 TCP18 + 绝对夹爪2 传入，前18维减当前 state，再按训练统计归一化；
不裁剪旧 prefix。返回时精确恢复旧命令，避免量化、分位数裁剪或 SO(3) 再投影
改变已排队动作。常数统计维度以 normalized=0 条件化，输出仍精确保留原值。
RTC 仅支持默认 `absolute_from_state` 输出、单设备 EDM 采样。

## 服务端协议

原有启动命令即可，默认开启 RTC 能力，metadata 中 `rtc_supported=true`。
普通请求行为不变。客户端通过已有 `__rtc_kwargs__` 字段传递：

```python
response = policy.infer(
    observation,
    prev_chunk_left_over=remaining_absolute_actions,  # (remaining, 20)
    inference_delay=14,                              # action 数
    execution_horizon=40,                            # 兼容 Xense 的调度参数
)
```

首次调用传 `prev_chunk_left_over=None`，服务端按 prefix=0 处理。
响应始终是完整 `(40,20)` chunk，并增加 `rtc_prefix_length`。
客户端按**实际已消费步数**裁剪响应，而不是按估计 prefix 裁剪。
不接受负数、浮点 delay、全部冻结、短于 delay 的 prefix 或 NaN/Inf。
服务端不保存跨请求 prefix。

## 机器人客户端

在装有参考版本 `xense-client` / `lerobot` 的机器人 Python 环境中使用：

```python
from xense_client.websocket_client_policy import WebsocketClientPolicy
from cosmos_policy.experiments.robot.bi_flexiv.rtc_client import DreamTacRTCActionChunkBroker

remote = WebsocketClientPolicy(host="SERVER_IP", port=8000)
assert remote.get_server_metadata()["rtc_supported"]
broker = DreamTacRTCActionChunkBroker(
    remote, frequency_hz=30.0, prefix=14, delay_margin=2, dry_run=True,
)
broker.warmup(initial_observation)  # 开始执行前完成预热
try:
    # 在现有 30 Hz 控制循环内调用；每次得到一个 (20,) 绝对动作。
    result = broker.infer(current_observation)
    action = result["actions"]
finally:
    broker.stop()  # 整个控制循环结束时调用
```

将 Dream-Tac 仓库加入机器人端 `PYTHONPATH` 即可导入此轻量适配器；该模块不
导入模型、CUDA 或训练依赖。适配器复用参考版本的异步队列，并设置 horizon=40、
trigger=prefix、blend_steps=0。不要再对已是绝对值的返回动作加 state。

参考版本预热把首个 chunk 的末尾动作当作启动前缀，会跳过首段动作；且在
delay>10 时把历史估计设为4，导致第一次实际推理的 prefix 只有6。适配器先
同步预热普通与 RTC 两条路径，以首个 chunk 的**开头**作为前缀，再启动后台
调度；历史种子设为 `prefix-delay_margin`，之后使用原版的滚动最大实际延迟。
`reset()` 先等待旧请求结束再清空队列，防止上一轮响应污染下一轮。调用者应
在停止控制循环后调用 reset。适配器依赖上述版本的 broker 内部字段；升级
Xense 后需运行下面的客户端测试。

队列触发阈值固定为构造时的 prefix，滚动估计不会自动提高该阈值。若部署 RTT
超出预算，应停止并重新测量、调整 prefix/控制频率；不能依赖裁剪后的估计
自动适应任意延迟。

## 复现测试

推理环境（须先 `conda activate dreamtac`，加载 CUDA 动态库路径；仅调用环境
中的 Python 绝对路径不足以初始化 Transformer Engine）：

```bash
python -m pytest -q \
  cosmos_policy/modules/rtc_test.py \
  cosmos_policy/experiments/robot/bi_flexiv/rtc_policy_test.py \
  cosmos_policy/experiments/robot/bi_flexiv/bi_flexiv_policy_test.py \
  cosmos_policy/models/policy_text2world_model_test.py

python -m cosmos_policy.experiments.robot.bi_flexiv.rtc_benchmark \
  --checkpoint "$DREAMTAC_CKPT" --stats "$DREAMTAC_STATS" \
  --t5-embeddings "$DREAMTAC_T5" --wan-vae "$DREAMTAC_WAN_VAE" \
  --data-dir /path/to/test_tube_0729 --episode-index 168 \
  --samples 30 --prefix 14 --loopback \
  --output artifacts/rtc/websocket_prefix14.json
```

29 项推理侧测试覆盖重基准、绝对夹爪、归一化范围外 prefix、1/5/10 步采样、
重复 latent 尾部、零 prefix、请求间无状态和真实 WebSocket handler。
基准报告保存每次原始延迟、推荐 prefix、队列可持续性和实际配置。

在具有参考 `xense-client` 的机器人环境另行运行：

```bash
python -m pytest -q cosmos_policy/experiments/robot/bi_flexiv/rtc_client_test.py
```

该测试使用原版 Xense 队列，以 30 Hz 消费 100 个动作，模拟 360 ms 推理延迟，
检查首次和后续 prefix 都覆盖实际消费步数，并检查从第0步开始、reset 后重新
预热的动作顺序。无机器人连接；缺少客户端依赖的
推理环境会跳过此独立测试。
