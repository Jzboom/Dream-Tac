# Dream-Tac earbud inference server

This directory provides the GPU-side inference endpoint for the 18-slot,
seven-camera LeRobot-Xense earbud policy. The WebSocket and MsgPack protocol is
compatible with `xense_client.WebsocketClientPolicy`.

## Request contract

```python
{
    "observation_seq": 1,  # optional, echoed in the response
    "state": np.ndarray((20,), dtype=np.float32),
    "images": {
        "head": np.ndarray((224, 224, 3), dtype=np.uint8),
        "left_wrist": np.ndarray((224, 224, 3), dtype=np.uint8),
        "right_wrist": np.ndarray((224, 224, 3), dtype=np.uint8),
        "left_tactile_0": np.ndarray((224, 224, 3), dtype=np.uint8),
        "left_tactile_1": np.ndarray((224, 224, 3), dtype=np.uint8),
        "right_tactile_0": np.ndarray((224, 224, 3), dtype=np.uint8),
        "right_tactile_1": np.ndarray((224, 224, 3), dtype=np.uint8),
    },
    "tactile_self_attn_gate": np.ndarray((2,), dtype=np.float32),
    "prompt": "the exact prompt stored in t5_embeddings.pkl",  # optional with server default
}
```

CHW images are also accepted. Images are directly resized to 224x224 and then
receive the deterministic 90%-area center crop used by the existing Cosmos
inference path. `images_raw` is not needed and should remain on the robot host.

The default response contains robot-ready absolute actions:

```python
{
    "actions": np.ndarray((20, 20), dtype=np.float32),
    "observation_relative_actions": np.ndarray((20, 20), dtype=np.float32),
    "action_space": "absolute_tcp18_absolute_gripper2",
    "server_timing": {...},
}
```

For a chunk starting at `t`, every TCP target is represented relative to the
same request state: `relative[k] = action[t+k] - observation.state[t]`. The
server converts it back with a direct addition, without cumulative summation.
Gripper targets remain absolute. Use `--action-output observation_relative` to
return this raw representation instead of robot-ready absolute targets.

OpenPI RTC is not supported. Use a synchronous `ActionChunkBroker` with
`action_horizon=20` for the first deployment.

## Start the server

```bash
export DREAMTAC_CKPT=/path/to/checkpoints/iter_XXXXXXXX
export DREAMTAC_WAN_VAE=/path/to/tokenizer/tokenizer.pth
export DREAMTAC_STATS=/path/to/dataset_statistics_lerobot_earbud.json
export DREAMTAC_T5=/path/to/t5_embeddings.pkl
export DREAMTAC_DEFAULT_PROMPT='the exact training task text'
export DREAMTAC_NORMALIZATION_MODE=q99

python -m cosmos_policy.experiments.robot.earbud.earbud_server \
  --host 0.0.0.0 \
  --port 8000 \
  --num-denoising-steps 5
```

New checkpoints use per-dimension q01/q99 normalization for observation-relative
action chunks and proprio, with clipping to `[-1, 1]`. The statistics JSON must
contain `action_chunk_size=20`, `gripper_start_idx=18`, `actions_q01`,
`actions_q99`, `proprio_q01`, and `proprio_q99`. The action statistics are
computed after every absolute chunk has been shifted by its starting
observation. Use
`DREAMTAC_NORMALIZATION_MODE=min_max` only for a checkpoint trained with min/max
statistics using the same action representation.

The server validates `state_t=18`, nine conditional slots, 69 tokenizer pixel
frames, statistics shapes, prompt cache membership, request shapes, and finite
action output before accepting robot traffic. By default it also performs one
fixed-shape warm-up inference before opening the port.
