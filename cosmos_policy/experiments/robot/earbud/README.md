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
    "temporal_delta_actions": np.ndarray((20, 20), dtype=np.float32),
    "action_space": "absolute_tcp18_absolute_gripper2",
    "server_timing": {...},
}
```

The server-side absolute conversion integrates the TCP deltas from the request
state. For production control that integrates from the last action actually
sent by the robot, start the server with `--action-output temporal_delta` and
use a Dream-Tac-aware robot-side integrator.

OpenPI RTC is not supported. Use a synchronous `ActionChunkBroker` with
`action_horizon=20` for the first deployment.

## Start the server

```bash
export DREAMTAC_CKPT=/path/to/checkpoints/iter_XXXXXXXX
export DREAMTAC_WAN_VAE=/path/to/tokenizer/tokenizer.pth
export DREAMTAC_STATS=/path/to/dataset_statistics_lerobot_earbud.json
export DREAMTAC_T5=/path/to/t5_embeddings.pkl
export DREAMTAC_DEFAULT_PROMPT='the exact training task text'

python -m cosmos_policy.experiments.robot.earbud.earbud_server \
  --host 0.0.0.0 \
  --port 8000 \
  --num-denoising-steps 5
```

The server validates `state_t=18`, nine conditional slots, 69 tokenizer pixel
frames, statistics shapes, prompt cache membership, request shapes, and finite
action output before accepting robot traffic. By default it also performs one
fixed-shape warm-up inference before opening the port.
