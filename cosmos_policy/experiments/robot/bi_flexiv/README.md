# Dream-Tac bi_flexiv 11-slot inference

This server loads checkpoints trained with the fixed layout below.

| Slot | Content | Role |
|---:|---|---|
| 0 | blank | condition |
| 1 | current proprioception | condition |
| 2–4 | head, left-wrist, right-wrist RGB | condition |
| 5–6 | merged left/right tactile | condition |
| 7 | action chunk | prediction |
| 8–10 | future head/left-wrist/right-wrist RGB | prediction |

The WAN VAE input contains 41 pixel frames. Future proprioception and future
tactile are not predicted.

## Tactile preprocessing

The request still supplies four raw tactile images. For each gripper, sensor
`_0` is stacked above `_1`; the resulting `800×700` image is resized with
`INTER_AREA` to `224×196`, then padded with 14 black pixels on both sides.
Training and inference call the same preprocessing function.

## Request

```python
{
    "observation_seq": 1,
    "state": np.ndarray((20,), dtype=np.float32),
    "images": {
        "head": np.ndarray((H, W, 3), dtype=np.uint8),
        "left_wrist": np.ndarray((H, W, 3), dtype=np.uint8),
        "right_wrist": np.ndarray((H, W, 3), dtype=np.uint8),
        "left_tactile_left": np.ndarray((400, 700, 3), dtype=np.uint8),
        "left_tactile_right": np.ndarray((400, 700, 3), dtype=np.uint8),
        "right_tactile_left": np.ndarray((400, 700, 3), dtype=np.uint8),
        "right_tactile_right": np.ndarray((400, 700, 3), dtype=np.uint8),
    },
    "tactile_self_attn_gate": np.ndarray((2,), dtype=np.float32),
    "prompt": "the exact prompt stored in t5_embeddings.pkl",
}
```

Legacy numeric tactile keys (`left_tactile_0`, etc.) are also accepted. The
response contains a `(30, 20)` action chunk and timing information. The three
future RGB outputs represent the single `t+30` frame, approximately one second
after the current observation for 30 Hz data. The client-side
`ActionChunkBroker` must also use `action_horizon=30`.

## Start server

```bash
export DREAMTAC_CKPT=/path/to/11_slot_checkpoint
export DREAMTAC_WAN_VAE=/path/to/tokenizer.pth
export DREAMTAC_STATS=/path/to/dataset_statistics_lerobot_bi_flexiv.json
export DREAMTAC_T5=/path/to/t5_embeddings.pkl
export DREAMTAC_DEFAULT_PROMPT='the exact training prompt'

python -m cosmos_policy.experiments.robot.bi_flexiv.bi_flexiv_server \
  --host 0.0.0.0 \
  --port 8000
```

The defaults are 10 joint denoising calls and diffusion-step residual caching.
Only calls 1 and 3 run the full DiT block stack. Use
`--no-diffusion-step-cache` for an uncached comparison. KV cache, asymmetric
attention, future-frame freezing, and runtime slot ablations are not included.

The server validates `state_t=11`, seven conditional slots, 41 tokenizer pixel
frames, statistics shapes, prompt membership, and request shapes before use.
