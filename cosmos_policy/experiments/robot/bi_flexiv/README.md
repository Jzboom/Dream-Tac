# Dream-Tac bi_flexiv 11-slot inference

This server loads checkpoints trained with the fixed layout below.

| Slot | Content | Role |
|---:|---|---|
| 0 | blank | condition |
| 1 | current proprioception | condition |
| 2–4 | head, left-wrist, right-wrist RGB histories at `[-90,-60,-30,0]` | condition |
| 5–6 | merged left/right tactile histories at `[-3,-2,-1,0]` | condition |
| 7 | action chunk | prediction |
| 8–10 | future RGB sequences at `[+10,+20,+30,+40]` | prediction |

The WAN VAE input contains 41 pixel frames. Future proprioception and future
tactile are not predicted.

Checkpoints from the earlier single-frame, 30-step 11-slot layout are not
compatible; train this layout as a fresh job from the Cosmos Predict2 base.

Self-attention uses two bidirectional blocks: queries in slots 0–6 see only
slots 0–6, while queries in slots 7–10 see all slots. Action and future images
therefore remain jointly denoised without allowing prediction content to leak
into condition representations.

## Tactile preprocessing

The request supplies the four most recent frames for each raw tactile camera.
At each timestamp, sensor `_0` is stacked above `_1`; the resulting `800×700` image is resized with
`INTER_AREA` to `224×196`, then padded with 14 black pixels on both sides.
Training and inference call the same preprocessing function.

## Request

```python
{
    "observation_seq": 1,
    "state": np.ndarray((20,), dtype=np.float32),
    "images": {
        "head": np.ndarray((4, H, W, 3), dtype=np.uint8),
        "left_wrist": np.ndarray((4, H, W, 3), dtype=np.uint8),
        "right_wrist": np.ndarray((4, H, W, 3), dtype=np.uint8),
        "left_tactile_left": np.ndarray((4, 400, 700, 3), dtype=np.uint8),
        "left_tactile_right": np.ndarray((4, 400, 700, 3), dtype=np.uint8),
        "right_tactile_left": np.ndarray((4, 400, 700, 3), dtype=np.uint8),
        "right_tactile_right": np.ndarray((4, 400, 700, 3), dtype=np.uint8),
    },
    "tactile_self_attn_gate": np.ndarray((2,), dtype=np.float32),
    "prompt": "the exact prompt stored in t5_embeddings.pkl",
}
```

The RGB frames must be in chronological `[-90,-60,-30,0]` order, while every
tactile camera must use the recent chronological `[-3,-2,-1,0]` order. Legacy
numeric tactile keys (`left_tactile_0`, etc.) are also accepted, but legacy HWC
single frames are rejected. `tactile_self_attn_gate` keeps its instantaneous
`t` versus `t-1` meaning and is computed by the client.

The response contains a `(40, 20)` action chunk and timing information. When
future decoding is enabled, each of the three RGB outputs has shape
`(4,H,W,3)` in `[+10,+20,+30,+40]` order. The client-side `ActionChunkBroker`
must use `action_horizon=40`.

## Start server

```bash
export DREAMTAC_CKPT=/path/to/11_slot_checkpoint
export DREAMTAC_WAN_VAE=/path/to/tokenizer.pth
export DREAMTAC_STATS=/path/to/dataset_statistics_lerobot_bi_flexiv_chunk40.json
export DREAMTAC_T5=/path/to/t5_embeddings.pkl
export DREAMTAC_DEFAULT_PROMPT='the exact training prompt'

python -m cosmos_policy.experiments.robot.bi_flexiv.bi_flexiv_server \
  --host 0.0.0.0 \
  --port 8000
```

The defaults are 10 joint denoising calls and diffusion-step residual caching.
Only calls 1 and 3 run the full DiT block stack. Use
`--no-diffusion-step-cache` for an uncached comparison. Cross-request KV cache,
future-frame freezing, and runtime slot ablations are not included.

The server is stateless and validates `state_t=11`, the seven-slot block-causal
prefix, 41 tokenizer pixel frames, statistics shapes, prompt membership, and
the exact four-frame request shapes before use.
