# Dream-Tac bi_flexiv inference server

This directory provides the GPU-side inference endpoint for the 18-slot,
seven-camera LeRobot bi_flexiv policy. The WebSocket and MsgPack protocol is
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
export DREAMTAC_CKPT=/home/p3/data_sda1/checkpoints_haonan/0814_256_batch_200_episode/checkpoints/iter_000020000
export DREAMTAC_WAN_VAE=/home/p3/data_sda1/checkpoints_haonan/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth
export DREAMTAC_STATS=/home/p3/data_sda1/checkpoints_haonan/0814_256_batch_200_episode/checkpoints/dataset_statistics_lerobot_bi_flexiv.json
export DREAMTAC_T5=/home/p3/data_sda1/checkpoints_haonan/datasets/test_tube_0729_0808_160_temp/t5_embeddings.pkl
export DREAMTAC_DEFAULT_PROMPT='Invert the test tube, pick up the pipette, mount the tip to pipette, aspirate from beaker and dispense into the tube, eject the tip, return the pipette, and cap with the stopper.'

python -m cosmos_policy.experiments.robot.bi_flexiv.bi_flexiv_server \
  --host 0.0.0.0 \
  --port 8000 \
  --num-denoising-steps 5
```

The statistics contents are unchanged by the platform rename. For an existing
checkpoint, `DREAMTAC_STATS` may continue to point to a file named
`dataset_statistics_lerobot_earbud.json`; newly generated files use the
`dataset_statistics_lerobot_bi_flexiv.json` name.

## Future-image prediction vs GT

Offline evaluation can decode all seven predicted future slots (head, two
wrists, and four tactile views) with the WAN VAE. The online WebSocket server
does not decode or save future images, so its inference latency is unchanged.

Before offline decoding, only the non-image slots 0/1/9/10 are restored from the
clean placeholder latent sequence so that proprio/action injections do not
create temporal VAE artifacts. Predicted future slots 11 through 17 are never
restored or replaced; they remain the model predictions being evaluated.

For offline evaluation on a LeRobot episode:

```bash
python -m cosmos_policy.experiments.robot.bi_flexiv.offline_future_image_eval \
  --data-dir /path/to/lerobot_dataset \
  --episode-index 0 \
  --start 0 \
  --stride 20 \
  --num-denoising-steps 5 \
  --output-dir ./bi_flexiv_future_comparisons
```

The script uses the same environment variables as the server. For every start
at `t`, it compares the decoded prediction with the dataset frame at `t+20`.
Starts whose target would be clamped to the last episode frame are skipped by
default; pass `--include-padded-future` to include them.

`--num-denoising-steps` directly controls the diffusion sampler. Its precedence
is command-line value, then `DREAMTAC_NUM_DENOISING_STEPS`, then fallback `5`.
The selected value and seed are printed at startup and included in each output
filename as `_stepsNNN_seedN`.

The only saved files are side-by-side PNG panels under `comparisons/`, with GT
on the left and Dream-Tac prediction on the right. Separate prediction/GT PNGs
and metric JSON files are not written.

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
