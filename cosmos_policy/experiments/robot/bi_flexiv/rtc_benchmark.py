"""Measure warmed inference and RTC prefix preservation on a real offline episode.

No training or robot connection is used. JSON records raw timings and a 30 Hz
prefix recommendation; network RTT must be measured separately on deployment.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import threading
import time
from pathlib import Path

import numpy as np

from cosmos_policy.datasets.lerobot_bi_flexiv_dataset import LeRobotBiFlexivDataset
from cosmos_policy.experiments.robot.bi_flexiv.bi_flexiv_policy import (
    CHUNK_SIZE, DreamTacBiFlexivPolicy, DreamTacBiFlexivPolicyConfig,
)
from cosmos_policy.modules.rtc import prefix_from_latency


@contextlib.contextmanager
def loopback_client(policy):
    """Exercise the production binary WebSocket handler on an ephemeral port."""
    from websockets.asyncio.server import serve
    from websockets.sync.client import connect
    from cosmos_policy.experiments.robot.bi_flexiv import msgpack_numpy
    from cosmos_policy.experiments.robot.bi_flexiv.bi_flexiv_server import DreamTacWebsocketServer

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    async def start():
        handler = DreamTacWebsocketServer(policy, host='127.0.0.1', port=0)
        return await serve(handler._handler, '127.0.0.1', 0, compression=None, max_size=None)
    server = None
    try:
        server = asyncio.run_coroutine_threadsafe(start(), loop).result(timeout=10)
        port = server.sockets[0].getsockname()[1]
        with connect(f'ws://127.0.0.1:{port}', compression=None, max_size=None) as websocket:
            metadata = msgpack_numpy.unpackb(websocket.recv())
            assert metadata['rtc_supported']
            packer = msgpack_numpy.Packer()
            def infer(obs, **kwargs):
                websocket.send(packer.pack(dict(obs, __rtc_kwargs__=kwargs)))
                result = websocket.recv()
                if isinstance(result, str):
                    raise RuntimeError(result)
                return msgpack_numpy.unpackb(result)
            yield infer
    finally:
        if server is not None:
            async def close():
                server.close()
                await server.wait_closed()
            asyncio.run_coroutine_threadsafe(close(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'stats', 't5-embeddings', 'wan-vae', 'data-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--episode-index', type=int, default=168)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--prefix', type=int, default=14)
    parser.add_argument('--frequency-hz', type=float, default=30)
    parser.add_argument('--output', default='artifacts/rtc/benchmark.json')
    parser.add_argument('--diffusion-step-cache', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--loopback', action='store_true', help='Measure binary WebSocket RTT on localhost')
    args = parser.parse_args()
    if args.samples < 2 or not 0 < args.prefix < CHUNK_SIZE:
        parser.error('samples must be >= 2 and prefix in [1, 39]')
    config = DreamTacBiFlexivPolicyConfig(
        checkpoint_path=args.checkpoint, dataset_stats_path=args.stats,
        t5_embeddings_path=args.t5_embeddings, default_prompt='', wan_vae_path=args.wan_vae,
        diffusion_step_cache=args.diffusion_step_cache,
    )
    dataset = LeRobotBiFlexivDataset(
        data_dir=args.data_dir, chunk_size=CHUNK_SIZE, final_image_size=224,
        t5_text_embeddings_path='', normalize_images=False, normalize_actions=False,
        normalize_proprio=False, use_image_aug=False, use_stronger_image_aug=False,
    )
    try:
        observations = [dataset.get_inference_sample(args.episode_index, i * 10)['observation']
                        for i in range(args.samples + 3)]
        policy = DreamTacBiFlexivPolicy(config)
        warmup = policy.infer(observations[0])
        prefix = warmup['actions'][-args.prefix:].copy()
        for _ in range(2):
            policy.infer(observations[1], prev_chunk_left_over=prefix, inference_delay=args.prefix)
        records = []
        max_error = 0.0
        suffix_differences = []
        transport = loopback_client(policy) if args.loopback else contextlib.nullcontext(policy.infer)
        with transport as infer:
            # Warm the serving thread and transport too, not only the caller's
            # CUDA context. Startup calls must not enter the control loop.
            if args.loopback:
                infer(observations[1])
                infer(observations[2], prev_chunk_left_over=prefix, inference_delay=args.prefix)
            for i, obs in enumerate(observations[3:]):
                baseline = None
                for mode in ('baseline', 'rtc'):
                    start = time.perf_counter()
                    response = infer(obs, **({
                        'prev_chunk_left_over': prefix, 'inference_delay': args.prefix,
                        'execution_horizon': CHUNK_SIZE,
                    } if mode == 'rtc' else {}))
                    elapsed = time.perf_counter() - start
                    assert response['actions'].shape == (40, 20)
                    assert np.isfinite(response['actions']).all()
                    if mode == 'rtc':
                        np.testing.assert_array_equal(response['actions'][:args.prefix], prefix)
                        max_error = max(max_error, float(np.abs(response['actions'][:args.prefix] - prefix).max()))
                        suffix_differences.append(float(np.abs(response['actions'][args.prefix:] - baseline[args.prefix:]).max()))
                        prefix = response['actions'][-args.prefix:].copy()
                    else:
                        baseline = response['actions']
                    records.append(dict(mode=mode, seconds=elapsed, timing=response['server_timing']))
                    print(f'{i + 1}/{args.samples} {mode}: {elapsed * 1000:.1f} ms', flush=True)
        summary = {}
        for mode in ('baseline', 'rtc'):
            values = [r['seconds'] for r in records if r['mode'] == mode]
            summary[mode] = dict(mean_seconds=float(np.mean(values)),
                                 p95_seconds=float(np.percentile(values, 95)),
                                 max_seconds=max(values))
        delay = prefix_from_latency(summary['rtc']['max_seconds'], args.frequency_hz, margin=0)
        recommended = delay + 2
        report = dict(config=dataclasses.asdict(config), episode=args.episode_index,
                      frequency_hz=args.frequency_hz, tested_prefix=args.prefix,
                      recommended_prefix=recommended, queue_trigger=recommended,
                      feasible=CHUNK_SIZE - delay >= recommended,
                      max_prefix_error=max_error, warmup_timing=warmup['server_timing'],
                      suffix_max_differences=suffix_differences,
                      transport='localhost_websocket' if args.loopback else 'local_policy',
                      summary=summary, records=records,
                      note='Recompute using deployed client RTT; no robot quality test. Cold start excluded.')
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ('config', 'records')}, indent=2))
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
