"""xense-client-compatible WebSocket server for Dream-Tac earbud inference.

Example:

.. code-block:: bash

   python -m cosmos_policy.experiments.robot.earbud.earbud_server \
     --checkpoint /data/dreamtac/checkpoints/iter_000003000 \
     --stats /data/dreamtac/dataset_statistics_lerobot_earbud.json \
     --t5-embeddings /data/dreamtac/t5_embeddings.pkl \
     --wan-vae /data/dreamtac/tokenizer/tokenizer.pth \
     --default-prompt "insert the earbuds into the charging case"
"""

from __future__ import annotations

import argparse
import asyncio
import http
import logging
import os
import time
import traceback

import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

from cosmos_policy.experiments.robot.earbud import msgpack_numpy
from cosmos_policy.experiments.robot.earbud.earbud_policy import DreamTacEarbudPolicy, DreamTacEarbudPolicyConfig

logger = logging.getLogger(__name__)


class DreamTacWebsocketServer:
    def __init__(self, policy: DreamTacEarbudPolicy, *, host: str, port: int) -> None:
        self.policy = policy
        self.host = host
        self.port = port

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.policy.metadata))

        previous_total_ms: float | None = None
        while True:
            try:
                request_start = time.perf_counter()
                message = await websocket.recv()
                if isinstance(message, str):
                    raise ValueError("Inference requests must be binary MsgPack frames")
                observation = msgpack_numpy.unpackb(message)
                if not isinstance(observation, dict):
                    raise ValueError(f"Observation must be a dictionary, got {type(observation).__name__}")
                rtc_kwargs = observation.pop("__rtc_kwargs__", {})

                infer_start = time.perf_counter()
                response = self.policy.infer(observation, **rtc_kwargs)
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
                timing = dict(response.get("server_timing", {}))
                timing["infer_ms"] = infer_ms
                if previous_total_ms is not None:
                    timing["prev_total_ms"] = previous_total_ms
                response["server_timing"] = timing
                await websocket.send(packer.pack(response))
                previous_total_ms = (time.perf_counter() - request_start) * 1000.0
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                error_message = traceback.format_exc()
                logger.error("Inference request failed:\n%s", error_message)
                await websocket.send(error_message)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Dream-Tac inference failed; traceback was sent in the previous frame",
                )
                break

    async def run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info("Dream-Tac inference server ready at ws://%s:%d", self.host, self.port)
            await server.serve_forever()


def _health_check(
    connection: websocket_server.ServerConnection,
    request: websocket_server.Request,
) -> websocket_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("DREAMTAC_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DREAMTAC_PORT", "8000")))
    parser.add_argument(
        "--config",
        default=os.environ.get("DREAMTAC_CONFIG", "cosmos_predict2_2b_480p_lerobot_earbud_tactile"),
    )
    parser.add_argument(
        "--config-file", default=os.environ.get("DREAMTAC_CONFIG_FILE", "cosmos_policy/config/config.py")
    )
    parser.add_argument("--checkpoint", default=os.environ.get("DREAMTAC_CKPT", ""))
    parser.add_argument("--wan-vae", default=os.environ.get("DREAMTAC_WAN_VAE", ""))
    parser.add_argument("--stats", default=os.environ.get("DREAMTAC_STATS", ""))
    parser.add_argument("--t5-embeddings", default=os.environ.get("DREAMTAC_T5", ""))
    parser.add_argument("--default-prompt", default=os.environ.get("DREAMTAC_DEFAULT_PROMPT", ""))
    parser.add_argument(
        "--num-denoising-steps",
        type=int,
        default=int(os.environ.get("DREAMTAC_NUM_DENOISING_STEPS", "5")),
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("DREAMTAC_SEED", "0")))
    parser.add_argument(
        "--action-output",
        choices=("absolute_from_state", "observation_relative"),
        default=os.environ.get("DREAMTAC_ACTION_OUTPUT", "absolute_from_state"),
        help=(
            "absolute_from_state is compatible with the existing xense-openpi ActionChunkBroker; "
            "observation_relative returns the raw chunk relative to the request state"
        ),
    )
    parser.add_argument(
        "--normalization-mode",
        choices=("q99", "min_max"),
        default=os.environ.get("DREAMTAC_NORMALIZATION_MODE", "q99"),
        help="Must match the normalization used to train the checkpoint.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=None)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-normalized-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-prompt-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _require(value: str, parser: argparse.ArgumentParser, flag: str) -> str:
    if not value:
        parser.error(f"{flag} is required (or set the corresponding DREAMTAC_* environment variable)")
    return value


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = DreamTacEarbudPolicyConfig(
        checkpoint_path=_require(args.checkpoint, parser, "--checkpoint"),
        dataset_stats_path=_require(args.stats, parser, "--stats"),
        t5_embeddings_path=_require(args.t5_embeddings, parser, "--t5-embeddings"),
        default_prompt=_require(args.default_prompt, parser, "--default-prompt"),
        config_name=args.config,
        config_file=args.config_file,
        wan_vae_path=args.wan_vae or None,
        num_denoising_steps=args.num_denoising_steps,
        seed=args.seed,
        center_crop=args.center_crop,
        jpeg_quality=args.jpeg_quality,
        clip_normalized_actions=args.clip_normalized_actions,
        action_output=args.action_output,
        normalization_mode=args.normalization_mode,
        allow_prompt_fallback=args.allow_prompt_fallback,
    )

    logger.info("Loading Dream-Tac policy from %s", config.checkpoint_path)
    policy = DreamTacEarbudPolicy(config)
    logger.info("Policy metadata: %s", policy.metadata)
    if args.warmup:
        logger.info("Running fixed-shape warm-up inference")
        warmup_result = policy.warmup()
        logger.info(
            "Warm-up complete: actions=%s timing=%s",
            warmup_result["actions"].shape,
            warmup_result.get("server_timing"),
        )

    asyncio.run(DreamTacWebsocketServer(policy, host=args.host, port=args.port).run())


if __name__ == "__main__":
    main()
