"""Robot-side Xense RTC adapter; import in the robot's xense-client environment.

Compatible with Jzboom/xense-openpi 45054b5869f73503c133e5d765a3f9e06534266c.
Uses its queue/actual-consumption merge with atomic snapshots, validated live
responses, and bounded shutdown. The transport supports cancellation and
reconnection after failed requests.
"""

import copy
import math
import threading
import time

import numpy as np
from websockets.sync.client import connect

from xense_client import msgpack_numpy
from xense_client.rtc_action_chunk_broker import RTCActionChunkBroker
from xense_client.websocket_client_policy import WebsocketClientPolicy


class DreamTacWebsocketClientPolicy(WebsocketClientPolicy):
    """Bound connection/response waits and discard timed-out protocol streams."""

    def __init__(self, host="0.0.0.0", port=None, api_key=None, *, request_timeout=120.0):
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be finite and positive")
        self.request_timeout = request_timeout
        self._failed = False
        super().__init__(host=host, port=port, api_key=api_key)

    def _wait_for_server(self):
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        ws = connect(
            self._uri, compression=None, max_size=None, additional_headers=headers,
            open_timeout=self.request_timeout, close_timeout=1.0,
        )
        try:
            return ws, msgpack_numpy.unpackb(ws.recv(timeout=self.request_timeout))
        except Exception:
            ws.close()
            raise

    def infer(self, obs, **kwargs):
        if self._failed:
            raise RuntimeError("RTC connection failed; reset before sending another request")
        payload = dict(obs)
        payload.pop("__rtc_kwargs__", None)
        if kwargs:
            payload["__rtc_kwargs__"] = kwargs
        try:
            self._ws.send(self._packer.pack(payload))
            response = self._ws.recv(timeout=self.request_timeout)
            if isinstance(response, str):
                raise RuntimeError(f"Error in inference server: {response}")
            return msgpack_numpy.unpackb(response)
        except Exception:
            self._failed = True
            self._ws.close()
            raise

    def reset(self):
        if self._failed:
            self._ws, self._server_metadata = self._wait_for_server()
            self._failed = False

    def cancel_pending(self):
        self._failed = True
        self._ws.close()


class DreamTacRTCActionChunkBroker(RTCActionChunkBroker):
    def __init__(
        self, policy, *, frequency_hz=30.0, prefix=14, delay_margin=2, dry_run=False, stop_timeout=2.0,
    ):
        if not isinstance(prefix, int) or isinstance(prefix, bool) or not 1 <= prefix < 40:
            raise ValueError('prefix must be an integer in [1, 39]')
        if not isinstance(delay_margin, int) or isinstance(delay_margin, bool) or not 0 <= delay_margin < prefix:
            raise ValueError('delay_margin must be an integer in [0, prefix)')
        if not 0 < frequency_hz < float('inf'):
            raise ValueError('frequency_hz must be finite and positive')
        if 2 * prefix - delay_margin > 40:
            raise ValueError('Chunk 40 cannot sustain this latency with the requested margin')
        if not math.isfinite(stop_timeout) or stop_timeout <= 0:
            raise ValueError("stop_timeout must be finite and positive")
        self._stop_timeout = stop_timeout
        self._queue_transaction = threading.RLock()
        self._worker_error = None
        self._dreamtac_prefix = prefix
        super().__init__(
            policy, frequency_hz=frequency_hz,
            action_queue_size_to_get_new_actions=prefix,
            rtc_enabled=True, execution_horizon=40,
            default_delay=prefix, delay_margin=delay_margin,
            blend_steps=0, delta_state_dim=0, dry_run=dry_run,
        )

    def warmup(self, obs):
        self._check_running()
        if self._warmup_done:
            return
        # No action has been executed yet. Upstream takes the LAST prefix
        # of a discarded warmup chunk, which skips the beginning of the plan.
        # Warm both model paths synchronously, carrying the FIRST actions.
        # The worker is started only by infer(), after this queue is ready.
        first = self._policy.infer(
            obs, prev_chunk_left_over=None, inference_delay=0, execution_horizon=40,
        )
        initial_actions = self._validated_actions(first)
        prefix = initial_actions[:self._dreamtac_prefix].copy()
        second = self._policy.infer(
            obs, prev_chunk_left_over=prefix,
            inference_delay=self._dreamtac_prefix, execution_horizon=40,
        )
        actions = self._validated_actions(second)
        if not np.array_equal(actions[:self._dreamtac_prefix], prefix):
            raise ValueError("RTC server changed the warmup action prefix")
        self._action_queue.merge(
            new_original_actions=actions, new_processed_actions=actions,
            estimated_delay=0,
            action_index_before_inference=self._action_queue.get_action_index(),
        )
        self._recent_real_delays.clear()
        self._recent_real_delays.append(self._dreamtac_prefix - self._delay_margin)
        self._warmup_done = True
        self._first_inference_done.set()

    @staticmethod
    def _validated_actions(result):
        actions = np.asarray(result['actions'], dtype=np.float32)
        if actions.shape != (40, 20) or not np.isfinite(actions).all():
            raise ValueError("RTC server must return finite (40, 20) absolute actions")
        return actions.copy()

    def _check_running(self):
        if self._worker_error is not None:
            raise RuntimeError("RTC inference failed; stop control and reset the broker") from self._worker_error
        if self._stop_event.is_set():
            raise RuntimeError("RTC broker is stopped; reset before reuse")

    def _get_actions_loop(self):
        # All queue reads, consumption and merges share one transaction lock.
        # Release it during the remote request so the control loop keeps moving.
        while not self._stop_event.is_set():
            try:
                with self._queue_transaction:
                    if self._stop_event.is_set():
                        return
                    if self._latest_obs is None or self._action_queue.qsize() > self._dreamtac_prefix:
                        request = None
                    else:
                        index = self._action_queue.get_action_index()
                        remaining = self._action_queue.get_left_over()
                        if remaining is None or len(remaining) == 0:
                            raise RuntimeError("RTC action queue exhausted")
                        remaining = remaining.copy()
                        delay = min(max(self._recent_real_delays) + self._delay_margin, len(remaining))
                        request = (self._latest_obs, index, remaining, delay)
                if request is None:
                    self._stop_event.wait(0.001)
                    continue
                obs, index, remaining, delay = request
                started = time.perf_counter()
                result = self._policy.infer(
                    obs, prev_chunk_left_over=remaining.copy(), inference_delay=delay, execution_horizon=40,
                )
                latency = time.perf_counter() - started
                actions = self._validated_actions(result)
                if not np.array_equal(actions[:delay], remaining[:delay]):
                    raise ValueError("RTC server changed the queued action prefix")
                with self._queue_transaction:
                    if self._stop_event.is_set():
                        return  # Never merge an old response after stop/reset.
                    consumed = self._action_queue.get_action_index() - index
                    if consumed > delay:
                        raise RuntimeError("RTC inference exceeded its frozen prefix budget")
                    merge_start = time.perf_counter()
                    self._action_queue.merge(
                        new_original_actions=actions, new_processed_actions=actions,
                        estimated_delay=delay, action_index_before_inference=index,
                    )
                    self._last_real_delay = consumed
                    self._recent_real_delays.append(consumed)
                    self._latency_tracker.add(latency)
                    self._inference_seq += 1
                    if self._dry_run:
                        self._last_inference_metrics = {
                            "inference_seq": self._inference_seq,
                            "infer_round_trip_ms": latency * 1000,
                            "inference_delay_steps": math.ceil(latency * self._frequency_hz),
                            "estimated_delay_steps": delay,
                            "real_delay_steps": consumed,
                            "merge_ms": (time.perf_counter() - merge_start) * 1000,
                            "queue_size_after_merge": self._action_queue.qsize(),
                            "latency_p95_ms": (self._latency_tracker.p95() or 0) * 1000,
                            "recent_max_real_delay": max(self._recent_real_delays),
                            "delay_margin": self._delay_margin,
                            "delay_for_next_infer_steps": max(self._recent_real_delays) + self._delay_margin,
                        }
            except Exception as exc:
                with self._queue_transaction:
                    if not self._stop_event.is_set():
                        self._worker_error = exc
                    self._stop_event.set()
                return

    def stop(self):
        with self._queue_transaction:
            self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._stop_timeout)
            if self._thread.is_alive():
                # Our transport can interrupt recv. Generic policies may not
                # support cancellation; refuse reset while they are in flight.
                cancel = getattr(self._policy, "cancel_pending", None)
                if cancel is not None:
                    cancel()
                    self._thread.join(timeout=self._stop_timeout)
                if self._thread.is_alive():
                    raise TimeoutError("RTC request is still running; queue preserved, reset refused")

    def reset(self):
        # Caller must stop the control loop before resetting an episode.
        self.stop()
        with self._queue_transaction:
            super().reset()
            self._worker_error = None
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._get_actions_loop, daemon=True)
            self._thread_started = False

    def infer(self, obs):
        self._check_running()
        if not self._warmup_done:
            self.warmup(obs)
        # Own the observation, including arrays: camera/control buffers may be
        # reused by the caller while the worker serializes its request.
        observation = copy.deepcopy(obs)
        with self._queue_transaction:
            self._check_running()
            self._latest_obs = observation
            # The upstream ActionQueue.get() error path uses logger.warning,
            # which is absent on this robot's spdlog logger. Check under the
            # same transaction lock before entering that path.
            if self._action_queue.qsize() == 0:
                self._worker_error = RuntimeError("RTC action queue exhausted; inference latency exceeded budget")
                self._stop_event.set()
                self._check_running()
            action = self._action_queue.get()
            if action is None:
                self._worker_error = RuntimeError("RTC action queue exhausted; inference latency exceeded budget")
                self._stop_event.set()
                self._check_running()
            self._start_thread_if_needed()
            result = {"actions": action}
            if self._dry_run and self._last_inference_metrics is not None:
                result["rtc_metrics"] = dict(self._last_inference_metrics)
            return result
