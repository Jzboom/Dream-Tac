"""Robot-side Xense RTC adapter; import in the robot's xense-client environment.

Compatible with Jzboom/xense-openpi 45054b5869f73503c133e5d765a3f9e06534266c.
Keeps its asynchronous queue/actual-consumption merge, but fixes the initial
latency history seed for Dream-Tac's delays (>10 action steps).
"""

import threading

import numpy as np

from xense_client.rtc_action_chunk_broker import RTCActionChunkBroker


class DreamTacRTCActionChunkBroker(RTCActionChunkBroker):
    def __init__(self, policy, *, frequency_hz=30.0, prefix=14, delay_margin=2, dry_run=False):
        if not isinstance(prefix, int) or isinstance(prefix, bool) or not 1 <= prefix < 40:
            raise ValueError('prefix must be an integer in [1, 39]')
        if not isinstance(delay_margin, int) or isinstance(delay_margin, bool) or not 0 <= delay_margin < prefix:
            raise ValueError('delay_margin must be an integer in [0, prefix)')
        if not 0 < frequency_hz < float('inf'):
            raise ValueError('frequency_hz must be finite and positive')
        if 2 * prefix - delay_margin > 40:
            raise ValueError('Chunk 40 cannot sustain this latency with the requested margin')
        self._dreamtac_prefix = prefix
        super().__init__(
            policy, frequency_hz=frequency_hz,
            action_queue_size_to_get_new_actions=prefix,
            rtc_enabled=True, execution_horizon=40,
            default_delay=prefix, delay_margin=delay_margin,
            blend_steps=0, delta_state_dim=0, dry_run=dry_run,
        )

    def warmup(self, obs):
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

    def reset(self):
        # Quiesce in-flight work before clearing an episode's queue. Otherwise
        # the previous episode's response can arrive after reset and refill it.
        self.stop()
        super().reset()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._get_actions_loop, daemon=True)
        self._thread_started = False

    def infer(self, obs):
        if not self._warmup_done:
            self.warmup(obs)
        return super().infer(obs)
