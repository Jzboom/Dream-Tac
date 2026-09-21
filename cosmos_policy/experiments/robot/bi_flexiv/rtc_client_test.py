import json
import time
import threading
import numpy as np
import pytest

pytest.importorskip("xense_client.rtc_action_chunk_broker")
from cosmos_policy.experiments.robot.bi_flexiv.rtc_client import DreamTacRTCActionChunkBroker

class Policy:
    def __init__(self):
        self.calls = []
    def infer(self, obs, **kwargs):
        self.calls.append(kwargs['inference_delay'])
        actions = np.arange(800, dtype=np.float32).reshape(40, 20)
        prev = kwargs['prev_chunk_left_over']
        delay = kwargs['inference_delay']
        if prev is not None:
            actions[:delay] = prev[:delay]
        time.sleep(0.36)
        return {'actions': actions}
    def reset(self):
        pass

def test_xense_broker_covers_delay_at_30hz_after_warmup():
    policy = Policy()
    broker = DreamTacRTCActionChunkBroker(policy, dry_run=True)
    try:
        broker.warmup({'state': np.zeros(20)})
        assert list(broker._recent_real_delays) == [12]
        metrics = []
        start = time.perf_counter()
        for i in range(100):
            result = broker.infer({'state': np.zeros(20)})
            assert result['actions'].shape == (20,)
            if 'rtc_metrics' in result:
                metrics.append(result['rtc_metrics'])
            time.sleep(max(0, start + (i + 1) / 30 - time.perf_counter()))
        assert len(policy.calls) >= 5
        assert policy.calls[2] == 14, policy.calls
        live = [m for m in metrics if m['inference_seq'] > 1]
        assert live and all(m['real_delay_steps'] <= m['estimated_delay_steps'] for m in live)
        print(json.dumps({'actions_consumed':100,'frequency_hz':30,'simulated_inference_seconds':0.36,
                          'requested_prefixes':policy.calls,'minimum_queue_after_merge':min(m['queue_size_after_merge'] for m in live),
                          'all_real_delays_covered':True}))
    finally:
        broker.stop()


def test_warmup_starts_at_first_action_and_reset_rewarms():
    policy = Policy()
    broker = DreamTacRTCActionChunkBroker(policy, dry_run=True)
    try:
        for episode in range(2):
            broker.warmup({'state': np.zeros(20)})
            # Original Xense warmup incorrectly starts with action 26 (520..539).
            for index in range(3):
                result = broker.infer({'state': np.zeros(20)})
                np.testing.assert_array_equal(result['actions'], np.arange(index * 20, (index + 1) * 20))
            if episode == 0:
                broker.reset()
        assert policy.calls == [0, 14, 0, 14]
    finally:
        broker.stop()


def test_reset_waits_for_inflight_response_before_clearing_queue():
    entered = threading.Event()
    release = threading.Event()
    reset_done = threading.Event()

    class SlowPolicy(Policy):
        def infer(self, obs, **kwargs):
            if len(self.calls) == 2:
                entered.set()
                assert release.wait(5)
            return super().infer(obs, **kwargs)

    broker = DreamTacRTCActionChunkBroker(SlowPolicy(), dry_run=True)
    reset_thread = None
    try:
        broker.warmup({'state': np.zeros(20)})
        for _ in range(26):
            broker.infer({'state': np.zeros(20)})
        assert entered.wait(5)

        def reset():
            broker.reset()
            reset_done.set()

        reset_thread = threading.Thread(target=reset)
        reset_thread.start()
        assert not reset_done.wait(0.05)
        release.set()
        assert reset_done.wait(5)
        assert broker._action_queue.qsize() == 0
        assert not broker._warmup_done
        broker.warmup({'state': np.zeros(20)})
        np.testing.assert_array_equal(broker.infer({'state': np.zeros(20)})['actions'], np.arange(20))
    finally:
        release.set()
        if reset_thread is not None:
            reset_thread.join(timeout=5)
        broker.stop()
