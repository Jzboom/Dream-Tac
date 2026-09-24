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


class ImmediatePolicy:
    def __init__(self, live=None):
        self.calls = 0
        self.live = live

    def infer(self, obs, **kwargs):
        self.calls += 1
        actions = np.repeat(np.arange(40, dtype=np.float32)[:, None], 20, axis=1)
        previous, delay = kwargs['prev_chunk_left_over'], kwargs['inference_delay']
        if previous is not None:
            actions[:delay] = previous[:delay]
        if self.calls > 2 and self.live is not None:
            actions = self.live(actions, obs, kwargs)
        return {'actions': actions}

    def reset(self):
        pass


def consume_to_trigger(broker):
    for _ in range(26):
        broker.infer({'state': np.zeros(20)})


def test_snapshot_and_consumption_are_atomic():
    snapshot_entered, release_snapshot = threading.Event(), threading.Event()
    request_entered, release_response = threading.Event(), threading.Event()
    control_done = threading.Event()

    def live(actions, obs, kwargs):
        request_entered.set()
        assert release_response.wait(3)
        return actions

    broker = DreamTacRTCActionChunkBroker(ImmediatePolicy(live))
    broker.warmup({})
    queue = broker._action_queue
    original_snapshot = queue.get_left_over

    def paused_snapshot(*args, **kwargs):
        snapshot_entered.set()
        assert release_snapshot.wait(3)
        return original_snapshot(*args, **kwargs)

    queue.get_left_over = paused_snapshot
    consumer = None
    try:
        consume_to_trigger(broker)
        assert snapshot_entered.wait(3)
        consumed = []
        def consume():
            consumed.append(broker.infer({'state': np.zeros(20)})['actions'][0])
            control_done.set()
        consumer = threading.Thread(target=consume)
        consumer.start()
        assert not control_done.wait(0.05), 'control consumed inside the queue snapshot'
        release_snapshot.set()
        assert request_entered.wait(3) and control_done.wait(3)
        assert consumed == [26]
        merged = threading.Event()
        original_merge = queue.merge
        def merge(**kwargs):
            original_merge(**kwargs)
            merged.set()
        queue.merge = merge
        release_response.set()
        assert merged.wait(3)
        # The old split snapshot discarded action 27 here and returned 28.
        assert broker.infer({})['actions'][0] == 27
    finally:
        release_snapshot.set()
        release_response.set()
        if consumer:
            consumer.join(3)
        broker.stop()


@pytest.mark.parametrize('bad', ['nan', 'inf', 'shape', 'prefix'])
def test_bad_live_response_fails_control_without_merging(bad):
    def live(actions, obs, kwargs):
        if bad == 'shape':
            return actions[:-1]
        actions[0 if bad == 'prefix' else 30, 0] = {'nan': np.nan, 'inf': np.inf, 'prefix': 999}[bad]
        return actions
    broker = DreamTacRTCActionChunkBroker(ImmediatePolicy(live))
    try:
        broker.warmup({})
        original_queue = broker._action_queue.queue.copy()
        consume_to_trigger(broker)
        assert broker._stop_event.wait(3)
        np.testing.assert_array_equal(broker._action_queue.queue, original_queue)
        with pytest.raises(RuntimeError, match='RTC inference failed') as error:
            broker.infer({})
        assert isinstance(error.value.__cause__, ValueError)
    finally:
        broker.stop()


def test_stalled_request_cannot_reset_or_refill_another_episode():
    entered, release = threading.Event(), threading.Event()
    def live(actions, obs, kwargs):
        entered.set()
        assert release.wait(3)
        return actions
    policy = ImmediatePolicy(live)
    broker = DreamTacRTCActionChunkBroker(policy, stop_timeout=0.05)
    try:
        broker.warmup({})
        consume_to_trigger(broker)
        assert entered.wait(3)
        old_queue = broker._action_queue.queue.copy()
        started = time.monotonic()
        with pytest.raises(TimeoutError, match='reset refused'):
            broker.reset()
        assert time.monotonic() - started < 0.5
        np.testing.assert_array_equal(broker._action_queue.queue, old_queue)
        assert broker._warmup_done
        release.set()
        broker._thread.join(3)
        np.testing.assert_array_equal(broker._action_queue.queue, old_queue)
        policy.live = None
        broker.reset()
        assert broker.infer({})['actions'][0] == 0
    finally:
        release.set()
        broker._thread.join(3)
        broker.stop()


def test_queue_exhaustion_fails_immediately_instead_of_blocking_control():
    entered, release = threading.Event(), threading.Event()
    def live(actions, obs, kwargs):
        entered.set()
        assert release.wait(3)
        return actions
    broker = DreamTacRTCActionChunkBroker(ImmediatePolicy(live))
    try:
        broker.warmup({})
        consume_to_trigger(broker)
        assert entered.wait(3)
        for _ in range(14):
            broker.infer({})
        started = time.monotonic()
        with pytest.raises(RuntimeError, match='RTC inference failed'):
            broker.infer({})
        assert time.monotonic() - started < 0.2
    finally:
        release.set()
        broker.stop()


def test_observation_is_owned_while_live_request_is_pending():
    entered, release = threading.Event(), threading.Event()
    observed = []
    def live(actions, obs, kwargs):
        entered.set()
        assert release.wait(3)
        observed.append(obs['state'].copy())
        return actions
    broker = DreamTacRTCActionChunkBroker(ImmediatePolicy(live))
    try:
        broker.warmup({})
        obs = {'state': np.ones(20)}
        for _ in range(26):
            broker.infer(obs)
        assert entered.wait(3)
        obs['state'][:] = 999
        release.set()
        broker.stop()
        np.testing.assert_array_equal(observed[0], np.ones(20))
    finally:
        release.set()
        broker.stop()


def test_websocket_timeout_discards_stream_and_reset_reconnects():
    from websockets.sync.server import serve
    from xense_client import msgpack_numpy
    from cosmos_policy.experiments.robot.bi_flexiv.rtc_client import DreamTacWebsocketClientPolicy

    release = threading.Event()
    received = []
    def handler(ws):
        ws.send(msgpack_numpy.packb({'rtc_supported': True}))
        request = msgpack_numpy.unpackb(ws.recv())
        received.append(request)
        if len(received) == 1:
            release.wait(3)
        else:
            ws.send(msgpack_numpy.packb({'actions': np.zeros((40, 20), np.float32)}))
    with serve(handler, '127.0.0.1', 0, close_timeout=0.1) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        remote = DreamTacWebsocketClientPolicy('127.0.0.1', server.socket.getsockname()[1], request_timeout=0.1)
        obs = {'state': np.zeros(20)}
        try:
            with pytest.raises(TimeoutError):
                remote.infer(obs, inference_delay=0)
            assert '__rtc_kwargs__' not in obs
            with pytest.raises(RuntimeError, match='connection failed'):
                remote.infer(obs)
            release.set()
            remote.reset()
            assert remote.infer(obs)['actions'].shape == (40, 20)
        finally:
            release.set()
            remote.cancel_pending()
            server.shutdown()
            thread.join(3)


def test_response_exceeding_frozen_prefix_is_not_merged():
    entered, release = threading.Event(), threading.Event()
    def live(actions, obs, kwargs):
        assert kwargs['inference_delay'] == 3
        entered.set()
        assert release.wait(3)
        return actions
    broker = DreamTacRTCActionChunkBroker(ImmediatePolicy(live))
    try:
        broker.warmup({})
        broker._recent_real_delays.clear()
        broker._recent_real_delays.append(1)
        consume_to_trigger(broker)
        assert entered.wait(3)
        old_queue = broker._action_queue.queue.copy()
        for _ in range(4):
            broker.infer({})
        release.set()
        assert broker._stop_event.wait(3)
        with pytest.raises(RuntimeError) as error:
            broker.infer({})
        assert 'prefix budget' in str(error.value.__cause__)
        np.testing.assert_array_equal(broker._action_queue.queue, old_queue)
    finally:
        release.set()
        broker.stop()


def test_stop_cancels_real_websocket_request_before_reset():
    from websockets.sync.server import serve
    from websockets.exceptions import ConnectionClosed
    from xense_client import msgpack_numpy
    from cosmos_policy.experiments.robot.bi_flexiv.rtc_client import DreamTacWebsocketClientPolicy

    entered = threading.Event()
    def handler(ws):
        ws.send(msgpack_numpy.packb({'rtc_supported': True}))
        try:
            for call in range(3):
                request = msgpack_numpy.unpackb(ws.recv())
                if call == 2:
                    entered.set()
                    ws.recv()  # Wait for client cancellation; do not respond.
                    return
                a = np.repeat(np.arange(40, dtype=np.float32)[:, None], 20, axis=1)
                kw = request['__rtc_kwargs__']
                if kw['prev_chunk_left_over'] is not None:
                    a[:kw['inference_delay']] = kw['prev_chunk_left_over'][:kw['inference_delay']]
                ws.send(msgpack_numpy.packb({'actions': a}))
        except ConnectionClosed:
            pass
    with serve(handler, '127.0.0.1', 0, close_timeout=0.1) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        remote = DreamTacWebsocketClientPolicy('127.0.0.1', server.socket.getsockname()[1], request_timeout=5)
        broker = DreamTacRTCActionChunkBroker(remote, stop_timeout=0.05)
        try:
            broker.warmup({})
            consume_to_trigger(broker)
            assert entered.wait(3)
            started = time.monotonic()
            broker.reset()
            assert time.monotonic() - started < 2
            assert not broker._thread_started
            assert broker._action_queue.qsize() == 0
            assert broker.infer({})['actions'][0] == 0
        finally:
            broker.stop()
            remote.cancel_pending()
            server.shutdown()
            thread.join(3)
