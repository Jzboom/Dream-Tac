from __future__ import annotations

import av
import numpy as np

from cosmos_policy.datasets.lerobot_bi_flexiv_dataset import (
    LeRobotBiFlexivDataset,
    _group_nearby_frame_indices,
    build_observation_relative_action_chunk,
)
from cosmos_policy.utils.bi_flexiv_video_layout import (
    CONDITION_IMAGE_KEYS,
    FUTURE_IMAGE_OFFSETS,
    RGB_IMAGE_KEYS,
    RGB_HISTORY_OFFSETS,
    TACTILE_HISTORY_OFFSETS,
    build_pixel_frame_sequence,
    clamped_relative_indices,
)
from cosmos_policy.utils.tactile_self_attn_gate import scalar_gate_from_raw


def test_observation_relative_action_chunk_preserves_absolute_grippers_and_pads() -> None:
    raw_actions = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
    raw_proprio = np.full_like(raw_actions, 3.0)

    chunk = build_observation_relative_action_chunk(
        raw_actions,
        raw_proprio,
        relative_step_idx=4,
        chunk_size=4,
        gripper_start_idx=3,
    )

    expected = np.stack([raw_actions[4], raw_actions[5], raw_actions[5], raw_actions[5]])
    expected[:, :3] -= raw_proprio[4, :3]
    np.testing.assert_array_equal(chunk, expected)


def test_q99_normalization_clips_and_keeps_constant_dimensions() -> None:
    values = np.array([[-2.0, 4.0, 5.0], [2.0, 8.0, -5.0]], dtype=np.float32)
    stats = {
        "actions_q01": np.array([-1.0, 4.0, 0.0], dtype=np.float32),
        "actions_q99": np.array([1.0, 4.0, 10.0], dtype=np.float32),
    }

    normalized = LeRobotBiFlexivDataset._rescale_array(values, stats, "actions", normalization_mode="q99")

    np.testing.assert_array_equal(
        normalized,
        np.array([[-1.0, 1.0, 0.0], [1.0, 1.0, -1.0]], dtype=np.float32),
    )


def test_history_and_future_indices_clamp_to_episode_endpoints() -> None:
    assert clamped_relative_indices(20, RGB_HISTORY_OFFSETS, 100) == (0, 0, 0, 20)
    assert clamped_relative_indices(2, TACTILE_HISTORY_OFFSETS, 100) == (0, 0, 1, 2)
    assert clamped_relative_indices(20, TACTILE_HISTORY_OFFSETS, 100) == (17, 18, 19, 20)
    assert clamped_relative_indices(80, FUTURE_IMAGE_OFFSETS, 100) == (90, 99, 99, 99)


def test_tactile_gate_uses_the_two_most_recent_history_frames() -> None:
    history_frames = {}
    for key in LeRobotBiFlexivDataset.TACTILE_KEYS:
        history_frames[key] = np.stack(
            [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
            + [np.full((2, 2, 3), 255, dtype=np.uint8)]
        )

    left_gate, right_gate = LeRobotBiFlexivDataset._compute_per_arm_tactile_gate(
        object.__new__(LeRobotBiFlexivDataset),
        relative_step_idx=10,
        history_frames=history_frames,
    )

    assert left_gate == scalar_gate_from_raw(1.0)
    assert right_gate == scalar_gate_from_raw(1.0)


def test_pixel_sequence_preserves_history_and_future_order() -> None:
    history = {
        key: np.stack([np.full((2, 2, 3), base + index, dtype=np.uint8) for index in range(4)])
        for base, key in zip((10, 20, 30, 40, 50), CONDITION_IMAGE_KEYS, strict=True)
    }
    future = {
        key: np.stack([np.full((2, 2, 3), base + index, dtype=np.uint8) for index in range(4)])
        for base, key in zip((60, 70, 80), RGB_IMAGE_KEYS, strict=True)
    }

    frames = build_pixel_frame_sequence(history, future)

    assert frames.shape == (41, 2, 2, 3)
    np.testing.assert_array_equal(frames[:5], 0)
    np.testing.assert_array_equal(frames[5:9, 0, 0, 0], [10, 11, 12, 13])
    np.testing.assert_array_equal(frames[17:21, 0, 0, 0], [40, 41, 42, 43])
    np.testing.assert_array_equal(frames[25:29], 0)
    np.testing.assert_array_equal(frames[29:33, 0, 0, 0], [60, 61, 62, 63])
    np.testing.assert_array_equal(frames[37:41, 0, 0, 0], [80, 81, 82, 83])


def test_inference_future_placeholders_repeat_latest_rgb_history() -> None:
    history = {
        key: np.stack([np.full((1, 1, 3), base + index, dtype=np.uint8) for index in range(4)])
        for base, key in zip((10, 20, 30, 40, 50), CONDITION_IMAGE_KEYS, strict=True)
    }

    frames = build_pixel_frame_sequence(history)

    np.testing.assert_array_equal(frames[29:33, 0, 0, 0], [13, 13, 13, 13])
    np.testing.assert_array_equal(frames[33:37, 0, 0, 0], [23, 23, 23, 23])
    np.testing.assert_array_equal(frames[37:41, 0, 0, 0], [33, 33, 33, 33])


def test_nearby_frame_indices_are_grouped_without_merging_sparse_history() -> None:
    assert _group_nearby_frame_indices((10, 40, 70, 100, 110, 120, 130, 140)) == (
        (10,),
        (40,),
        (70,),
        (100, 110, 120, 130, 140),
    )
    assert _group_nearby_frame_indices((100, 97, 99, 98, 100)) == ((97, 98, 99, 100),)


def test_pyav_decoder_returns_the_exact_requested_frame_after_seek(tmp_path) -> None:
    path = tmp_path / "seek-test.mp4"
    output = av.open(str(path), mode="w")
    stream = output.add_stream("mpeg4", rate=10)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    for value in range(12):
        image = np.full((16, 16, 3), value * 10, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
    output.close()

    container = av.open(str(path), mode="r")
    harness = object.__new__(LeRobotBiFlexivDataset)
    harness.fps = 10
    decoded = harness._decode_frame(container, 7, str(path))
    container.close()

    assert decoded.shape == (16, 16, 3)
    assert decoded.dtype == np.uint8
    assert abs(float(decoded.mean()) - 70.0) < 5.0


def test_pyav_decoder_reads_multiple_targets_after_one_seek(tmp_path) -> None:
    path = tmp_path / "multi-frame-seek-test.mp4"
    output = av.open(str(path), mode="w")
    stream = output.add_stream("mpeg4", rate=10)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    for value in range(12):
        image = np.full((16, 16, 3), value * 10, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
    output.close()

    class _CountingContainer:
        def __init__(self, container):
            self._container = container
            self.streams = container.streams
            self.seek_count = 0

        def seek(self, *args, **kwargs):
            self.seek_count += 1
            return self._container.seek(*args, **kwargs)

        def decode(self, *args, **kwargs):
            return self._container.decode(*args, **kwargs)

    container = av.open(str(path), mode="r")
    counting_container = _CountingContainer(container)
    harness = object.__new__(LeRobotBiFlexivDataset)
    harness.fps = 10
    decoded = harness._decode_frames(counting_container, (3, 5, 8), str(path))
    container.close()

    assert counting_container.seek_count == 1
    assert tuple(decoded) == (3, 5, 8)
    for frame_idx, frame in decoded.items():
        assert abs(float(frame.mean()) - frame_idx * 10.0) < 5.0
