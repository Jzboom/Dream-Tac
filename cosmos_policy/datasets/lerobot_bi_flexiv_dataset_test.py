from __future__ import annotations

import av
import numpy as np

from cosmos_policy.datasets.lerobot_bi_flexiv_dataset import (
    LeRobotBiFlexivDataset,
    build_observation_relative_action_chunk,
)


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
