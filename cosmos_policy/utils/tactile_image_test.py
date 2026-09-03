from __future__ import annotations

import numpy as np
import pytest

from cosmos_policy.utils.tactile_image import merge_tactile_pair_vertical


def test_merge_tactile_pair_vertical_uses_raw_pair_before_resize_and_padding() -> None:
    upper = np.full((400, 700, 3), 31, dtype=np.uint8)
    lower = np.full((400, 700, 3), 79, dtype=np.uint8)

    merged = merge_tactile_pair_vertical(upper, lower)

    assert merged.shape == (224, 224, 3)
    assert merged.flags.c_contiguous
    np.testing.assert_array_equal(merged[:, :14], 0)
    np.testing.assert_array_equal(merged[:, 210:], 0)
    np.testing.assert_array_equal(merged[40, 40], np.full(3, 31, dtype=np.uint8))
    np.testing.assert_array_equal(merged[180, 40], np.full(3, 79, dtype=np.uint8))


def test_merge_tactile_pair_vertical_rejects_mismatched_inputs() -> None:
    first = np.zeros((400, 700, 3), dtype=np.uint8)
    second = np.zeros((399, 700, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="identical shapes"):
        merge_tactile_pair_vertical(first, second)


def test_merge_tactile_pair_vertical_requires_uint8_rgb() -> None:
    valid = np.zeros((400, 700, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="dtype uint8"):
        merge_tactile_pair_vertical(valid.astype(np.float32), valid)
    with pytest.raises(ValueError, match="HWC RGB"):
        merge_tactile_pair_vertical(valid[..., 0], valid[..., 0])
