"""Backward-compatible imports for the former earbud-specific module name.

New code should import :mod:`cosmos_policy.datasets.lerobot_bi_flexiv_dataset`.
"""

from cosmos_policy.datasets.lerobot_bi_flexiv_dataset import *  # noqa: F403
from cosmos_policy.datasets.lerobot_bi_flexiv_dataset import LeRobotBiFlexivDataset

LeRobotEarbudDataset = LeRobotBiFlexivDataset
