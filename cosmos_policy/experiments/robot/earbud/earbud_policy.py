"""Backward-compatible imports for the renamed bi_flexiv policy."""

from cosmos_policy.experiments.robot.bi_flexiv.bi_flexiv_policy import *  # noqa: F403
from cosmos_policy.experiments.robot.bi_flexiv.bi_flexiv_policy import (
    DreamTacBiFlexivPolicy,
    DreamTacBiFlexivPolicyConfig,
)

DreamTacEarbudPolicy = DreamTacBiFlexivPolicy
DreamTacEarbudPolicyConfig = DreamTacBiFlexivPolicyConfig
