"""Backward-compatible entry point for the renamed bi_flexiv server."""

from cosmos_policy.experiments.robot.bi_flexiv.bi_flexiv_server import *  # noqa: F403
from cosmos_policy.experiments.robot.bi_flexiv.bi_flexiv_server import main


if __name__ == "__main__":
    main()
