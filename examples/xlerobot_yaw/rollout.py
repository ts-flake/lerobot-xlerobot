# !/usr/bin/env python
"""Autonomous policy rollout (evaluation) for XLeRobotYaw.

Thin wrapper around ``lerobot-rollout`` that injects two non-identity xlerobot
processors so the policy I/O matches how it was trained (see ``record.py``):

- ``robot_observation_processor`` — drops the observation DOFs the policy wasn't
  trained on.
- ``teleop_action_processor`` — a feature-select that shapes
  ``dataset_features[ACTION]`` (the action schema). ``context.py`` derives
  ``ordered_action_keys`` from that schema, so a feature-selected policy's joint
  output is mapped correctly. (The policy already outputs joint ``.pos`` targets,
  so there is **no inverse kinematics** on the policy path; ``robot_action_processor``
  stays at the identity default.)

Use ``--enable_*_control`` to declare which controls the policy was trained with
(defaults mirror ``record.py``: arms + base on, head off).

NOTE — base velocity is still dropped (a second-tier fix): ``lerobot-rollout``
filters both the policy observation and action to ``.pos`` only
(``context.py`` lines 279/284), so the base ``.vel`` action a default-recorded
policy emits is silently ignored here. Arm/head joints are mapped correctly;
deploying base control would require routing ``.vel`` into the policy-facing
obs/action upstream.

Example:

```shell
uv run examples/xlerobot_yaw/rollout.py \
    --strategy.type=base \
    --policy.path=<hf_user>/<my_xlerobot_policy> \
    --robot.type=xlerobot_yaw --robot.port1=/dev/ttyACM0 --robot.port2=/dev/ttyACM1 \
    --task="Grab the cube" --duration=60 --display_data=true
```
"""

import logging
from dataclasses import dataclass

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots.xlerobot_yaw import XLeRobotYaw  # registers --robot.type=xlerobot_yaw
from lerobot.rollout import RolloutConfig, build_rollout_context, create_strategy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun

from teleop_common import (
    build_action_feature_select,
    build_observation_feature_select,
    features_to_ignore,
)

logger = logging.getLogger(__name__)


@dataclass
class XLeRobotRolloutConfig(RolloutConfig):
    # Which controls the policy was trained with. Selects the observation and
    # action features so they match the recorded dataset (see record.py).
    # Defaults mirror the record config.
    enable_left_arm_control: bool = True
    enable_base_control: bool = True
    enable_head_control: bool = False


@parser.wrap()
def rollout(cfg: XLeRobotRolloutConfig):
    init_logging()

    if cfg.display_data:
        init_rerun(session_name="xlerobot_yaw_rollout", ip=cfg.display_ip, port=cfg.display_port)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    # Shape the policy-facing observation AND the dataset action schema to match how
    # the policy was trained (see record.py). We need the robot's feature names, so
    # construct an *unconnected* robot purely for metadata; build_rollout_context()
    # builds and connects the real one.
    robot_meta = XLeRobotYaw(cfg.robot)
    ignore = features_to_ignore(
        robot_meta,
        enable_left_arm_control=cfg.enable_left_arm_control,
        enable_base_control=cfg.enable_base_control,
        enable_head_control=cfg.enable_head_control,
    )
    logger.info("Feature-select drops %d feature(s): %s", len(ignore), ignore)
    # teleop_action_processor shapes dataset_features[ACTION] (the action schema);
    # context.py derives ordered_action_keys from it, so the policy's feature-selected
    # joint output is mapped correctly instead of being silently mis-labeled.
    teleop_action_processor = build_action_feature_select(ignore)
    robot_observation_processor = build_observation_feature_select(ignore)

    logger.info("Building rollout context...")
    ctx = build_rollout_context(
        cfg,
        shutdown_event,
        teleop_action_processor=teleop_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    strategy = create_strategy(cfg.strategy)
    logger.info("Rollout strategy: %s | policy: %s", cfg.strategy.type, cfg.policy.pretrained_path)

    try:
        strategy.setup(ctx)
        logger.info("Rollout setup complete, starting rollout...")
        strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        strategy.teardown(ctx)

    logger.info("Rollout finished")


def main():
    register_third_party_plugins()
    rollout()


if __name__ == "__main__":
    main()
