#!/usr/bin/env python
"""Autonomous policy rollout (evaluation) for XLeRobotX.

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

Sim-trained policies (RoboTwin soarmX) with remote inference
-------------------------------------------------------------
``--sim_convention=true`` adds ``SimConventionObservation`` / ``SimConventionAction``
(deg <-> rad, gripper 0=open..100=closed <-> 1=open..0=closed, URDF clipping) around the
policy, and ``--prepare_pose=true`` first moves both arms to the RoboTwin start pose
(``homestate``, gripper open) and the head to ``--head_pose_deg``, because every training
episode starts from that exact state. Remote inference needs ``--robot.arms_only=true``
(position-hold contract, 14-D state/action). ``--dry_run=true`` runs inference and logs the
actions but keeps the arms at their measured pose.

```shell
# policy server (GPU laptop), see examples/remote_inference/server_smolvla_robotwin.yaml
lerobot-policy-server --config_path=server_smolvla_robotwin.yaml \
    --zenoh.listen_endpoints='[tcp/<laptop-ip>:7447]'

# robot computer (Jetson)
python lerobot_robot_xlerobot/examples/rollout.py \
    --strategy.type=base \
    --inference.type=remote --inference.endpoint=tcp/<laptop-ip>:7447 \
    --inference.deployment=xlerobotx-robotwin \
    --inference.semantics=xlerobotx-urdf-rad-gripper-open1-v1 \
    --inference.encoding=jpeg --inference.jpeg_quality=90 \
    --robot.type=xlerobot_x --robot.id=xlerobot_x --robot.arms_only=true \
    --robot.port1=/dev/feetech_left --robot.port2=/dev/feetech_right \
    --robot.cameras='{head: {type: opencv, index_or_path: /dev/..., width: 640, height: 480, fps: 30, fourcc: MJPG}, left_wrist: {...}, right_wrist: {...}}' \
    --rename_map='{"observation.images.head": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2", "observation.images.right_wrist": "observation.images.camera3"}' \
    --sim_convention=true --prepare_pose=true --dry_run=true \
    --fps=16.67 --task="<instruction>" --duration=60
```

Example (policy trained on real xlerobot data):

```shell
python lerobot_robot_xlerobot/examples/rollout.py \
    --strategy.type=base \
    --policy.path=<hf_user>/<my_xlerobot_policy> \
    --robot.type=xlerobot_x --robot.id=xlerobot_x --robot.port1=/dev/ttyACM0 --robot.port2=/dev/ttyACM1 \
    --task="Grab the cube" --duration=60 --display_data=true
```
"""

import logging
from dataclasses import dataclass, field

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot_robot_xlerobot import XLeRobotX  # registers --robot.type=xlerobot_x
from lerobot.rollout import RolloutConfig, build_rollout_context, create_strategy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun

from lerobot.inference import RemoteInferenceConfig
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot_robot_xlerobot.robot_action_observation_processor import RobotObservationFeatureSelect
from lerobot_robot_xlerobot.sim_convention_processor import (
    DryRunAction,
    SimConventionAction,
    SimConventionObservation,
    sim_home_real,
)
from teleop_common import (
    build_action_feature_select,
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

    # Policy I/O in the RoboTwin sim convention (rad, gripper 1=open); see module docstring.
    sim_convention: bool = False
    # Per-motor calibration correction in degrees (motor name -> deg), applied in both directions.
    joint_offset_deg: dict[str, float] = field(default_factory=dict)
    # Before the policy starts: move arms to the sim start pose and the head to `head_pose_deg`.
    prepare_pose: bool = False
    prepare_duration_s: float = 3.0
    head_pose_deg: dict[str, float] = field(default_factory=lambda: {"head_yaw": 0.0, "head_pitch": 35.0})
    # Run inference but hold the measured pose; would-be actions are logged.
    dry_run: bool = False


def build_robot_processors(cfg: XLeRobotRolloutConfig, ignore: list[str]):
    """Robot-side observation / action pipelines for the configured policy convention."""
    obs_steps = [RobotObservationFeatureSelect(ignore)]
    act_steps = []
    if cfg.sim_convention:
        obs_steps.append(SimConventionObservation(offset_deg=dict(cfg.joint_offset_deg)))
        act_steps.append(SimConventionAction(offset_deg=dict(cfg.joint_offset_deg)))
    if cfg.dry_run:
        act_steps.append(DryRunAction())
    robot_observation_processor = RobotProcessorPipeline(
        steps=obs_steps, to_transition=observation_to_transition, to_output=transition_to_observation
    )
    robot_action_processor = None  # identity default
    if act_steps:
        robot_action_processor = RobotProcessorPipeline(
            steps=act_steps,
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
    return robot_observation_processor, robot_action_processor


def prepare_start_pose(robot: XLeRobotX, cfg: XLeRobotRolloutConfig) -> None:
    """Move to the sim start pose (both arms, grippers open) and the configured head pose."""
    targets = sim_home_real(dict(cfg.joint_offset_deg))
    if not cfg.robot.arms_only and not cfg.enable_left_arm_control:
        targets = {k: v for k, v in targets.items() if not k.startswith("left_arm_")}
    targets.update(cfg.head_pose_deg)
    logger.info("Moving to start pose over %.1fs: %s", cfg.prepare_duration_s,
                {k: round(v, 1) for k, v in targets.items()})
    robot.move_to(targets, duration_s=cfg.prepare_duration_s)


@parser.wrap()
def rollout(cfg: XLeRobotRolloutConfig):
    init_logging()

    if cfg.display_data:
        init_rerun(session_name="xlerobot_x_rollout", ip=cfg.display_ip, port=cfg.display_port)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    # Shape the policy-facing observation AND the dataset action schema to match how
    # the policy was trained (see record.py). We need the robot's feature names, so
    # construct an *unconnected* robot purely for metadata; build_rollout_context()
    # builds and connects the real one.
    robot_meta = XLeRobotX(cfg.robot)
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
    robot_observation_processor, robot_action_processor = build_robot_processors(cfg, ignore)
    if isinstance(cfg.inference, RemoteInferenceConfig) and not cfg.robot.arms_only:
        raise ValueError("Remote inference needs --robot.arms_only=true (position-hold contract).")

    logger.info("Building rollout context...")
    ctx = build_rollout_context(
        cfg,
        shutdown_event,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    if cfg.prepare_pose:
        # After connect (the connect pose stays `initial_position`, so teardown returns there).
        prepare_start_pose(ctx.hardware.robot_wrapper.inner, cfg)

    strategy = create_strategy(cfg.strategy)
    logger.info("Rollout strategy: %s | policy: %s", cfg.strategy.type,
                getattr(cfg.policy, "pretrained_path", None) or f"remote:{getattr(cfg.inference, 'deployment', '')}")

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
