# !/usr/bin/env python
"""Shared helpers for the XLeRobotYaw teleoperate / record entry points.

Both ``teleoperate.py`` and ``record.py`` select a teleop device with
``--teleop.type`` and drive it through the same ee-delta -> joint command
pipeline; the device-agnostic glue lives here.

Importing this module also registers the VR and gamepad teleop configs with
draccus, so ``--teleop.type=xlerobot_yaw_vr|xlerobot_yaw_gamepad`` resolves.
"""

import time
import logging
from pathlib import Path

from lerobot.model.kinematics import RobotKinematics
from lerobot.model.rr_kinematics import RRKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
    observation_to_transition,
    transition_to_observation,
)
from lerobot.robots.xlerobot_yaw import XLeRobotYaw
from lerobot.robots.xlerobot_yaw.robot_kinematics_processor import (
    InverseKinematicsDeltaToJoints,
    AnalyticalInverseKinematicsDeltaToJoints,
    BaseJointAction,
    JointClipNormValue,
    SafeGoalPosition,
    EMAJointAction,
    RoundOffAction,
    LogAction,
)
from lerobot.robots.xlerobot_yaw.robot_action_observation_processor import (
    RobotActionFeatureSelect,
    RobotObservationFeatureSelect,
)
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.teleoperators.xlerobot_yaw_vr import XLeRobotYawVR, XLeRobotYawVRConfig
from lerobot.teleoperators.xlerobot_yaw_gamepad import XLeRobotYawGamepad, XLeRobotYawGamepadConfig
from lerobot.teleoperators.xlerobot_yaw_gamepad.gamepad_utils import PS5Gamepad
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


def make_teleop_device(config: TeleoperatorConfig) -> Teleoperator:
    """Instantiate the teleop device (the gamepad needs a concrete PS5Gamepad handle)."""
    if isinstance(config, XLeRobotYawVRConfig):
        return XLeRobotYawVR(config)
    if isinstance(config, XLeRobotYawGamepadConfig):
        return XLeRobotYawGamepad(config, PS5Gamepad(id=0))
    raise ValueError(f"Unsupported teleop config for xlerobot_yaw: {type(config).__name__}")


def build_ee_delta_to_joints_processor(
    robot: XLeRobotYaw,
    teleop_config: TeleoperatorConfig,
    fps: int,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    """ee-delta teleop action -> safe robot joint command (shared by all devices)."""
    if teleop_config.use_placo_ik:
        logger.info("Using Placo IK")
        urdf_path = str(Path(__file__).resolve().parent / "simulation/so101_yaw/so101_yaw.urdf")
        ik_step = InverseKinematicsDeltaToJoints(
            kinematics_left=RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="gripper_back",
                joint_names=[f"joint{i}" for i in range(1, 7)],  # exclude the gripper joint
            ),
            kinematics_right=RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="gripper_back",
                joint_names=[f"joint{i}" for i in range(1, 7)],  # exclude the gripper joint
            ),
            motor_names=robot.left_arm_motors + robot.right_arm_motors + robot.head_motors,
        )
    else:
        logger.info("Using analytical IK")
        ik_step = AnalyticalInverseKinematicsDeltaToJoints(
            kinematics_left=RRKinematics(use_degrees=True, offsets=[90, 90], reversed=[True, False]),
            kinematics_right=RRKinematics(use_degrees=True, offsets=[90, 90], reversed=[True, False]),
            motor_names=robot.left_arm_motors + robot.right_arm_motors + robot.head_motors,
        )

    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            ik_step,
            BaseJointAction(robot=robot),
            JointClipNormValue(robot=robot),
            SafeGoalPosition(
                max_relative_target=robot.config.max_relative_target,
                motor_names=robot.left_arm_motors + robot.right_arm_motors + robot.head_motors,
            ),
            EMAJointAction(fps=fps, ema_alpha=0.9),
            RoundOffAction(),
            LogAction(logger),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def wait_until_ready(teleop_device: Teleoperator, timeout_s: float = 15.0) -> bool:
    """Block until the teleop device is connected and calibrated.

    VR connects asynchronously (a background server thread), so we poll until it
    comes up; the gamepad connects synchronously, so the first check already
    reflects its real state and a missing controller fails fast after the timeout.
    """
    start = time.perf_counter()
    while not teleop_device.is_connected:
        if time.perf_counter() - start > timeout_s:
            return False
        precise_sleep(0.1)
    # Some devices (e.g. VR) require calibration; others report `None` (no-op).
    while teleop_device.is_calibrated is False:
        precise_sleep(0.1)
    return True


def features_to_ignore(
    robot: XLeRobotYaw,
    *,
    enable_left_arm_control: bool = True,
    enable_base_control: bool = True,
    enable_head_control: bool = False,
) -> list[str]:
    """Action/observation feature names to drop for the controls that are disabled.

    Shared by record (to shape the recorded dataset) and rollout (to shape the
    policy's observation and action schema so they match that dataset). The same
    list is applied to both the action and observation feature-selects.
    """
    ignore: list[str] = []
    if not enable_left_arm_control:
        ignore += [feat for feat in robot.action_features if "left_arm_" in feat]
    if not enable_base_control:
        ignore += ["x.vel", "y.vel", "theta.vel"]
    if not enable_head_control:
        ignore += [feat for feat in robot.action_features if "head_" in feat]
    return ignore


def build_action_feature_select(
    features_to_ignore: list[str],
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    """Pipeline that drops the disabled controls from the robot action."""
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[RobotActionFeatureSelect(features_to_ignore)],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def build_observation_feature_select(
    features_to_ignore: list[str],
) -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    """Pipeline that drops the disabled controls from the robot observation."""
    return RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[RobotObservationFeatureSelect(features_to_ignore)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
