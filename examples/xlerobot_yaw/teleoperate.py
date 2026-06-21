# !/usr/bin/env python
"""Unified teleoperation entry point for XLeRobotYaw.

Pick the teleop device with ``--teleop.type``:

```shell
# VR controllers
uv run examples/xlerobot_yaw/teleoperate.py \
    --robot.type=xlerobot_yaw --robot.port1=/dev/ttyACM0 --robot.port2=/dev/ttyACM1 \
    --teleop.type=xlerobot_yaw_vr \
    --fps=30 --display_data=true
```

```shell
# PS5 gamepad
uv run examples/xlerobot_yaw/teleoperate.py \
    --robot.type=xlerobot_yaw --robot.port1=/dev/ttyACM0 --robot.port2=/dev/ttyACM1 \
    --teleop.type=xlerobot_yaw_gamepad \
    --fps=30 --display_data=true
```

The device-specific logic — mapping raw input to ee-deltas and the control
``events`` (exit / back-to-zero / ...) — lives in each teleoperator class
(``XLeRobotYawVR`` / ``XLeRobotYawGamepad``). This script only wires the shared
parts: the robot action processor (ee-delta -> safe joint command via IK) and
the teleop loop.
"""

import time
import logging
import traceback
from pathlib import Path

from lerobot.configs import parser
from lerobot.model.kinematics import RobotKinematics
from lerobot.model.rr_kinematics import RRKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
from lerobot.robots.xlerobot_yaw import XLeRobotYaw
from lerobot.robots.xlerobot_yaw.robot_action_processor import (
    InverseKinematicsDeltaToJoints,
    AnalyticalInverseKinematicsDeltaToJoints,
    BaseJointAction,
    JointClipNormValue,
    SafeGoalPosition,
    EMAJointAction,
    RoundOffAction,
    LogAction,
)
from lerobot.robots.xlerobot_yaw.utils.action_utils import move_robot_to_position, move_robot_to_zero_position
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.teleoperators.xlerobot_yaw_vr import XLeRobotYawVR, XLeRobotYawVRConfig
from lerobot.teleoperators.xlerobot_yaw_gamepad import XLeRobotYawGamepad, XLeRobotYawGamepadConfig
from lerobot.teleoperators.xlerobot_yaw_gamepad.gamepad_utils import PS5Gamepad
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.color_logger import init_color_logging

logger = logging.getLogger(__name__)


def make_teleop_device(config: TeleoperatorConfig) -> Teleoperator:
    """Instantiate the teleop device (the gamepad needs a concrete PS5Gamepad handle)."""
    if isinstance(config, XLeRobotYawVRConfig):
        return XLeRobotYawVR(config)
    if isinstance(config, XLeRobotYawGamepadConfig):
        return XLeRobotYawGamepad(config, PS5Gamepad(id=0))
    raise ValueError(f"Unsupported teleop config for xlerobot_yaw teleoperate: {type(config).__name__}")


def build_robot_action_processor(
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


@parser.wrap()
def main(cfg: TeleoperateConfig):
    # Initialize logging
    init_color_logging(cfg.teleop.console_level)

    FPS = cfg.fps
    ZERO_POSITION_OFFSET = cfg.teleop.zero_position_offset

    # Initialize the robot and teleoperator
    robot = XLeRobotYaw(cfg.robot)
    teleop_device = make_teleop_device(cfg.teleop)

    # Build pipeline to convert teleop ee-delta action to joint action
    robot_action_processor = build_robot_action_processor(robot, cfg.teleop, FPS)

    # Connect to the robot and teleoperator
    robot.connect()
    teleop_device.connect()

    # Init rerun viewer
    if cfg.display_data:
        init_rerun(session_name=f"xlerobot_yaw_teleop_{cfg.teleop.type}", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    if not robot.is_connected:
        logger.error("❌ Robot is not connected!")
        exit(1)

    if not wait_until_ready(teleop_device):
        logger.error("❌ Teleop device is not connected!")
        exit(1)

    # Record the initial position to restore on exit
    initial_obs = robot.get_observation()
    initial_pos = {k.removesuffix(".pos"): v for k, v in initial_obs.items() if k.endswith(".pos")}

    # Move to zero position
    move_robot_to_zero_position(robot, fps=FPS, duration=3.0, end_offset=ZERO_POSITION_OFFSET)

    # Main teleop loop
    logger.info("Starting teleop loop. Move your teleop device to teleoperate the robot...")
    try:
        while True:
            t0 = time.perf_counter()

            # Get robot observation
            robot_obs = robot.get_observation()

            # Optional: forward camera frames to the in-VR overlay.
            # No-op for devices that don't request it (e.g. gamepad).
            if getattr(teleop_device, "wants_observation_feedback", False):
                teleop_device.send_feedback(robot_obs)

            # Get teleop action
            teleop_obs = teleop_device.get_action()

            # Check for exit or reset (events are defined per-device in `teleop_device.events`)
            if teleop_device.events["exit_teleop"]:
                break
            if teleop_device.events["back_robot_to_zero"]:
                move_robot_to_zero_position(robot, fps=FPS, duration=3.0, end_offset=ZERO_POSITION_OFFSET)
                robot_action_processor.reset()
                teleop_device.events["back_robot_to_zero"] = False
                continue

            # teleop ee-delta -> joint pose -> joint transition
            joint_action = robot_action_processor((teleop_obs, robot_obs))

            # Send action to robot
            # The joint action to send is the actual action the robot takes, `ensure_safe_goal_position`
            # is applied in the action processor.
            _ = robot.send_action(joint_action)

            # Visualize
            if cfg.display_data:
                log_rerun_data(observation=robot_obs, action=joint_action, compress_images=display_compressed_images)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    except Exception as e:
        logger.error(f"❌ Error in teleop loop: {e}")
        traceback.print_exc()

    finally:
        # Clean up
        if initial_pos:
            move_robot_to_position(robot, end_pos=initial_pos, fps=FPS, duration=3.0)

        robot.disconnect()
        teleop_device.disconnect()


if __name__ == "__main__":
    main()
