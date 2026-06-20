# !/usr/bin/env python

import time
import logging
import draccus
from dataclasses import dataclass, field
import traceback
from pathlib import Path

from lerobot.model.kinematics import RobotKinematics
from lerobot.model.rr_kinematics import RRKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.xlerobot_yaw import XLeRobotYaw, XLeRobotYawConfig
from lerobot.robots.xlerobot_yaw.robot_action_processor import (
    InverseKinematicsDeltaToJoints,
    AnalyticalInverseKinematicsDeltaToJoints,
    BaseJointAction,
    JointClipNormValue,
    SafeGoalPosition,
    EMAJointAction,
    RoundOffAction,
    LogAction
)
from lerobot.robots.xlerobot_yaw.utils.action_utils import move_robot_to_position, move_robot_to_zero_position
from lerobot.teleoperators.xlerobot_yaw_vr import XLeRobotYawVR, XLeRobotYawVRConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from lerobot.utils.color_logger import init_color_logging

@dataclass
class Config:
    robot: XLeRobotYawConfig
    teleop: XLeRobotYawVRConfig
    display_data: bool = True

@draccus.wrap()
def main(cfg: Config):
    # Initialize logging
    init_color_logging(cfg.teleop.console_level)
    logger = logging.getLogger(__name__)

    # Initialize the robot and teleoperator
    robot_config = cfg.robot
    teleop_config = cfg.teleop

    FPS = teleop_config.fps
    ZERO_POSITION_OFFSET = teleop_config.zero_position_offset

    # Initialize the robot and teleoperator
    robot = XLeRobotYaw(robot_config)
    teleop_device = XLeRobotYawVR(teleop_config)

    # Build pipeline to convert vr action to joint action
    if teleop_config.use_placo_ik:
        logger.info("Using Placo IK")
        urdf_path = str(Path(__file__).resolve().parent / "simulation/so101_yaw/so101_yaw.urdf")
        ik_step = InverseKinematicsDeltaToJoints(
            kinematics_left=RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="gripper_back",
                joint_names=[f"joint{i}" for i in range(1, 7)] # exclude the gripper joint
            ),
            kinematics_right=RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="gripper_back",
                joint_names=[f"joint{i}" for i in range(1, 7)] # exclude the gripper joint
            ),
            motor_names=robot.left_arm_motors + robot.right_arm_motors + robot.head_motors
        )
    else:
        logger.info("Using analytical IK")
        ik_step = AnalyticalInverseKinematicsDeltaToJoints(
            kinematics_left=RRKinematics(use_degrees=True, offsets=[90, 90], reversed=[True, False]),
            kinematics_right=RRKinematics(use_degrees=True, offsets=[90, 90], reversed=[True, False]),
            motor_names=robot.left_arm_motors + robot.right_arm_motors + robot.head_motors,
        )

    vr_to_robot_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            ik_step,
            BaseJointAction(robot=robot),
            JointClipNormValue(robot=robot),
            SafeGoalPosition(
                max_relative_target=robot.config.max_relative_target,
                motor_names=robot.left_arm_motors + robot.right_arm_motors + robot.head_motors
            ),
            EMAJointAction(fps=FPS, ema_alpha=0.9),
            RoundOffAction(),
            LogAction(logger),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect to the robot and teleoperator
    robot.connect()
    teleop_device.connect()

    # Init rerun viewer
    if cfg.display_data:
        init_rerun(session_name="xlerobot_yaw_teleop_vr")

    if not robot.is_connected:
        logger.error("❌ Robot is not connected!")
        exit(1)

    while not teleop_device.is_connected:
        precise_sleep(0.1) # Wait for VR to connect

    while not teleop_device.is_calibrated:
        precise_sleep(0.1) # Wait for VR to calibrate

    # Record the initial position
    initial_obs = robot.get_observation()
    initial_pos = {k.removesuffix(".pos"): v for k, v in initial_obs.items() if k.endswith(".pos")}

    # Move to zero position
    move_robot_to_zero_position(robot, fps=FPS, duration=3.0, end_offset=ZERO_POSITION_OFFSET)

    # Main teleop loop
    logger.info("Starting teleop loop. Move your VR controllers to teleoperate the robot...")
    try:
        while True:
            t0 = time.perf_counter()

            # Get robot observation
            robot_obs = robot.get_observation()

            # Optional: forward camera frames to the in-VR overlay.
            # OFF by default (config.stream_cameras_to_vr); no-op otherwise.
            if teleop_device.wants_observation_feedback:
                teleop_device.send_feedback(robot_obs)

            # Get teleop action
            vr_obs = teleop_device.get_action()

            # Check for exit or reset
            if teleop_device.events["exit_teleop"]:
                break
            if teleop_device.events["back_robot_to_zero"]:
                move_robot_to_zero_position(robot, fps=FPS, duration=3.0, end_offset=ZERO_POSITION_OFFSET)
                vr_to_robot_joints_processor.reset()
                teleop_device.events["back_robot_to_zero"] = False
                continue

            # VR -> joint pose -> joint transition
            joint_action = vr_to_robot_joints_processor((vr_obs, robot_obs))

            # Send action to robot
            # The joint action to send is the actual action the robot takes, `ensure_safe_goal_position`
            # is applied in the action processor.
            _ = robot.send_action(joint_action)

            # Visualize
            if cfg.display_data:
                log_rerun_data(observation=robot_obs, action=joint_action)

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