#!/usr/bin/env python
"""Shared helper for the hardware-light teleop -> IK -> placo visualization tests.

The tests drive the *real* lerobot IK processor steps
(``InverseKinematicsDeltaToJoints`` / ``AnalyticalInverseKinematicsDeltaToJoints``)
from a real teleoperator, exactly as ``teleoperate_vr.py`` / ``teleoperate_ps5.py``
do, but with no robot attached: each frame the commanded joints are fed straight
back as the next observation (perfect tracking). Rendering is placo's kinematic
display.

That feedback line is the seam where a future MuJoCo forward-dynamics renderer
plugs in: replace "observation = last commanded joints" with "observation = sim
joint state after stepping physics", and tracking lag / gravity / IK resync come
alive for free. placo itself cannot do this — its ``KinematicsSolver`` is purely
geometric and it has no contact/integration engine.
"""

import sys
from pathlib import Path

import numpy as np
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, robot_frame_viz

from lerobot.model.kinematics import RobotKinematics
from lerobot.model.rr_kinematics import RRKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.xlerobot_yaw import XLeRobotYaw, XLeRobotYawConfig
from lerobot.robots.xlerobot_yaw.robot_action_processor import (
    AnalyticalInverseKinematicsDeltaToJoints,
    InverseKinematicsDeltaToJoints,
)

URDF_PATH = str(Path(__file__).resolve().parent.parent / "simulation/so101_yaw/so101_yaw.urdf")
EE_FRAME = "gripper_back"
JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]  # placo joints, excl. gripper
RIGHT_ARM_JOINTS = [
    "right_arm_shoulder_pan", "right_arm_shoulder_lift", "right_arm_elbow_flex",
    "right_arm_wrist_flex", "right_arm_wrist_yaw", "right_arm_wrist_roll",
]


def _build_ik_pipeline(use_placo_ik: bool, motor_names: list[str]) -> RobotProcessorPipeline:
    """The real IK processor step wrapped in a minimal one-step pipeline."""
    if use_placo_ik:
        ik_step = InverseKinematicsDeltaToJoints(
            kinematics_left=RobotKinematics(urdf_path=URDF_PATH, target_frame_name=EE_FRAME, joint_names=JOINT_NAMES),
            kinematics_right=RobotKinematics(urdf_path=URDF_PATH, target_frame_name=EE_FRAME, joint_names=JOINT_NAMES),
            motor_names=motor_names,
        )
    else:
        ik_step = AnalyticalInverseKinematicsDeltaToJoints(
            kinematics_left=RRKinematics(use_degrees=True, offsets=[90, 90], reversed=[True, False]),
            kinematics_right=RRKinematics(use_degrees=True, offsets=[90, 90], reversed=[True, False]),
            motor_names=motor_names,
        )
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[ik_step],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def _init_observation(motor_names: list[str], zero_offset: dict[str, float]) -> dict[str, float]:
    """Zero-pose observation in degrees, with the teleop's zero_position_offset applied."""
    obs = {f"{m}.pos": 0.0 for m in motor_names}
    for motor, deg in zero_offset.items():
        if f"{motor}.pos" in obs:
            obs[f"{motor}.pos"] = float(deg)
    return obs


def _sync_display(disp: RobotKinematics, obs: dict[str, float]) -> None:
    """Push the right-arm joint degrees from the observation into the placo display model."""
    for name, motor in zip(JOINT_NAMES, RIGHT_ARM_JOINTS):
        disp.robot.set_joint(name, np.deg2rad(obs[f"{motor}.pos"]))
    disp.robot.update_kinematics()


def run_teleop_ik_test(teleop, use_placo_ik: bool, zero_offset: dict[str, float], fps: int = 30) -> None:
    """Loop: teleop action -> real IK step -> placo display, with perfect-tracking feedback.

    The right arm is the controlled/visualized arm (matching the legacy tests); the
    other arms sit at their zero pose so the full-robot IK step has valid observations.
    """
    robot = XLeRobotYaw(XLeRobotYawConfig())  # constructed only for the real motor-name lists (no connect)
    motor_names = robot.left_arm_motors + robot.right_arm_motors + robot.head_motors
    pipeline = _build_ik_pipeline(use_placo_ik, motor_names)
    obs = _init_observation(motor_names, zero_offset)

    disp = RobotKinematics(urdf_path=URDF_PATH, target_frame_name=EE_FRAME, joint_names=JOINT_NAMES)
    _sync_display(disp, obs)
    viz = robot_viz(disp.robot)

    print(f"\nMode: {'Placo IK' if use_placo_ik else 'Analytical IK'} — move the right controller/stick to teleoperate.\n")

    @schedule(interval=1.0 / fps)
    def loop():
        nonlocal obs

        action = teleop.get_action()

        if teleop.events.get("exit_teleop"):
            print("Exit.")
            teleop.disconnect()
            sys.exit(0)

        if teleop.events.get("back_robot_to_zero"):
            pipeline.reset()
            obs = _init_observation(motor_names, zero_offset)
            teleop.events["back_robot_to_zero"] = False
        else:
            joint_action = pipeline((action, obs))
            # Perfect-tracking feedback: commanded joints become the next observation.
            for key, val in joint_action.items():
                if isinstance(key, str) and key.endswith(".pos") and key.removesuffix(".pos") in motor_names:
                    obs[key] = float(val)

        _sync_display(disp, obs)
        viz.display(disp.robot.state.q)
        robot_frame_viz(disp.robot, EE_FRAME)

    run_loop()
