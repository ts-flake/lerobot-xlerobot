#!/usr/bin/env python
"""Drive the real PS5 gamepad teleop through the real IK processor step, in placo (no robot).

Reuses ``XLeRobotYawGamepad`` and ``InverseKinematicsDeltaToJoints`` /
``AnalyticalInverseKinematicsDeltaToJoints`` exactly as ``teleoperate.py`` does — the
only difference is there is no robot, so commanded joints are fed back as the next
observation (perfect tracking) and rendering is placo's kinematic display.

Usage:
  uv run examples/xlerobot_yaw/tests/test_teleop_ps5.py --mode ik
  uv run examples/xlerobot_yaw/tests/test_teleop_ps5.py --mode analytical

Controls (right arm): right stick = EE x/z, RB + stick = EE y/pitch,
  RB + a/b/x/y = roll/yaw, Logo = exit.
"""

import argparse

from lerobot.teleoperators.xlerobot_yaw_gamepad import XLeRobotYawGamepad, XLeRobotYawGamepadConfig
from lerobot.teleoperators.xlerobot_yaw_gamepad.gamepad_utils import PS5Gamepad
from sim_utils import run_teleop_ik_test

parser = argparse.ArgumentParser(description="Test the real IK step from a PS5 gamepad without a robot.")
parser.add_argument("--mode", choices=["ik", "analytical"], default="ik",
                    help="'ik' = placo full 6-DOF IK; 'analytical' = 2-link analytical IK")
parser.add_argument("--fps", default=30)
args = parser.parse_args()

config = XLeRobotYawGamepadConfig(use_placo_ik=(args.mode == "ik"), fps=int(args.fps))
teleop = XLeRobotYawGamepad(config, PS5Gamepad(id=0))
teleop.connect()

run_teleop_ik_test(teleop, config.use_placo_ik, config.zero_position_offset, fps=config.fps)
