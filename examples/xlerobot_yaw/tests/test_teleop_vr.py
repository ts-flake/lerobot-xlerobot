#!/usr/bin/env python
"""Drive the real VR teleop through the real IK processor step, visualized in placo (no robot).

Reuses ``XLeRobotYawVR`` and ``InverseKinematicsDeltaToJoints`` /
``AnalyticalInverseKinematicsDeltaToJoints`` exactly as ``teleoperate.py`` does — the
only difference is there is no robot, so commanded joints are fed back as the next
observation (perfect tracking) and rendering is placo's kinematic display.

Usage:
  uv run examples/xlerobot_yaw/tests/test_teleop_vr.py --mode ik
  uv run examples/xlerobot_yaw/tests/test_teleop_vr.py --mode analytical

Controls (right controller): move/rotate = EE x/y/z + roll/pitch/yaw,
  Right B = recalibrate, Left X = exit, Left Y = reset to zero.
"""

import argparse

from lerobot.teleoperators.xlerobot_yaw_vr import XLeRobotYawVR, XLeRobotYawVRConfig
from lerobot.utils.robot_utils import precise_sleep
from sim_utils import run_teleop_ik_test

parser = argparse.ArgumentParser(description="Test the real IK step from VR controllers without a robot.")
parser.add_argument("--mode", choices=["ik", "analytical"], default="ik",
                    help="'ik' = placo full 6-DOF IK; 'analytical' = 2-link analytical IK")
parser.add_argument("--fps", default=30)
args = parser.parse_args()

config = XLeRobotYawVRConfig(use_placo_ik=(args.mode == "ik"), fps=int(args.fps))
teleop = XLeRobotYawVR(config)
teleop.connect()

print("Waiting for VR connection...")
while not teleop.is_connected:
    precise_sleep(0.1)
print("VR connected. Waiting for calibration (press B on right controller)...")
while not teleop.is_calibrated:
    precise_sleep(0.1)

run_teleop_ik_test(teleop, config.use_placo_ik, config.zero_position_offset, fps=config.fps)
