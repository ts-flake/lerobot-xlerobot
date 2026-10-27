"""Gamepad teleoperator for XLeRobot X. Importing registers ``--teleop.type=xlerobot_x_gamepad``."""

from .config_xlerobot_x_gamepad import XLeRobotXGamepadConfig
from .xlerobot_x_gamepad import XLeRobotXGamepad

__all__ = ["XLeRobotXGamepadConfig", "XLeRobotXGamepad"]
