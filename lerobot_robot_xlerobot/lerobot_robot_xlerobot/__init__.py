"""XLeRobot X: dual-arm (SO101 + wrist yaw) mobile robot, as a LeRobot third-party plugin.

Importing this package registers ``--robot.type=xlerobot_x``.
"""

from .config_xlerobot_x import XLeRobotXConfig
from .xlerobot_x import XLeRobotX

__all__ = ["XLeRobotXConfig", "XLeRobotX"]
