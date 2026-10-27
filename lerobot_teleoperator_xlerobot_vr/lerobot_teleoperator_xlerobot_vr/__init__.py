"""VR (XLeVR / Quest) teleoperator for XLeRobot X. Importing registers ``--teleop.type=xlerobot_x_vr``."""

from .config_xlerobot_x_vr import XLeRobotXVRConfig
from .xlerobot_x_vr import XLeRobotXVR

__all__ = ["XLeRobotXVRConfig", "XLeRobotXVR"]
