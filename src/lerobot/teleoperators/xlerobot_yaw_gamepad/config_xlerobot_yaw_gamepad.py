#!/usr/bin/env python

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("xlerobot_yaw_gamepad")
@dataclass
class XLeRobotYawGamepadConfig(TeleoperatorConfig):
    fps: int = 30 # Hz
    console_level: str = 'info' # Logging level
    gamepad_id: int = 0 # Which controller to open when none is injected (device index)
    stepsize: float | dict[str, float] = field(default_factory=lambda: {'ang': 20.0, 'pos': 0.04, 'gripper': 60.0}) # Stepsize in **velocity unit**, deg/s for angle, m/s for position
    enable_left_arm_control: bool = True # Enable left arm control; disable to use only right arm
    enable_base_control: bool = True # Enable base control
    enable_head_control: bool = False # Enable head control
    record_dataset: bool = False # Use the left thumbstick for record control.
    use_placo_ik: bool = True # Use placo IK. Otherwise, use analytical IK, where IK only solve the 2-link planar arm in x-z plane, and All other joints are directly controlled.

    # Zero position offset in degrees
    zero_position_offset: dict[str, float] = field(default_factory=lambda: {
        'left_arm_shoulder_lift': -90,
        'left_arm_elbow_flex': 85,
        'left_arm_wrist_roll': -90,
        'left_arm_wrist_flex': 55,
        'right_arm_shoulder_lift': -90,
        'right_arm_elbow_flex': 85,
        'right_arm_wrist_flex': 55,
        'right_arm_wrist_roll': -90,
        'head_pitch': 35
    })

    stepsize_factors: dict[str, float] = field(default_factory=lambda: {
        'arm': {
            'roll': 2,
            'pitch': 1,
            'yaw': 1,
            'x': 1,
            'y': 1,
            'z': 1
        },
        'head': {
            'pitch': 4,
            'yaw': 4
        }
    })

    def __post_init__(self):
        if not self.use_placo_ik:
            self.stepsize_factors['arm']['roll'] = 4
            self.stepsize_factors['arm']['yaw'] = 4
            self.stepsize_factors['arm']['y'] = 4
