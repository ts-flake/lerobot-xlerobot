#!/usr/bin/env python

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("xlerobot_yaw_vr")
@dataclass
class XLeRobotYawVRConfig(TeleoperatorConfig):
    fps: int = 30 # Hz
    console_level: str = 'info' # Logging level
    stepsize: float | dict[str, float] = field(default_factory=lambda: {'ang': 20.0, 'pos': 0.04, 'gripper': 80.0}) # Stepsize in **velocity unit**, deg/s for angle, m/s for position
    enable_left_arm_control: bool = True # Enable left arm control with VR controller; disable to use only right arm
    enable_head_control: bool = False # Enable head control with VR headset;
    enable_base_control: bool = True
    use_thumbstick_for_head: bool = False # Use left VR controller's thumbstick for head control; False: use VR headset
    grip_to_activate: bool = True # Press the grip button to activate tracking; Reverse: press the grip button to deactivate
    record_dataset: bool = False # Use the left thumbstick for record control, if this is True, the `use_thumbstick_for_head` will be False
    use_placo_ik: bool = True # Use placo IK. Otherwise, use analytical IK, where IK only solve the 2-link planar arm in x-z plane, and All other joints are directly controlled.
    stream_cameras_to_vr: bool = False # Push robot camera frames to the in-VR overlay. OFF by default: teleop is unaffected unless explicitly enabled.

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

    # VR-to-robot dead zones, in **velocity units** (deg/s for rotations,
    # m/s for translations). The teleop multiplies each by the per-tick elapsed
    # time (from VR source timestamps) before comparing to the per-tick delta —
    # so behavior is frame-rate independent and slow steady motion no longer
    # gets filtered out at high VR refresh rates.
    vr2robot_dead_zones: dict[str, dict[str, float]] = field(default_factory=lambda: {
        'arm': {
            'roll':   5.0,  # deg/s
            'pitch':  5.0,
            'yaw':    10.0,
            'x':     0.025,  # m/s
            'y':     0.025,
            'z':     0.025,
        },
        'head': {
            'yaw':   10.0,   # deg/s
            'pitch': 10.0,
        }
    })

    # VR to robot multiplicative factors
    vr2robot_factors: dict[str, dict[str, float]] = field(default_factory=lambda: {
        'arm': {
            'roll': 1,
            'pitch': 1,
            'yaw': 1,
            'x': 1,
            'y': 1,
            'z': 1
            },
        'head': {
            'yaw': 2,
            'pitch': 2
        }
    })

    # VR to robot limits in **velocity units** (deg/s for rotations,
    # m/s for translations).
    vr2robot_limits: dict[str, dict[str, float]] = field(default_factory=lambda: {
        'arm': {'pos': 0.1, 'ang': 60.0}, 'head': {'ang': 45.0}
    })

    def __post_init__(self):
        if self.record_dataset:
            self.use_thumbstick_for_head = False
        
        if not self.use_placo_ik:
            self.vr2robot_factors['arm']['y'] = 200 # direct shoulder_pan angle control
            self.vr2robot_factors['arm']['pitch'] = 2
            # self.vr2robot_factors['arm']['roll'] = 2

    