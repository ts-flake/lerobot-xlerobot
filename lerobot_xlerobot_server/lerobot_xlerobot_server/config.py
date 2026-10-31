"""Server configuration: a YAML file overlaid on the defaults below."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

import importlib.util

_URDF_REL = Path("examples") / "simulation" / "so101_x" / "so101_x.urdf"


def _default_urdf() -> Path:
    """so101_x.urdf next to the installed lerobot_robot_xlerobot (editable install:
    <dist>/lerobot_robot_xlerobot/__init__.py -> <dist>/examples/...); falls back to the
    repo layout (<repo>/lerobot_robot_xlerobot/examples/...)."""
    spec = importlib.util.find_spec("lerobot_robot_xlerobot")
    if spec is not None and spec.origin:  # origin is None for a namespace package
        cand = Path(spec.origin).resolve().parents[1] / _URDF_REL
        if cand.exists():
            return cand
    repo = Path(__file__).resolve().parents[2]
    return repo / "lerobot_robot_xlerobot" / _URDF_REL


DEFAULT_URDF = _default_urdf()

DEFAULTS: dict[str, Any] = {
    "robot": {
        # --robot.id of lerobot-calibrate; selects
        # ~/.cache/huggingface/lerobot/calibration/robots/xlerobot_x/<id>.json
        "id": "xlerobot_x",
        "port1": "/dev/feetech_left",   # left arm + head
        "port2": "/dev/feetech_right",  # right arm + base
    },
    "server": {
        "bind": "tcp://0.0.0.0:5560",
    },
    "kinematics": {
        # null -> the so101_x.urdf shipped with lerobot_robot_xlerobot (absolute path:
        # placo resolves package:// meshes only from an absolute URDF path)
        "urdf": None,
        # FK/IK frame. gripper_tip = fingertip centre (6 cm beyond gripper_back, the
        # frame the teleop IK uses). The tool axis is +X of this frame.
        "tcp_frame": "gripper_tip",
    },
    "home": {
        "enabled": True,
        "duration_s": 3.0,
        "fps": 30,
        # home pose = URDF zero + offset (deg, per motor; missing motors -> 0).
        # Also the default rollout begin pose. Same format as the teleop
        # configs' zero_position_offset.
        "offset": {
            "left_arm_shoulder_lift": -80.0,
            "left_arm_elbow_flex": 80.0,
            "right_arm_shoulder_lift": -80.0,
            "right_arm_elbow_flex": 80.0,
            "head_pitch": 35.0,
        },
    },
    "motion": {
        "hz": 30.0,               # waypoint rate of the per-arm motion threads
        "dq_max_deg": 3.0,        # max joint change between waypoints (set_ee_target)
        "n_max": 5,               # max intermediate waypoints per set_ee_target
        "max_step_m": 0.05,       # set_ee_target: max TCP translation vs the current target
        "max_rot_deg": 15.0,      # set_ee_target: max TCP rotation vs the current target
        "ik_pos_tol_m": 0.001,
        "ik_ori_tol_deg": 1.0,
        "move_joints_s": 2.0,     # default duration of move_joints
    },
    # Gripper: linear map from the LeRobot 0-100 value to the fingertip opening (m).
    # On XLeRobot X, 0 = fully open and 100 = closed (measured 2026-10-30).
    "gripper": {
        "width_at_0_m": 0.081,
        "width_at_100_m": 0.0,
    },
    # meshcat view (--viz) only: distance between the two arm bases along y.
    "viz": {
        "arm_separation_m": 0.3,
    },
    # Cameras served by get_obs; see cameras.py for the per-camera keys. Replaces the
    # default as a whole. `xlerobot-server --list-cameras` lists /dev/v4l/by-path links.
    "cameras": {},
    "on_exit": {
        # Move arms + head back to the pose recorded at startup (before homing),
        # assumed to be a safe resting pose. A second Ctrl+C aborts the move.
        "return_to_start": True,
        "duration_s": 3.0,
        # After that: false keeps the motors holding their last goal; true turns
        # torque off (fine if the start pose is supported, e.g. arms resting).
        "torque_off": False,
    },
}


# dict-valued keys whose user value replaces the default instead of merging into it
REPLACE_KEYS = {"offset", "cameras"}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            raise KeyError(f"unknown config key: {k}")
        if isinstance(out[k], dict) and isinstance(v, dict) and k not in REPLACE_KEYS:
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    over = {}
    if path is not None:
        with open(path) as f:
            over = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, over)
    urdf = cfg["kinematics"]["urdf"]
    cfg["kinematics"]["urdf"] = str(Path(urdf).expanduser().resolve() if urdf else DEFAULT_URDF)
    return cfg
