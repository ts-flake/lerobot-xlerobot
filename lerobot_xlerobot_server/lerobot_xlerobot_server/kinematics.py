"""Per-arm forward kinematics on so101_x.urdf (placo).

Conventions at the server boundary (also in PROTOCOL.md):
  joints   degrees, motor order shoulder_pan .. wrist_roll == URDF joint1 .. joint6
           (calibration zero == URDF zero, same sign; this is what the teleop IK
           already relies on)
  pose7    [x, y, z, qx, qy, qz, qw]: metres, quaternion xyzw (scipy order), of the
           TCP frame in that arm's own base frame (URDF root)

Each arm gets its own placo RobotWrapper: placo keeps joint state inside the
wrapper, so sharing one between arms (or threads) would race. FK and IK (ik.py) of one
arm share that wrapper under one lock.

The wrappers load a mesh-free copy of the URDF (``kinematic_urdf``): with meshes, one
placo wrapper of so101_x takes ~230 MB and ~1.5-3 s to load; without, a few MB.
"""
from __future__ import annotations

import hashlib
import tempfile
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF_JOINTS = [f"joint{i}" for i in range(1, 7)]


def mat_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion [x, y, z, w] with w >= 0."""
    r = np.asarray(r, dtype=float)
    t = np.trace(r)
    if t > 0.0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def T_to_pose7(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[:3, 3], mat_to_quat_xyzw(T[:3, :3])])


def pose7_to_T(pose7) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = pose7[:3]
    T[:3, :3] = quat_xyzw_to_mat(pose7[3:7])
    return T


def urdf_joint_limits_deg(urdf_path: str, joints=URDF_JOINTS) -> np.ndarray:
    """[[lower, upper], ...] in degrees for ``joints``, read from the URDF."""
    root = ET.parse(urdf_path).getroot()
    lim = {}
    for j in root.iter("joint"):
        el = j.find("limit")
        if el is not None and j.get("name") in joints:
            lim[j.get("name")] = (float(el.get("lower")), float(el.get("upper")))
    missing = [j for j in joints if j not in lim]
    if missing:
        raise ValueError(f"URDF {urdf_path} has no limits for {missing}")
    return np.rad2deg(np.array([lim[j] for j in joints]))


def kinematic_urdf(urdf_path: str) -> str:
    """Path to a copy of ``urdf_path`` without <visual>/<collision> (cached in the temp dir)."""
    src = Path(urdf_path).read_bytes()
    out = Path(tempfile.gettempdir()) / f"xlerobot_kin_{hashlib.sha1(src).hexdigest()[:12]}.urdf"
    if not out.exists():
        root = ET.fromstring(src)
        for link in root.iter("link"):
            for tag in ("visual", "collision"):
                for el in link.findall(tag):
                    link.remove(el)
        tmp = out.with_name(out.name + f".{threading.get_ident()}.tmp")
        ET.ElementTree(root).write(tmp, encoding="utf-8", xml_declaration=True)
        tmp.replace(out)  # atomic: concurrent servers never read a partial file
    return str(out)


class ArmKinematics:
    """FK for one arm. Thread-safe: ``lock`` guards the wrapper (shared with ArmIK)."""

    def __init__(self, urdf_path: str, tcp_frame: str = "gripper_tip") -> None:
        import placo  # lerobot[kinematics]

        self.robot = placo.RobotWrapper(kinematic_urdf(urdf_path))
        self.tcp_frame = tcp_frame
        self.limits_deg = urdf_joint_limits_deg(urdf_path)
        self.lock = threading.RLock()
        self.fk_T(np.zeros(6))  # fail fast on a bad frame name

    def fk_T(self, q_deg) -> np.ndarray:
        q = np.deg2rad(np.asarray(q_deg, dtype=float)[:6])
        with self.lock:
            for name, v in zip(URDF_JOINTS, q):
                self.robot.set_joint(name, v)
            self.robot.update_kinematics()
            return np.array(self.robot.get_T_world_frame(self.tcp_frame))

    def fk_pose7(self, q_deg) -> np.ndarray:
        return T_to_pose7(self.fk_T(q_deg))
