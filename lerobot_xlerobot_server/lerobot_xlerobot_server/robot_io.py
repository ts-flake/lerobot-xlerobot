"""Hardware access: the XLeRobotX buses behind one lock per bus.

Bus layout (lerobot_robot_xlerobot): bus1 = left arm + head, bus2 = right arm + base.
Every read/write of a bus goes through that bus's RLock so that, from plan step 3 on,
the per-bus motion threads and the REP thread never interleave packets on one serial
line. The two buses never block each other.

``FakeRobotIO`` implements the same interface without hardware (tests, dry runs).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

ARMS = ("left", "right")
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
HEAD_MOTORS = ["head_yaw", "head_pitch"]
BUS_OF_ARM = {"left": "bus1", "right": "bus2"}


def arm_motors(side: str) -> list[str]:
    """Six arm joints (URDF order) followed by the gripper."""
    return [f"{side}_arm_{j}" for j in ARM_JOINTS] + [f"{side}_arm_gripper"]


def bus_of_motor(motor: str) -> str:
    return "bus2" if motor.startswith("right_arm_") else "bus1"


ALL_POSE_MOTORS = arm_motors("left") + arm_motors("right") + HEAD_MOTORS


def min_jerk(tau: float) -> float:
    """Quintic 0 -> 1 with zero velocity and acceleration at both ends."""
    tau = min(max(tau, 0.0), 1.0)
    return tau ** 3 * (10 - 15 * tau + 6 * tau * tau)


class RobotIO:
    """Interface.

    read_arm(side)          -> (q_deg[6], gripper_0_100)
    read_head()             -> q_deg[2]
    read_pose()             -> {motor: deg} for both arms (incl. grippers) + head
    move_joints(targets, duration_s, fps, should_abort) -> bool completed
                               min-jerk from the present position; motors not in
                               ``targets`` are left alone; returns False if aborted
                               (motors then hold where they are)
    set_torque(side, on)    -> enabling first sets the goal to the present position
    torque_enabled(side)    -> bool
    out_of_range(goals)     -> motors whose goal lies outside the calibrated range
    """

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def read_arm(self, side: str) -> tuple[np.ndarray, float]: ...
    def read_head(self) -> np.ndarray: ...
    def read_pose(self) -> dict[str, float]: ...
    def set_torque(self, side: str, on: bool) -> None: ...
    def torque_enabled(self, side: str) -> bool: ...
    def out_of_range(self, goals: dict[str, float]) -> list[str]: ...

    def _write_goals(self, goals: dict[str, float]) -> None: ...

    def move_joints(self, targets: dict[str, float], duration_s: float, fps: int,
                    should_abort: Optional[Callable[[], bool]] = None) -> bool:
        unknown = set(targets) - set(ALL_POSE_MOTORS)
        if unknown:
            raise KeyError(f"unknown motors: {sorted(unknown)}")
        for side in ARMS:
            if not self.torque_enabled(side) and any(m in targets for m in arm_motors(side)):
                raise RuntimeError(f"{side} arm torque is off; enable it before moving")
        start = self.read_pose()
        names = list(targets)
        q0 = np.array([start[m] for m in names])
        q1 = np.array([float(targets[m]) for m in names])
        n = max(1, int(round(duration_s * fps)))
        dt = duration_s / n
        for i in range(1, n + 1):
            t0 = time.perf_counter()
            if should_abort is not None and should_abort():
                logger.warning("move aborted at %d/%d; holding position", i - 1, n)
                return False
            q = q0 + (q1 - q0) * min_jerk(i / n)
            self._write_goals(dict(zip(names, q)))
            self._sleep(max(0.0, dt - (time.perf_counter() - t0)))
        return True

    def _sleep(self, s: float) -> None:
        time.sleep(s)


class XLeRobotIO(RobotIO):
    def __init__(self, robot_cfg: dict[str, Any], torque_off_on_exit: bool) -> None:
        self.robot_cfg = robot_cfg
        self.torque_off_on_exit = torque_off_on_exit
        self.robot = None
        self.locks = {"bus1": threading.RLock(), "bus2": threading.RLock()}
        self._torque = {s: True for s in ARMS}

    def connect(self) -> None:
        from lerobot_robot_xlerobot import XLeRobotX, XLeRobotXConfig

        cfg = XLeRobotXConfig(
            id=self.robot_cfg["id"],
            port1=self.robot_cfg["port1"],
            port2=self.robot_cfg["port2"],
            cameras={},  # cameras arrive in a later step
            disable_torque_on_disconnect=self.torque_off_on_exit,
        )
        robot = XLeRobotX(cfg)
        # calibrate=False: never drop into the interactive calibration prompt from a
        # server. If the motors do not hold the calibration yet, write the existing
        # file (what XLeRobotX.calibrate() does on ENTER); with no file, refuse.
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            if not robot.calibration:
                robot.disconnect()
                raise RuntimeError(
                    f"No calibration for robot.id={cfg.id!r} "
                    f"({robot.calibration_fpath}). Run lerobot-calibrate first."
                )
            logger.info("writing calibration file %s to the motors", robot.calibration_fpath)
            robot.bus1.write_calibration(
                {m: robot.calibration[m] for m in robot.left_arm_motors + robot.head_motors}
            )
            robot.bus2.write_calibration(
                {m: robot.calibration[m] for m in robot.right_arm_motors + robot.base_motors}
            )
        self.robot = robot
        for side in ARMS:
            have = getattr(robot, f"{side}_arm_motors")
            if have != arm_motors(side):
                raise RuntimeError(f"unexpected {side} arm motor order: {have}")
        if robot.head_motors != HEAD_MOTORS:
            raise RuntimeError(f"unexpected head motors: {robot.head_motors}")

    def disconnect(self) -> None:
        if self.robot is not None:
            with self.locks["bus1"], self.locks["bus2"]:
                self.robot.disconnect()
            self.robot = None

    def _bus(self, name: str):
        return getattr(self.robot, name)

    def read_arm(self, side: str) -> tuple[np.ndarray, float]:
        bus = BUS_OF_ARM[side]
        motors = arm_motors(side)
        with self.locks[bus]:
            pos = self._bus(bus).sync_read("Present_Position", motors)
        q = np.array([pos[m] for m in motors[:6]], dtype=float)
        return q, float(pos[motors[6]])

    def read_head(self) -> np.ndarray:
        with self.locks["bus1"]:
            pos = self.robot.bus1.sync_read("Present_Position", HEAD_MOTORS)
        return np.array([pos[m] for m in HEAD_MOTORS], dtype=float)

    def read_pose(self) -> dict[str, float]:
        out: dict[str, float] = {}
        with self.locks["bus1"]:
            out.update(self.robot.bus1.sync_read("Present_Position", arm_motors("left") + HEAD_MOTORS))
        with self.locks["bus2"]:
            out.update(self.robot.bus2.sync_read("Present_Position", arm_motors("right")))
        return {k: float(v) for k, v in out.items()}

    def _write_goals(self, goals: dict[str, float]) -> None:
        per_bus: dict[str, dict[str, float]] = {"bus1": {}, "bus2": {}}
        for m, v in goals.items():
            # clamp to the calibrated range of motion (XLeRobotX's own clip)
            per_bus[bus_of_motor(m)][m] = self.robot._clip_norm_value(m, float(v))
        for bus, vals in per_bus.items():
            if vals:
                with self.locks[bus]:
                    self._bus(bus).sync_write("Goal_Position", vals)

    def set_torque(self, side: str, on: bool) -> None:
        bus = BUS_OF_ARM[side]
        motors = arm_motors(side)
        with self.locks[bus]:
            b = self._bus(bus)
            if on:
                # The motor would jump back to its old Goal_Position the moment torque
                # comes on; make the goal the present position first.
                b.sync_write("Goal_Position", b.sync_read("Present_Position", motors))
                b.enable_torque(motors)
            else:
                b.disable_torque(motors)
        self._torque[side] = on

    def torque_enabled(self, side: str) -> bool:
        return self._torque[side]

    def out_of_range(self, goals: dict[str, float]) -> list[str]:
        return [m for m, v in goals.items()
                if abs(self.robot._clip_norm_value(m, float(v)) - float(v)) > 1e-6]


class FakeRobotIO(RobotIO):
    """In-memory stand-in: goals are reached instantly; no sleeping."""

    def __init__(self, start_pose: Optional[dict[str, float]] = None,
                 ranges: Optional[dict[str, tuple[float, float]]] = None) -> None:
        self.ranges = ranges or {}
        self._lock = threading.Lock()
        self.pos = {m: 0.0 for m in ALL_POSE_MOTORS}
        self.pos.update(start_pose or {})
        self._torque = {s: True for s in ARMS}
        self.connected = False
        self.writes = 0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_arm(self, side: str) -> tuple[np.ndarray, float]:
        m = arm_motors(side)
        return np.array([self.pos[x] for x in m[:6]]), self.pos[m[6]]

    def read_head(self) -> np.ndarray:
        return np.array([self.pos[m] for m in HEAD_MOTORS])

    def read_pose(self) -> dict[str, float]:
        return dict(self.pos)

    def _write_goals(self, goals: dict[str, float]) -> None:
        with self._lock:
            self.writes += 1
            self.pos.update({m: float(v) for m, v in goals.items()})

    def _sleep(self, s: float) -> None:
        pass

    def set_torque(self, side: str, on: bool) -> None:
        self._torque[side] = on

    def torque_enabled(self, side: str) -> bool:
        return self._torque[side]

    def out_of_range(self, goals: dict[str, float]) -> list[str]:
        return [m for m, v in goals.items()
                if m in self.ranges and not self.ranges[m][0] <= v <= self.ranges[m][1]]
