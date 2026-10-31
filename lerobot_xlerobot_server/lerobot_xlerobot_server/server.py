"""ZMQ REP server for the XLeRobot X arms.

    xlerobot-server --config server.yaml
    xlerobot-server --fake          # no hardware (FakeRobotIO + synthetic cameras)
    xlerobot-server --fake --viz    # ... and show the arms in meshcat

Lifecycle: connect buses -> record start pose -> connect cameras -> home -> start the
per-arm motion threads -> serve -> (Ctrl+C) -> stop motion -> return to the start
pose -> disconnect. A second Ctrl+C during a startup/exit move aborts it; the arms
then hold where they are.

Messages are JSON; replies are multipart (header + attachments); see PROTOCOL.md.
One request at a time (REP): motion requests only queue waypoints and return.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import time
import traceback
from typing import Any, Callable, Optional

import numpy as np
import zmq

from . import ARMS, PROTOCOL_VERSION
from .cameras import CameraHub, FakeCameraHub, encode_jpeg, list_v4l_by_path
from .config import load_config
from .ik import ArmIK, rotation_angle_deg
from .kinematics import ArmKinematics, pose7_to_T, T_to_pose7, urdf_joint_limits_deg
from .motion import ArmMotion, linear_waypoints, min_jerk_waypoints
from .robot_io import FakeRobotIO, RobotIO, XLeRobotIO, arm_motors

logger = logging.getLogger("xlerobot_server")


class RequestError(Exception):
    """A bad request: reported to the client as {"ok": false, "code": ...}."""

    def __init__(self, code: str, msg: str) -> None:
        super().__init__(msg)
        self.code = code


def _rejected(reason: str, **extra: Any) -> dict[str, Any]:
    """A well-formed motion request the server declines: ok, but accepted=false."""
    return {"accepted": False, "reason": reason, **extra}


class XLeRobotServer:
    def __init__(self, cfg: dict[str, Any], io: RobotIO, cams: Optional[CameraHub] = None) -> None:
        self.cfg = cfg
        self.io = io
        self.cams = cams if cams is not None else CameraHub({})
        kcfg, mcfg = cfg["kinematics"], cfg["motion"]
        self.kin = {s: ArmKinematics(kcfg["urdf"], kcfg["tcp_frame"]) for s in ARMS}
        self.ik = {s: ArmIK(self.kin[s], mcfg["ik_pos_tol_m"], mcfg["ik_ori_tol_deg"]) for s in ARMS}
        self.limits_deg = urdf_joint_limits_deg(kcfg["urdf"])
        self.motion: dict[str, ArmMotion] = {}
        self.gripper_cmd: dict[str, Optional[float]] = {s: None for s in ARMS}
        self.target_T: dict[str, np.ndarray] = {}
        self.started_at = time.time()
        self.start_pose: Optional[dict[str, float]] = None
        self.handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "ping": self._ping,
            "get_state": self._get_state,
            "get_obs": self._get_obs,
            "set_torque": self._set_torque,
            "set_ee_target": self._set_ee_target,
            "move_joints": self._move_joints,
            "go_home": self._go_home,
            "hold": self._hold,
            "set_gripper": self._set_gripper,
        }

    # -- lifecycle ---------------------------------------------------------
    # -- gripper units --------------------------------------------------------
    def gripper_width_m(self, value: float) -> float:
        g = self.cfg["gripper"]
        return g["width_at_0_m"] + (g["width_at_100_m"] - g["width_at_0_m"]) * float(value) / 100.0

    def gripper_value(self, width_m: float) -> float:
        g = self.cfg["gripper"]
        return 100.0 * (float(width_m) - g["width_at_0_m"]) / (g["width_at_100_m"] - g["width_at_0_m"])

    def home_q(self, side: str) -> np.ndarray:
        off = self.cfg["home"]["offset"]
        return np.array([float(off.get(m, 0.0)) for m in arm_motors(side)[:6]])

    def start(self, should_abort: Optional[Callable[[], bool]] = None) -> None:
        self.io.connect()
        self.start_pose = self.io.read_pose()
        logger.info("start pose recorded: %s",
                    {k: round(v, 1) for k, v in self.start_pose.items()})
        # cameras before any motion: a camera failure must not leave the arms at home
        self.cams.connect()
        if self.cams.names:
            logger.info("cameras: %s", self.cams.names)
        h = self.cfg["home"]
        if h["enabled"]:
            logger.info("moving to home pose (%.1f s)", h["duration_s"])
            targets = {m: float(h["offset"].get(m, 0.0)) for m in self.start_pose}
            unknown = set(h["offset"]) - set(targets)
            if unknown:
                raise KeyError(f"home.offset has unknown motors: {sorted(unknown)}")
            self.io.move_joints(targets, h["duration_s"], h["fps"], should_abort)
        for side in ARMS:
            m = ArmMotion(side, self.io, self.cfg["motion"]["hz"])
            self.motion[side] = m
            self.target_T[side] = self.kin[side].fk_T(m.q_cmd)
            m.start()
            p = T_to_pose7(self.target_T[side])
            logger.info("%s arm q=%s tcp=%s mm", side, np.round(m.q_cmd, 1), np.round(p[:3] * 1e3, 1))

    def stop(self, should_abort: Optional[Callable[[], bool]] = None) -> None:
        """Stop motion, return to the start pose (if configured), disconnect. Never raises."""
        for m in self.motion.values():
            try:
                m.stop()
            except Exception:
                logger.exception("stopping motion thread failed")
        e = self.cfg["on_exit"]
        try:
            if e["return_to_start"] and self.start_pose is not None:
                for side in ARMS:
                    if not self.io.torque_enabled(side):
                        logger.warning("%s arm torque was off; enabling it to return", side)
                        self.io.set_torque(side, True)
                logger.info("returning to start pose (%.1f s); Ctrl+C again to abort",
                            e["duration_s"])
                done = self.io.move_joints(self.start_pose, e["duration_s"],
                                           self.cfg["home"]["fps"], should_abort)
                logger.info("start pose %s", "reached" if done else "NOT reached (aborted)")
        except Exception:
            logger.exception("return to start pose failed; arms hold where they are")
        finally:
            try:
                self.cams.disconnect()
            except Exception:
                logger.exception("camera disconnect failed")
            try:
                self.io.disconnect()
            except Exception:
                logger.exception("disconnect failed")

    # -- dispatch ----------------------------------------------------------
    def handle(self, req: Any) -> dict[str, Any]:
        try:
            if not isinstance(req, dict) or "cmd" not in req:
                raise RequestError("bad_request", 'request must be a JSON object with "cmd"')
            fn = self.handlers.get(req["cmd"])
            if fn is None:
                raise RequestError("unknown_cmd", f"unknown cmd {req['cmd']!r}")
            rep = fn(req)
            rep.setdefault("ok", True)
        except RequestError as e:
            rep = {"ok": False, "code": e.code, "error": str(e)}
        except Exception as e:  # never let one request kill the server
            logger.exception("handler failed")
            rep = {"ok": False, "code": "internal", "error": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc(limit=5)}
        rep["protocol"] = PROTOCOL_VERSION
        rep["t"] = time.time()
        return rep

    @staticmethod
    def _arms_arg(req: dict[str, Any]) -> tuple[str, ...]:
        arm = req.get("arm", "both")
        if arm == "both":
            return ARMS
        if arm in ARMS:
            return (arm,)
        raise RequestError("bad_arg", f'arm must be "left", "right" or "both", got {arm!r}')

    @staticmethod
    def _one_arm(req: dict[str, Any]) -> str:
        arm = req.get("arm")
        if arm not in ARMS:
            raise RequestError("bad_arg", f'arm must be "left" or "right", got {arm!r}')
        return arm

    @staticmethod
    def _vec(req: dict[str, Any], key: str, n: int) -> np.ndarray:
        v = req.get(key)
        try:
            a = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            a = None
        if a is None or a.shape != (n,) or not np.all(np.isfinite(a)):
            raise RequestError("bad_arg", f'"{key}" must be a list of {n} finite numbers')
        return a

    def _check_joints(self, side: str, q: np.ndarray) -> Optional[str]:
        """None if q (6 joints, deg) is inside URDF limits and the calibrated range."""
        lo, hi = self.limits_deg[:, 0] - 1e-6, self.limits_deg[:, 1] + 1e-6
        if np.any(q < lo) or np.any(q > hi):
            return "urdf_joint_limit"
        bad = self.io.out_of_range(dict(zip(arm_motors(side)[:6], q)))
        return f"calibrated_range:{','.join(bad)}" if bad else None

    # -- handlers: state ------------------------------------------------------
    def _ping(self, req: dict[str, Any]) -> dict[str, Any]:
        return {
            "server": "lerobot_xlerobot_server",
            "uptime_s": time.time() - self.started_at,
            "tcp_frame": self.cfg["kinematics"]["tcp_frame"],
            "arms": list(ARMS),
            "cmds": sorted(self.handlers),
            "motion": {k: self.cfg["motion"][k] for k in ("hz", "max_step_m", "max_rot_deg")},
        }

    def _get_state(self, req: dict[str, Any]) -> dict[str, Any]:
        return self._state(self._arms_arg(req), bool(req.get("head", True)))

    def _state(self, sides: tuple[str, ...], head: bool) -> dict[str, Any]:
        arms = {}
        for side in sides:
            q, grip = self.io.read_arm(side)
            a = {
                "q_deg": q.tolist(),
                "gripper": grip,
                "gripper_m": self.gripper_width_m(grip),
                "gripper_cmd": self.gripper_cmd[side],
                "ee_pose": self.kin[side].fk_pose7(q).tolist(),
                "torque": self.io.torque_enabled(side),
            }
            m = self.motion.get(side)
            if m is not None:
                a["target_ee"] = T_to_pose7(self.target_T[side]).tolist()
                a["q_cmd_deg"] = np.asarray(m.q_cmd).tolist()
                a["moving"] = m.moving
                if m.last_error:
                    a["error"] = m.last_error
            arms[side] = a
        rep: dict[str, Any] = {"arms": arms}
        if head:
            rep["head_q_deg"] = self.io.read_head().tolist()
        return rep

    def _get_obs(self, req: dict[str, Any]) -> dict[str, Any]:
        names = req.get("cams", "all")
        if names == "all":
            names = self.cams.names
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise RequestError("bad_arg", '"cams" must be "all" or a list of camera names')
        unknown = [n for n in names if n not in self.cams.names]
        if unknown:
            raise RequestError("bad_arg", f"unknown cameras {unknown}; have {self.cams.names}")
        quality = req.get("quality", 85)
        if not isinstance(quality, int) or not 1 <= quality <= 100:
            raise RequestError("bad_arg", '"quality" must be an int in 1..100')
        images, blobs = [], []
        for n in names:
            rgb, age_ms = self.cams.grab(n)
            jpg = encode_jpeg(rgb, quality)
            images.append({"name": n, "shape": list(rgb.shape), "age_ms": age_ms, "bytes": len(jpg)})
            blobs.append(jpg)
        rep: dict[str, Any] = {"images": images, "_attachments": blobs}
        if req.get("state", True):
            rep["state"] = self._state(ARMS, True)
        return rep

    # -- handlers: torque -------------------------------------------------------
    def _set_torque(self, req: dict[str, Any]) -> dict[str, Any]:
        enable = req.get("enable")
        if not isinstance(enable, bool):
            raise RequestError("bad_arg", '"enable" must be true or false')
        for side in self._arms_arg(req):
            m = self.motion.get(side)
            if not enable and m is not None:
                m.resync()  # drop queued motion before letting go
            self.io.set_torque(side, enable)
            if enable:
                self.gripper_cmd[side] = None  # its goal was reset to the present position
            if enable and m is not None:
                m.resync()  # commanded joints := where the hand left the arm
                self.target_T[side] = self.kin[side].fk_T(m.q_cmd)
            logger.warning("%s arm torque %s", side, "ON" if enable else "OFF")
        return {"torque": {s: self.io.torque_enabled(s) for s in ARMS}}

    # -- handlers: motion ------------------------------------------------------
    def _motion_ready(self, side: str) -> Optional[dict[str, Any]]:
        if side not in self.motion:
            raise RequestError("internal", "motion not started")
        if not self.io.torque_enabled(side):
            return _rejected("torque_off")
        return None

    def _set_ee_target(self, req: dict[str, Any]) -> dict[str, Any]:
        side = self._one_arm(req)
        pose = self._vec(req, "pose7", 7)
        if not 0.5 < np.linalg.norm(pose[3:]) < 1.5:
            raise RequestError("bad_arg", "pose7 quaternion is not (close to) unit length")
        if (r := self._motion_ready(side)) is not None:
            return r
        mc = self.cfg["motion"]
        T_new = pose7_to_T(pose)
        T_cur = self.target_T[side]
        dp = float(np.linalg.norm(T_new[:3, 3] - T_cur[:3, 3]))
        dr = rotation_angle_deg(T_cur[:3, :3], T_new[:3, :3])
        if dp > mc["max_step_m"] + 1e-9 or dr > mc["max_rot_deg"] + 1e-9:
            return _rejected("step_too_large", step_m=dp, step_deg=dr)
        m = self.motion[side]
        q_from = np.asarray(m.q_cmd, float)
        res = self.ik[side].solve(q_from, T_new)
        extra = {"pos_err_mm": res.pos_err_m * 1e3, "ori_err_deg": res.ori_err_deg, "iters": res.iters}
        if not res.converged:
            return _rejected("ik_not_converged", **extra)
        if (why := self._check_joints(side, res.q_deg)) is not None:
            return _rejected(why, q_goal_deg=res.q_deg.tolist(), **extra)
        wps = linear_waypoints(q_from, res.q_deg, mc["dq_max_deg"], mc["n_max"])
        m.command(wps)
        self.target_T[side] = T_new
        return {"accepted": True, "q_goal_deg": res.q_deg.tolist(), "n_waypoints": len(wps),
                "duration_s": len(wps) / mc["hz"], **extra}

    def _joint_move(self, side: str, q_goal: np.ndarray, duration_s: float) -> dict[str, Any]:
        if (r := self._motion_ready(side)) is not None:
            return r
        if (why := self._check_joints(side, q_goal)) is not None:
            return _rejected(why)
        m = self.motion[side]
        m.command(min_jerk_waypoints(np.asarray(m.q_cmd, float), q_goal, duration_s, self.cfg["motion"]["hz"]))
        self.target_T[side] = self.kin[side].fk_T(q_goal)
        return {"accepted": True, "duration_s": duration_s}

    def _move_joints(self, req: dict[str, Any]) -> dict[str, Any]:
        side = self._one_arm(req)
        q = self._vec(req, "q_deg", 6)
        dur = float(req.get("duration_s", self.cfg["motion"]["move_joints_s"]))
        if not 0.2 <= dur <= 30:
            raise RequestError("bad_arg", '"duration_s" must be in [0.2, 30]')
        return self._joint_move(side, q, dur)

    def _go_home(self, req: dict[str, Any]) -> dict[str, Any]:
        dur = float(req.get("duration_s", self.cfg["home"]["duration_s"]))
        if not 0.2 <= dur <= 30:
            raise RequestError("bad_arg", '"duration_s" must be in [0.2, 30]')
        return {"arms": {s: self._joint_move(s, self.home_q(s), dur) for s in self._arms_arg(req)}}

    def _set_gripper(self, req: dict[str, Any]) -> dict[str, Any]:
        side = self._one_arm(req)
        has_v, has_w = "value" in req, "width_m" in req
        if has_v == has_w:
            raise RequestError("bad_arg", 'give exactly one of "value" (0-100) or "width_m"')
        try:
            value = float(req["value"]) if has_v else self.gripper_value(float(req["width_m"]))
        except (TypeError, ValueError):
            raise RequestError("bad_arg", "gripper target must be a number") from None
        if not np.isfinite(value) or not -1e-6 <= value <= 100 + 1e-6:
            g = self.cfg["gripper"]
            lo, hi = sorted((g["width_at_0_m"], g["width_at_100_m"]))
            raise RequestError("bad_arg", f"gripper target out of range: value {value:.1f} not in 0..100 "
                                          f"(width {lo}..{hi} m)")
        value = min(max(value, 0.0), 100.0)
        if not self.io.torque_enabled(side):
            return _rejected("torque_off")
        motor = arm_motors(side)[6]
        if self.io.out_of_range({motor: value}):
            return _rejected(f"calibrated_range:{motor}")
        # One register write; the servo moves on its own. Not queued behind arm waypoints,
        # so an arm command never cancels a gripper command (or the other way round).
        self.io._write_goals({motor: value})
        self.gripper_cmd[side] = value
        return {"accepted": True, "value": value, "width_m": self.gripper_width_m(value)}

    def _hold(self, req: dict[str, Any]) -> dict[str, Any]:
        for side in self._arms_arg(req):
            m = self.motion[side]
            m.resync()
            if self.io.torque_enabled(side):
                self.io._write_goals(dict(zip(m.motors, m.q_cmd)))
            self.target_T[side] = self.kin[side].fk_T(m.q_cmd)
        return {}


def serve(server: XLeRobotServer, bind: str, should_stop: Callable[[], bool]) -> None:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(bind)
    logger.info("serving protocol %s on %s", PROTOCOL_VERSION, bind)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    try:
        while not should_stop():
            if not poller.poll(200):  # wake regularly so Ctrl+C is honoured
                continue
            raw = sock.recv()
            try:
                req = json.loads(raw)
            except ValueError:
                req = None
            rep = server.handle(req)
            blobs = rep.pop("_attachments", [])
            # multipart: frame 0 = JSON header, then one frame per attachment
            sock.send_multipart([json.dumps(rep).encode()] + list(blobs))
    finally:
        sock.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML overlay on config.DEFAULTS")
    ap.add_argument("--bind", help="override server.bind, e.g. tcp://0.0.0.0:5560")
    ap.add_argument("--no-home", action="store_true", help="skip the startup home move")
    ap.add_argument("--no-return", action="store_true", help="skip the return to the start pose on exit")
    ap.add_argument("--fake", action="store_true", help="no hardware: FakeRobotIO + synthetic cameras")
    ap.add_argument("--viz", action="store_true", help="show measured arms + targets in meshcat")
    ap.add_argument("--list-cameras", action="store_true", help="list /dev/v4l/by-path capture devices and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.list_cameras:
        found = list_v4l_by_path()
        for link, dev in found:
            print(f"{dev:<14} {link}")
        if not found:
            print("no /dev/v4l/by-path/*-video-index0 devices")
        return
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    if args.no_home:
        cfg["home"]["enabled"] = False
    if args.no_return:
        cfg["on_exit"]["return_to_start"] = False
    if args.fake:
        io: RobotIO = FakeRobotIO()
        cams: CameraHub = FakeCameraHub(cfg["cameras"] or None)
    else:
        io = XLeRobotIO(cfg["robot"], cfg["on_exit"]["torque_off"])
        cams = CameraHub(cfg["cameras"])
    server = XLeRobotServer(cfg, io, cams)

    # 1st signal: stop serving (or abort the startup move). 2nd: abort the exit move.
    signals = [0]

    def _on_signal(signum, frame):
        signals[0] += 1
        logger.warning("signal %s (#%d)", signum, signals[0])

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    viz = None
    try:
        server.start(should_abort=lambda: signals[0] >= 1)
        if args.viz:
            from .viz import Viz
            viz = Viz(server)
            viz.start()
        if signals[0] == 0:
            serve(server, args.bind or cfg["server"]["bind"], lambda: signals[0] >= 1)
    except Exception:
        logger.exception("server error")
    finally:
        if viz is not None:
            viz.stop()
        seen = max(signals[0], 1)
        server.stop(should_abort=lambda: signals[0] > seen)
        logger.info("stopped (torque_off_on_exit=%s)", cfg["on_exit"]["torque_off"])
