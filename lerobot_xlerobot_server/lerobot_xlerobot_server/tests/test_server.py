"""Hardware-free tests: kinematics conventions, config, dispatch, ZMQ round trip.

    cd lerobot-xlerobot/lerobot_xlerobot_server && python -m pytest lerobot_xlerobot_server/tests -q

Needs placo + pyzmq (the lerobot[kinematics] env). XLEROBOT_URDF overrides the
URDF location.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from lerobot_xlerobot_server import PROTOCOL_VERSION
from lerobot_xlerobot_server.config import DEFAULT_URDF, load_config
from lerobot_xlerobot_server.kinematics import (  # noqa: F401
    ArmKinematics,
    mat_to_quat_xyzw,
    pose7_to_T,
    quat_xyzw_to_mat,
    urdf_joint_limits_deg,
)
from lerobot_xlerobot_server.robot_io import ALL_POSE_MOTORS, FakeRobotIO, min_jerk

URDF = os.environ.get("XLEROBOT_URDF", str(DEFAULT_URDF))
needs_urdf = pytest.mark.skipif(not Path(URDF).exists(), reason=f"no URDF at {URDF}")


def _cfg(tmp_path, **over):
    import yaml

    data = {"kinematics": {"urdf": URDF}}
    data.update(over)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    return load_config(p)


# -- rotations ---------------------------------------------------------------
def _random_rotation(rng):
    q = rng.normal(size=4)
    return quat_xyzw_to_mat(q / np.linalg.norm(q))


def test_quat_roundtrip_and_scipy_order():
    rng = np.random.default_rng(0)
    for _ in range(500):
        r = _random_rotation(rng)
        q = mat_to_quat_xyzw(r)
        assert q[3] >= 0 and abs(np.linalg.norm(q) - 1) < 1e-12
        assert np.allclose(quat_xyzw_to_mat(q), r, atol=1e-10)
    # 90 deg about +z: scipy xyzw is [0, 0, sin45, cos45]
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    assert np.allclose(mat_to_quat_xyzw(rz), [0, 0, np.sqrt(0.5), np.sqrt(0.5)])
    sp = pytest.importorskip("scipy.spatial.transform")
    for _ in range(100):
        r = _random_rotation(rng)
        q_sp = sp.Rotation.from_matrix(r).as_quat()
        q = mat_to_quat_xyzw(r)
        assert np.allclose(q, q_sp) or np.allclose(q, -q_sp)


# -- kinematics --------------------------------------------------------------
@needs_urdf
def test_joint_limits_from_urdf():
    lim = urdf_joint_limits_deg(URDF)
    assert lim.shape == (6, 2)
    assert np.allclose(lim[4], [-70.0, 70.0], atol=0.01)  # joint5 = wrist_yaw, +-1.22173 rad


@needs_urdf
def test_fk_reference_points():
    """Regression values from placo on so101_x.urdf (2026-10-29)."""
    kin = ArmKinematics(URDF, "gripper_tip")
    p0 = kin.fk_pose7(np.zeros(6))
    assert np.allclose(p0[:3], [0.4415, 0.0003, 0.2747], atol=2e-4)
    home = kin.fk_pose7([0, -80, 80, 0, 0, 0])
    assert np.allclose(home[:3], [0.2587, 0.0003, 0.1672], atol=2e-4)
    # tool axis is +X of the TCP frame: tip is 6 cm beyond gripper_back along it
    back = ArmKinematics(URDF, "gripper_back").fk_T([0, -80, 80, 0, 0, 0])
    tip = pose7_to_T(home)
    assert np.allclose(tip[:3, 3] - back[:3, 3], 0.06 * back[:3, 0], atol=1e-4)


# -- config ------------------------------------------------------------------
def test_config_defaults_and_overlay(tmp_path):
    cfg = _cfg(tmp_path, home={"offset": {"head_pitch": 10.0}})
    assert cfg["home"]["offset"] == {"head_pitch": 10.0}  # offset is replaced, not merged
    assert cfg["home"]["duration_s"] == 3.0
    assert cfg["on_exit"]["torque_off"] is False
    assert cfg["on_exit"]["return_to_start"] is True
    with pytest.raises(KeyError):
        _cfg(tmp_path, nope=1)


# -- dispatch ----------------------------------------------------------------
# A resting pose that is NOT the home pose, to see the return-to-start on exit.
REST = {m: 0.0 for m in ALL_POSE_MOTORS}
REST.update({"left_arm_shoulder_lift": -100.0, "left_arm_elbow_flex": 95.0,
             "right_arm_shoulder_lift": -100.0, "right_arm_elbow_flex": 95.0,
             "head_pitch": 10.0, "left_arm_gripper": 20.0})


@pytest.fixture
def server(tmp_path):
    from lerobot_xlerobot_server.server import XLeRobotServer

    s = XLeRobotServer(_cfg(tmp_path), FakeRobotIO(start_pose=REST))
    s.start()  # fake home: q = URDF zero + offset
    yield s
    for m in s.motion.values():
        m.stop()


@needs_urdf
def test_ping(server):
    rep = server.handle({"cmd": "ping"})
    assert rep["ok"] and rep["protocol"] == PROTOCOL_VERSION
    assert rep["tcp_frame"] == "gripper_tip" and "get_state" in rep["cmds"]


@needs_urdf
def test_get_state_after_home(server):
    rep = server.handle({"cmd": "get_state"})
    assert rep["ok"]
    for side in ("left", "right"):
        a = rep["arms"][side]
        assert np.allclose(a["q_deg"], [0, -80, 80, 0, 0, 0])
        assert np.allclose(a["ee_pose"][:3], [0.2587, 0.0003, 0.1672], atol=2e-4)
        assert len(a["ee_pose"]) == 7
    assert np.allclose(rep["head_q_deg"], [0, 35])
    assert rep["arms"]["left"]["torque"] is True
    one = server.handle({"cmd": "get_state", "arm": "left", "head": False})
    assert set(one["arms"]) == {"left"} and "head_q_deg" not in one


@needs_urdf
@pytest.mark.parametrize("req,code", [
    ({"cmd": "fly"}, "unknown_cmd"),
    ({"nocmd": 1}, "bad_request"),
    (None, "bad_request"),
    ({"cmd": "get_state", "arm": "middle"}, "bad_arg"),
])
def test_errors(server, req, code):
    rep = server.handle(req)
    assert rep["ok"] is False and rep["code"] == code


@needs_urdf
def test_internal_error_does_not_raise(server):
    def boom(side):
        raise IOError("bus timeout")

    server.io.read_arm = boom
    rep = server.handle({"cmd": "get_state"})
    assert rep["ok"] is False and rep["code"] == "internal" and "bus timeout" in rep["error"]


# -- ZMQ round trip ------------------------------------------------------------
def _free_port():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@needs_urdf
def test_zmq_roundtrip_and_timeout(server):
    from lerobot_xlerobot_server.client import ServerError, ServerTimeout, XLeRobotClient
    from lerobot_xlerobot_server.server import serve

    addr = f"tcp://127.0.0.1:{_free_port()}"
    c = XLeRobotClient(addr, timeout_s=0.3)
    # nothing listening yet -> timeout, and the client socket must still be usable after
    with pytest.raises(ServerTimeout):
        c.call("ping")

    stop = [False]
    t = threading.Thread(target=serve, args=(server, addr.replace("127.0.0.1", "*"), lambda: stop[0]), daemon=True)
    t.start()
    try:
        c.timeout_ms = 2000
        c._open()
        assert c.check_protocol()["protocol"] == PROTOCOL_VERSION
        rep = c.call("get_state", arm="right")
        assert set(rep["arms"]) == {"right"}
        with pytest.raises(ServerError) as ei:
            c.call("fly")
        assert ei.value.rep["code"] == "unknown_cmd"
        # REQ stays in lockstep after an error reply
        assert c.call("ping")["ok"]
    finally:
        stop[0] = True
        t.join(timeout=2)
        c.close()
    assert not t.is_alive()


# -- motion profile, torque, exit ----------------------------------------------
def test_min_jerk_profile():
    assert min_jerk(0) == 0 and min_jerk(1) == 1 and abs(min_jerk(0.5) - 0.5) < 1e-12
    eps = 1e-4  # zero slope at both ends
    assert min_jerk(eps) / eps < 1e-6 and (1 - min_jerk(1 - eps)) / eps < 1e-6


@needs_urdf
def test_start_pose_recorded_before_home(server):
    assert server.start_pose == REST
    assert server.io.pos["left_arm_shoulder_lift"] == -80.0  # now at home


@needs_urdf
def test_set_torque(server):
    rep = server.handle({"cmd": "set_torque", "arm": "left", "enable": False})
    assert rep["ok"] and rep["torque"] == {"left": False, "right": True}
    st = server.handle({"cmd": "get_state"})
    assert st["arms"]["left"]["torque"] is False and st["arms"]["right"]["torque"] is True
    # moving an arm whose torque is off is refused
    with pytest.raises(RuntimeError):
        server.io.move_joints({"left_arm_elbow_flex": 10.0}, 0.1, 30)
    assert server.handle({"cmd": "set_torque", "enable": True})["torque"] == {"left": True, "right": True}
    for bad in ({"cmd": "set_torque", "arm": "left"}, {"cmd": "set_torque", "enable": "no"}):
        r = server.handle(bad)
        assert r["ok"] is False and r["code"] == "bad_arg"


@needs_urdf
def test_exit_returns_to_start_even_with_torque_off(server):
    server.handle({"cmd": "set_torque", "arm": "right", "enable": False})
    server.stop()
    assert server.io.pos == pytest.approx(REST)
    assert server.io.torque_enabled("right") is True  # re-enabled to move back
    assert not server.io.connected


@needs_urdf
def test_exit_move_can_be_aborted(server):
    before = dict(server.io.pos)
    server.stop(should_abort=lambda: True)
    assert server.io.pos == before  # held where it was
    assert not server.io.connected


@needs_urdf
def test_exit_without_return(tmp_path):
    from lerobot_xlerobot_server.server import XLeRobotServer

    s = XLeRobotServer(_cfg(tmp_path, on_exit={"return_to_start": False}), FakeRobotIO(start_pose=REST))
    s.start()
    home = dict(s.io.pos)
    s.stop()
    assert s.io.pos == home


@needs_urdf
def test_exit_failure_still_disconnects(server):
    def boom(*a, **k):
        raise IOError("bus timeout")

    server.io.move_joints = boom
    server.stop()  # must not raise
    assert not server.io.connected


# -- cameras / get_obs -----------------------------------------------------------
@pytest.fixture
def cam_server(tmp_path):
    from lerobot_xlerobot_server.cameras import FakeCameraHub
    from lerobot_xlerobot_server.server import XLeRobotServer

    cams = FakeCameraHub({"head": {"path": "x"}, "left_wrist": {"path": "y", "width": 320, "height": 240}})
    s = XLeRobotServer(_cfg(tmp_path), FakeRobotIO(start_pose=REST), cams)
    s.start()
    yield s
    for m in s.motion.values():
        m.stop()


def test_camera_config_validation():
    from lerobot_xlerobot_server.cameras import normalize_cam_cfgs

    c = normalize_cam_cfgs({"head": {"path": "/dev/v4l/by-path/x"}})
    assert c["head"]["width"] == 640 and c["head"]["fourcc"] == "MJPG"
    with pytest.raises(KeyError):
        normalize_cam_cfgs({"head": {"width": 640}})            # no path
    with pytest.raises(KeyError):
        normalize_cam_cfgs({"head": {"path": "x", "fsp": 30}})  # typo


def test_cameras_config_replaces_default(tmp_path):
    cfg = _cfg(tmp_path, cameras={"head": {"path": "/dev/video0"}})
    assert cfg["cameras"] == {"head": {"path": "/dev/video0"}}


def test_jpeg_roundtrip_keeps_colour_order():
    cv2 = pytest.importorskip("cv2")
    from lerobot_xlerobot_server.cameras import encode_jpeg

    rgb = np.zeros((64, 64, 3), np.uint8)
    rgb[..., 0] = 255  # pure red in RGB
    dec = cv2.imdecode(np.frombuffer(encode_jpeg(rgb, 95), np.uint8), cv2.IMREAD_COLOR)  # BGR
    assert dec[..., 2].mean() > 240 and dec[..., 0].mean() < 15


@needs_urdf
def test_get_obs(cam_server):
    pytest.importorskip("cv2")
    rep = cam_server.handle({"cmd": "get_obs"})
    assert rep["ok"]
    assert [im["name"] for im in rep["images"]] == ["head", "left_wrist"]
    assert rep["images"][1]["shape"] == [240, 320, 3]
    blobs = rep["_attachments"]
    assert len(blobs) == 2 and all(b[:2] == b"\xff\xd8" for b in blobs)  # JPEG SOI
    assert rep["images"][0]["bytes"] == len(blobs[0])
    assert set(rep["state"]["arms"]) == {"left", "right"}
    one = cam_server.handle({"cmd": "get_obs", "cams": ["left_wrist"], "state": False, "quality": 50})
    assert [im["name"] for im in one["images"]] == ["left_wrist"] and "state" not in one


@needs_urdf
@pytest.mark.parametrize("req", [
    {"cmd": "get_obs", "cams": ["nope"]},
    {"cmd": "get_obs", "cams": "head"},
    {"cmd": "get_obs", "quality": 0},
    {"cmd": "get_obs", "quality": 85.5},
])
def test_get_obs_bad_args(cam_server, req):
    rep = cam_server.handle(req)
    assert rep["ok"] is False and rep["code"] == "bad_arg"


@needs_urdf
def test_get_obs_without_cameras(server):
    rep = server.handle({"cmd": "get_obs"})
    assert rep["ok"] and rep["images"] == [] and rep["_attachments"] == []


@needs_urdf
def test_camera_failure_happens_before_any_motion(tmp_path):
    from lerobot_xlerobot_server.cameras import FakeCameraHub
    from lerobot_xlerobot_server.server import XLeRobotServer

    class Broken(FakeCameraHub):
        def connect(self):
            raise RuntimeError("camera 'head' failed")

    io = FakeRobotIO(start_pose=REST)
    s = XLeRobotServer(_cfg(tmp_path), io, Broken())
    with pytest.raises(RuntimeError):
        s.start()
    assert io.pos == REST and io.writes == 0  # never homed


@needs_urdf
def test_zmq_get_obs_multipart(cam_server):
    pytest.importorskip("cv2")
    from lerobot_xlerobot_server.client import XLeRobotClient
    from lerobot_xlerobot_server.server import serve

    addr = f"tcp://127.0.0.1:{_free_port()}"
    stop = [False]
    t = threading.Thread(target=serve, args=(cam_server, addr.replace("127.0.0.1", "*"), lambda: stop[0]), daemon=True)
    t.start()
    c = XLeRobotClient(addr, timeout_s=2.0)
    try:
        rep, jpgs = c.get_obs()
        assert set(jpgs) == {"head", "left_wrist"}
        assert len(jpgs["head"]) == rep["images"][0]["bytes"]
        # single-frame replies still work after a multipart one (REQ lockstep intact)
        assert c.call("get_state", arm="left")["ok"]
    finally:
        stop[0] = True
        t.join(timeout=2)
        c.close()


# -- motion (protocol 0.4) -----------------------------------------------------------
HOME_TCP = np.array([0.2587, 0.0003, 0.1672])


def _target(server, side):
    return np.asarray(server.handle({"cmd": "get_state", "arm": side})["arms"][side]["target_ee"])


def _wait(server, side, timeout=3.0):
    assert server.motion[side].wait_idle(timeout)


@needs_urdf
def test_set_ee_target_moves_tcp_and_holds_orientation(server):
    t0 = _target(server, "left")
    assert np.allclose(t0[:3], HOME_TCP, atol=2e-4)
    for axis in range(3):
        goal = t0.copy()
        goal[axis] += 0.02
        rep = server.handle({"cmd": "set_ee_target", "arm": "left", "pose7": goal.tolist()})
        assert rep["ok"] and rep["accepted"], rep
        assert 1 <= rep["n_waypoints"] <= 6 and rep["pos_err_mm"] < 1.0
        _wait(server, "left")
        st = server.handle({"cmd": "get_state", "arm": "left"})["arms"]["left"]
        assert np.allclose(st["ee_pose"][:3], goal[:3], atol=1.5e-3)       # fake: no sag
        assert np.allclose(st["target_ee"], goal, atol=1e-9)
        R0 = quat_xyzw_to_mat(t0[3:])
        R1 = quat_xyzw_to_mat(st["ee_pose"][3:])
        assert np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1))) < 1.0
        back = server.handle({"cmd": "set_ee_target", "arm": "left", "pose7": t0.tolist()})
        assert back["accepted"]
        _wait(server, "left")
    # the right arm never moved
    assert np.allclose(_target(server, "right")[:3], HOME_TCP, atol=2e-4)


@needs_urdf
def test_waypoints_are_small_and_end_at_goal(server):
    from lerobot_xlerobot_server.motion import linear_waypoints

    q0, q1 = np.zeros(6), np.array([18.5, 0, 0, -15, 0, 0])
    w = linear_waypoints(q0, q1, 3.0, 5)
    assert len(w) == 6 and np.allclose(w[-1], q1)                 # capped at 5 intermediates
    assert len(linear_waypoints(q0, np.array([2.0, 0, 0, 0, 0, 0]), 3.0, 5)) == 1  # no intermediates
    steps = np.diff(np.vstack([q0] + w), axis=0)
    assert np.allclose(steps, steps[0])                            # evenly spaced


@needs_urdf
def test_set_ee_target_rejections(tmp_path, server):
    t0 = _target(server, "right")
    far = t0.copy(); far[0] += 0.06
    rep = server.handle({"cmd": "set_ee_target", "arm": "right", "pose7": far.tolist()})
    assert rep["ok"] and rep["accepted"] is False and rep["reason"] == "step_too_large"
    turned = t0.copy(); turned[3:] = [0, 0, np.sin(np.radians(10)), np.cos(np.radians(10))]  # 20 deg yaw
    rep = server.handle({"cmd": "set_ee_target", "arm": "right", "pose7": turned.tolist()})
    assert rep["reason"] == "step_too_large"
    server.handle({"cmd": "set_torque", "arm": "right", "enable": False})
    near = t0.copy(); near[2] += 0.01
    rep = server.handle({"cmd": "set_ee_target", "arm": "right", "pose7": near.tolist()})
    assert rep["reason"] == "torque_off"
    server.handle({"cmd": "set_torque", "arm": "right", "enable": True})
    for bad in ({"arm": "both", "pose7": near.tolist()}, {"arm": "right", "pose7": [0, 0, 0]},
                {"arm": "right", "pose7": [0, 0, 0, 0, 0, 0, 0]}, {"arm": "right", "pose7": [float("nan")] * 7}):
        r = server.handle({"cmd": "set_ee_target", **bad})
        assert r["ok"] is False and r["code"] == "bad_arg"


@needs_urdf
def test_calibrated_range_rejection(tmp_path):
    from lerobot_xlerobot_server.server import XLeRobotServer

    io = FakeRobotIO(start_pose=REST, ranges={"left_arm_shoulder_lift": (-85.0, -79.0)})
    s = XLeRobotServer(_cfg(tmp_path), io)
    s.start()
    try:
        t0 = _target(s, "left")
        goal = t0.copy(); goal[0] += 0.02  # needs shoulder_lift ~ -61 deg
        rep = s.handle({"cmd": "set_ee_target", "arm": "left", "pose7": goal.tolist()})
        assert rep["accepted"] is False and rep["reason"].startswith("calibrated_range:left_arm_shoulder_lift")
        assert io.pos["left_arm_shoulder_lift"] == -80.0
    finally:
        for m in s.motion.values():
            m.stop()


@needs_urdf
def test_ik_failure_is_rejected_not_executed(server):
    t0 = _target(server, "left")
    s = server
    # walk straight up until the arm can no longer hold the orientation / reach
    reasons = []
    for _ in range(40):
        goal = _target(s, "left"); goal[2] += 0.05
        rep = s.handle({"cmd": "set_ee_target", "arm": "left", "pose7": goal.tolist()})
        if not rep["accepted"]:
            reasons.append(rep["reason"])
            break
        _wait(s, "left")
    assert reasons and reasons[0] in ("ik_not_converged", "urdf_joint_limit")
    before = dict(s.io.pos)
    time.sleep(0.1)
    assert s.io.pos == before  # rejected command left the arm alone


@needs_urdf
def test_two_arms_commanded_back_to_back_move_together(server):
    t = {s: _target(server, s) for s in ("left", "right")}
    t0 = time.perf_counter()
    for side in ("left", "right"):
        g = t[side].copy(); g[0] += 0.02
        assert server.handle({"cmd": "set_ee_target", "arm": side, "pose7": g.tolist()})["accepted"]
    handled_ms = (time.perf_counter() - t0) * 1e3
    assert handled_ms < 100  # requests return before the motion is done
    assert server.motion["left"].moving and server.motion["right"].moving
    _wait(server, "left"); _wait(server, "right")


@needs_urdf
def test_new_command_replaces_queue(server):
    t0 = _target(server, "left")
    g1 = t0.copy(); g1[0] += 0.02
    g2 = t0.copy(); g2[2] += 0.02
    server.handle({"cmd": "set_ee_target", "arm": "left", "pose7": g1.tolist()})
    rep = server.handle({"cmd": "set_ee_target", "arm": "left", "pose7": g2.tolist()})
    assert rep["accepted"]
    _wait(server, "left")
    st = server.handle({"cmd": "get_state", "arm": "left"})["arms"]["left"]
    assert np.allclose(st["ee_pose"][:3], g2[:3], atol=1.5e-3)


@needs_urdf
def test_go_home_move_joints_hold(server):
    rep = server.handle({"cmd": "move_joints", "arm": "left", "q_deg": [10, -70, 70, 0, 0, 0], "duration_s": 0.3})
    assert rep["accepted"]
    _wait(server, "left")
    assert np.allclose(server.io.read_arm("left")[0], [10, -70, 70, 0, 0, 0])
    rep = server.handle({"cmd": "move_joints", "arm": "left", "q_deg": [0, 0, 0, 0, 90, 0]})  # wrist_yaw > 70
    assert rep["accepted"] is False and rep["reason"] == "urdf_joint_limit"
    rep = server.handle({"cmd": "go_home", "duration_s": 0.3})
    assert rep["arms"]["left"]["accepted"] and rep["arms"]["right"]["accepted"]
    _wait(server, "left")
    assert np.allclose(server.io.read_arm("left")[0], [0, -80, 80, 0, 0, 0])
    assert np.allclose(_target(server, "left")[:3], HOME_TCP, atol=2e-4)
    server.handle({"cmd": "move_joints", "arm": "right", "q_deg": [20, -60, 60, 0, 0, 0], "duration_s": 2.0})
    time.sleep(0.2)
    assert server.handle({"cmd": "hold", "arm": "right"})["ok"]
    assert not server.motion["right"].moving
    q_held = server.io.read_arm("right")[0]
    time.sleep(0.1)
    assert np.allclose(server.io.read_arm("right")[0], q_held)
    assert 0 < q_held[0] < 20  # stopped part way


@needs_urdf
def test_torque_on_resyncs_command(server):
    server.handle({"cmd": "set_torque", "arm": "left", "enable": False})
    # the hand moves the arm while torque is off
    server.io.pos.update({"left_arm_shoulder_lift": -70.0, "left_arm_elbow_flex": 75.0})
    server.handle({"cmd": "set_torque", "arm": "left", "enable": True})
    st = server.handle({"cmd": "get_state", "arm": "left"})["arms"]["left"]
    assert np.allclose(st["q_cmd_deg"][1:3], [-70, 75])
    assert np.allclose(st["target_ee"], st["ee_pose"], atol=1e-9)


# -- gripper (protocol 0.5) -----------------------------------------------------------
@needs_urdf
def test_set_gripper_value_and_width(server):
    rep = server.handle({"cmd": "set_gripper", "arm": "left", "value": 50})
    assert rep["ok"] and rep["accepted"] and rep["width_m"] == pytest.approx(0.0405)
    st = server.handle({"cmd": "get_state", "arm": "left"})["arms"]["left"]
    assert st["gripper"] == pytest.approx(50) and st["gripper_m"] == pytest.approx(0.0405)
    assert st["gripper_cmd"] == pytest.approx(50)
    rep = server.handle({"cmd": "set_gripper", "arm": "right", "width_m": 0.081})  # fully open
    assert rep["value"] == pytest.approx(0)
    assert server.io.pos["right_arm_gripper"] == pytest.approx(0)
    rep = server.handle({"cmd": "set_gripper", "arm": "right", "width_m": 0.0})    # closed
    assert rep["value"] == pytest.approx(100)
    rep = server.handle({"cmd": "set_gripper", "arm": "right", "value": 25})
    assert rep["width_m"] == pytest.approx(0.06075)
    assert server.io.pos["left_arm_gripper"] == pytest.approx(50)  # other arm untouched


@needs_urdf
def test_gripper_not_cancelled_by_arm_motion(server):
    server.handle({"cmd": "set_gripper", "arm": "left", "value": 80})
    t0 = _target(server, "left"); t0[2] += 0.01
    assert server.handle({"cmd": "set_ee_target", "arm": "left", "pose7": t0.tolist()})["accepted"]
    _wait(server, "left")
    assert server.io.pos["left_arm_gripper"] == pytest.approx(80)


@needs_urdf
@pytest.mark.parametrize("req", [
    {"arm": "left"}, {"arm": "left", "value": 10, "width_m": 0.01}, {"arm": "left", "value": 101},
    {"arm": "left", "width_m": 0.09}, {"arm": "left", "value": "x"}, {"arm": "both", "value": 10},
])
def test_set_gripper_bad_args(server, req):
    r = server.handle({"cmd": "set_gripper", **req})
    assert r["ok"] is False and r["code"] == "bad_arg"


@needs_urdf
def test_set_gripper_torque_off_and_reset(server):
    server.handle({"cmd": "set_gripper", "arm": "left", "value": 30})
    server.handle({"cmd": "set_torque", "arm": "left", "enable": False})
    assert server.handle({"cmd": "set_gripper", "arm": "left", "value": 60})["reason"] == "torque_off"
    server.handle({"cmd": "set_torque", "arm": "left", "enable": True})
    assert server.handle({"cmd": "get_state", "arm": "left"})["arms"]["left"]["gripper_cmd"] is None
