"""Command-line client: talk to the server without any harness.

    xlerobot-server-cli --addr tcp://<jetson-ip>:5560 ping
    xlerobot-server-cli --addr ... state            # both arms, once
    xlerobot-server-cli --addr ... state --arm left --watch 5
    xlerobot-server-cli --addr ... torque off --arm left    # asks for confirmation
    xlerobot-server-cli --addr ... torque on --arm left
    xlerobot-server-cli --addr ... obs --out obs/              # save one JPEG per camera
    xlerobot-server-cli --addr ... obs --n 50                  # round-trip latency stats
    xlerobot-server-cli --addr ... move --arm left --dx 0.01   # TCP +1 cm in the arm base frame
    xlerobot-server-cli --addr ... move --arm left --dz 0.02 --repeat 20 --back   # accuracy test
    xlerobot-server-cli --addr ... home [--arm left]
    xlerobot-server-cli --addr ... hold [--arm left]
    xlerobot-server-cli --addr ... gripper --arm left 50       # 0-100
    xlerobot-server-cli --addr ... gripper --arm left --width 0.03

`state` prints the TCP position in millimetres in each arm's base frame, for
checking FK against a ruler (PRD criterion 3). For that, turn the arm's torque off,
pose it by hand while watching `state --watch 2`, then turn torque back on.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys

from .client import ServerError, ServerTimeout, XLeRobotClient
from .kinematics import mat_to_quat_xyzw, quat_xyzw_to_mat


def _rot_xyz_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotation about the base x, then y, then z axes (extrinsic), degrees."""
    a, b, c = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _angle_deg(qa, qb) -> float:
    Ra, Rb = quat_xyzw_to_mat(qa), quat_xyzw_to_mat(qb)
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def _wait_idle(c: XLeRobotClient, arm: str, timeout_s: float = 10.0) -> dict:
    t_end = time.monotonic() + timeout_s
    while True:
        a = c.call("get_state", arm=arm, head=False)["arms"][arm]
        if not a.get("moving") or time.monotonic() > t_end:
            return a
        time.sleep(0.02)


JOINT_NAMES = ["pan", "lift", "elbow", "w_flex", "w_yaw", "w_roll"]


def _diag(c: XLeRobotClient, args, pre: dict, rep: dict, idle: dict, t_sent: float, t_idle: float) -> None:
    """Joint tracking after one command: is the error steady-state, slow, or a failure?"""
    q_goal = np.asarray(rep["q_goal_deg"])
    print(f"  diag: queue drained {1e3 * (t_idle - t_sent):.0f} ms after sending "
          f"({rep['n_waypoints']} waypoints, planned {1e3 * rep['duration_s']:.0f} ms)")
    print(f"  diag: q_goal - q_pre_measured  {np.round(q_goal - np.asarray(pre['q_deg']), 1)}  "
          f"({' '.join(JOINT_NAMES)})")
    if idle.get("error"):
        print(f"  diag: motion thread error: {idle['error']}")
    p_tgt = np.asarray(idle["target_ee"])[:3]
    for t in (0.1, 0.3, 0.6, 1.0, 2.0, 3.0):
        time.sleep(max(0.0, t_idle + t - time.perf_counter()))
        a = c.call("get_state", arm=args.arm, head=False)["arms"][args.arm]
        err = np.asarray(a["q_cmd_deg"]) - np.asarray(a["q_deg"])
        dist = np.linalg.norm(np.asarray(a["ee_pose"])[:3] - p_tgt) * 1e3
        print(f"  diag: +{t:.1f}s  q_cmd - q_measured {np.round(err, 1)}  |target - tcp| {dist:5.1f} mm")


def _move(c: XLeRobotClient, args) -> None:
    d = np.array([args.dx, args.dy, args.dz])
    drot = np.array([args.droll, args.dpitch, args.dyaw])
    rows = []
    for i in range(max(1, args.repeat)):
        sign = -1.0 if (args.back and i % 2 == 1) else 1.0
        pre = _wait_idle(c, args.arm)
        tgt = np.asarray(pre["target_ee"])
        R = _rot_xyz_deg(*(sign * drot)) @ quat_xyzw_to_mat(tgt[3:])
        new = np.concatenate([tgt[:3] + sign * d, mat_to_quat_xyzw(R)])
        rep = c.call("set_ee_target", arm=args.arm, pose7=new.tolist())
        if not rep["accepted"]:
            extra = {k: v for k, v in rep.items() if k not in ("ok", "protocol", "t", "accepted", "reason")}
            print(f"#{i:<3} REJECTED: {rep['reason']}  {extra}")
            return
        t_sent = time.perf_counter()
        idle = _wait_idle(c, args.arm)
        t_idle = time.perf_counter()
        if args.diag and i == 0:
            _diag(c, args, pre, rep, idle, t_sent, t_idle)
        time.sleep(args.settle)
        post = c.call("get_state", arm=args.arm, head=False)["arms"][args.arm]
        if post.get("error"):
            print(f"#{i:<3} motion thread error: {post['error']}")
        p0, p1 = np.asarray(pre["ee_pose"]), np.asarray(post["ee_pose"])
        moved = p1[:3] - p0[:3]
        cmd = sign * d
        n = np.linalg.norm(cmd)
        along = float(moved @ cmd / n) if n > 0 else 0.0
        off = float(np.linalg.norm(moved - (along * cmd / n if n > 0 else 0)))
        rot = _angle_deg(p0[3:], p1[3:])
        track = float(np.linalg.norm(np.asarray(post["target_ee"])[:3] - p1[:3]))
        rows.append((along, off, rot, track))
        print(f"#{i:<3} {'+' if sign > 0 else '-'}  travel {along * 1e3:6.2f} mm  off-axis {off * 1e3:5.2f} mm  "
              f"rot {rot:5.2f} deg  |target-measured| {track * 1e3:5.2f} mm  "
              f"(ik {rep['pos_err_mm']:.2f} mm, {rep['n_waypoints']} wp)")
    if len(rows) > 1:
        r = np.array(rows)
        print(f"summary n={len(r)}: travel {r[:, 0].mean() * 1e3:.2f} mm "
              f"[{r[:, 0].min() * 1e3:.2f}, {r[:, 0].max() * 1e3:.2f}]  off-axis max {r[:, 1].max() * 1e3:.2f} mm  "
              f"rot max {r[:, 2].max():.2f} deg  tracking max {r[:, 3].max() * 1e3:.2f} mm")


def _rpy_deg(q_xyzw) -> np.ndarray:
    """Extrinsic xyz (roll, pitch, yaw), degrees -- for reading only."""
    r = quat_xyzw_to_mat(q_xyzw)
    pitch = np.arcsin(-np.clip(r[2, 0], -1, 1))
    roll = np.arctan2(r[2, 1], r[2, 2])
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return np.rad2deg([roll, pitch, yaw])


def _print_state(rep: dict) -> None:
    for side, a in rep["arms"].items():
        p = np.asarray(a["ee_pose"])
        torque = ("" if a.get("torque", True) else "  TORQUE OFF") + ("  moving" if a.get("moving") else "")
        width = f" ({a['gripper_m'] * 1e3:.1f} mm)" if "gripper_m" in a else ""
        print(f"{side:>5}  q_deg {np.round(a['q_deg'], 1)}  gripper {a['gripper']:.1f}{width}{torque}")
        print(f"       tcp_mm {np.round(p[:3] * 1e3, 1)}  rpy_deg {np.round(_rpy_deg(p[3:]), 1)}")
    if "head_q_deg" in rep:
        print(f" head  q_deg {np.round(rep['head_q_deg'], 1)}")


def main(argv: list[str] | None = None) -> None:
    try:
        _main(argv)
    except (ServerError, ServerTimeout) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


def _main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--addr", default="tcp://localhost:5560")
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--json", action="store_true", help="print raw replies")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ping")
    st = sub.add_parser("state")
    st.add_argument("--arm", default="both", choices=["left", "right", "both"])
    st.add_argument("--watch", type=float, default=0.0, help="repeat at this rate (Hz)")
    tq = sub.add_parser("torque", help="debug: turn an arm's torque on/off")
    tq.add_argument("state", choices=["on", "off"])
    tq.add_argument("--arm", default="both", choices=["left", "right", "both"])
    tq.add_argument("-y", "--yes", action="store_true", help="skip the confirmation for 'off'")
    ob = sub.add_parser("obs", help="get_obs: camera JPEGs (+ state)")
    ob.add_argument("--cams", default="all", help='"all" or comma-separated names')
    ob.add_argument("--quality", type=int, default=85)
    ob.add_argument("--out", help="directory to save <name>.jpg (last round)")
    ob.add_argument("--n", type=int, default=1, help="repeat N times and report latency")
    mv = sub.add_parser("move", help="set_ee_target relative to the current target (arm base frame)")
    mv.add_argument("--arm", required=True, choices=["left", "right"])
    for k in ("dx", "dy", "dz"):
        mv.add_argument(f"--{k}", type=float, default=0.0, help="metres")
    for k in ("droll", "dpitch", "dyaw"):
        mv.add_argument(f"--{k}", type=float, default=0.0, help="degrees, about the base axes through the TCP")
    mv.add_argument("--repeat", type=int, default=1)
    mv.add_argument("--back", action="store_true", help="alternate +delta / -delta")
    mv.add_argument("--settle", type=float, default=0.3, help="seconds to wait after the move")
    mv.add_argument("--diag", action="store_true",
                    help="after the first move, print per-joint tracking error over 3 s")
    hm = sub.add_parser("home", help="go_home")
    hm.add_argument("--arm", default="both", choices=["left", "right", "both"])
    hm.add_argument("--duration", type=float, default=None)
    hd = sub.add_parser("hold", help="drop queued motion, hold the measured pose")
    hd.add_argument("--arm", default="both", choices=["left", "right", "both"])
    gp = sub.add_parser("gripper", help="set_gripper (0-100 or --width metres), then report")
    gp.add_argument("--arm", required=True, choices=["left", "right"])
    gp.add_argument("value", nargs="?", type=float, help="0..100 (on XLeRobot X: 0 = open, 100 = closed)")
    gp.add_argument("--width", type=float, help="opening in metres instead of a 0-100 value")
    gp.add_argument("--wait", type=float, default=1.5, help="seconds before reading back")
    args = ap.parse_args(argv)

    c = XLeRobotClient(args.addr, args.timeout)
    if args.cmd == "ping":
        rep = c.check_protocol()
        print(json.dumps(rep, indent=2) if args.json else
              f"ok  protocol {rep['protocol']}  tcp_frame {rep['tcp_frame']}  cmds {rep['cmds']}")
        return
    if args.cmd == "torque":
        on = args.state == "on"
        if not on and not args.yes:
            ans = input(f"Torque OFF for {args.arm} arm(s): the arm falls under gravity. "
                        "Support it. Continue? [y/N] ")
            if ans.strip().lower() != "y":
                print("aborted")
                return
        rep = c.call("set_torque", arm=args.arm, enable=on)
        print(json.dumps(rep) if args.json else f"ok  torque {rep['torque']}")
        return
    if args.cmd == "move":
        _move(c, args)
        return
    if args.cmd == "home":
        kw = {"arm": args.arm}
        if args.duration:
            kw["duration_s"] = args.duration
        print(json.dumps(c.call("go_home", **kw)["arms"]))
        return
    if args.cmd == "hold":
        c.call("hold", arm=args.arm)
        print("ok")
        return
    if args.cmd == "gripper":
        if (args.value is None) == (args.width is None):
            ap.error("gripper: give a 0-100 value or --width, not both")
        kw = {"value": args.value} if args.value is not None else {"width_m": args.width}
        rep = c.call("set_gripper", arm=args.arm, **kw)
        if not rep["accepted"]:
            print(f"REJECTED: {rep['reason']}")
            return
        print(f"set {rep['value']:.1f} ({rep['width_m'] * 1e3:.1f} mm)")
        t0 = time.perf_counter()
        while True:
            time.sleep(0.25)
            a = c.call("get_state", arm=args.arm, head=False)["arms"][args.arm]
            dt = time.perf_counter() - t0
            print(f"  +{dt:.2f}s  measured {a['gripper']:.1f} ({a['gripper_m'] * 1e3:.1f} mm)")
            if dt >= args.wait:
                return
    if args.cmd == "obs":
        cams = "all" if args.cams == "all" else [x for x in args.cams.split(",") if x]
        rtts = []
        for _ in range(max(1, args.n)):
            t0 = time.perf_counter()
            rep, jpgs = c.get_obs(cams=cams, quality=args.quality)
            rtts.append((time.perf_counter() - t0) * 1e3)
        if args.json:
            print(json.dumps(rep))
        for im in rep["images"]:
            print(f"{im['name']:>12}  {im['shape'][1]}x{im['shape'][0]}  "
                  f"{im['bytes'] / 1024:.0f} KiB  age {im['age_ms']:.0f} ms")
        if not rep["images"]:
            print("no cameras configured on the server")
        total_kib = sum(im["bytes"] for im in rep["images"]) / 1024
        r = np.array(rtts)
        print(f"  rtt ms  median {np.median(r):.0f}  p90 {np.percentile(r, 90):.0f}  "
              f"max {r.max():.0f}  (n={len(r)}, {total_kib:.0f} KiB/obs)")
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            for name, data in jpgs.items():
                (out / f"{name}.jpg").write_bytes(data)
            print(f"  saved {len(jpgs)} images to {out}/")
        return
    while True:
        t0 = time.perf_counter()
        rep = c.call("get_state", arm=args.arm)
        rtt_ms = (time.perf_counter() - t0) * 1e3
        if args.json:
            print(json.dumps(rep))
        else:
            _print_state(rep)
            print(f"       rtt {rtt_ms:.1f} ms")
        if args.watch <= 0:
            return
        time.sleep(max(0.0, 1.0 / args.watch - (time.perf_counter() - t0)))


if __name__ == "__main__":
    main()
