# XLeRobot arm server protocol

Version: **0.5** (`lerobot_xlerobot_server.PROTOCOL_VERSION`)

Versioning: `MAJOR.MINOR`. MINOR grows when requests or reply fields are added; old
clients keep working. MAJOR changes break compatibility. A client calls `ping` after
connecting and refuses a server whose MAJOR differs from its own.

## Transport

- ZMQ REQ/REP, one TCP port (default `tcp://0.0.0.0:5560`).
- A request is one frame holding a JSON object (UTF-8).
- A reply is a **multipart** message: frame 0 is the JSON header; frames 1..N are
  binary attachments (only `get_obs` has any, one JPEG per image, in the order of
  `images` in the header). Read replies with `recv_multipart` (since 0.3).
- The server handles one request at a time. No handler blocks for long: motion
  requests compute their joint waypoints, hand them to that arm's motion thread and
  return at once (the arms are on separate buses and move concurrently).
- Clients: use one REQ socket per thread, set a receive timeout, and replace the
  socket after a timeout (a REQ socket that sent without receiving cannot send again).

## Envelope

Request:

```json
{"cmd": "<name>", ...arguments}
```

Every reply carries:

| field | type | meaning |
|---|---|---|
| `ok` | bool | request succeeded |
| `protocol` | str | server protocol version |
| `t` | float | server wall-clock time (s, Unix epoch) when the reply was built |

On failure (`ok: false`):

| field | meaning |
|---|---|
| `code` | `bad_request` (not a JSON object or no `cmd`), `unknown_cmd`, `bad_arg`, `internal` (hardware or server exception) |
| `error` | human-readable message |
| `traceback` | only for `internal` |

## Conventions

- **Arms**: `"left"`, `"right"`; requests that take `arm` also accept `"both"` (default).
- **Joints**: degrees, in motor order shoulder_pan, shoulder_lift, elbow_flex,
  wrist_flex, wrist_yaw, wrist_roll (= URDF `joint1`..`joint6`). Calibration zero is
  the URDF zero, same sign.
- **Gripper**: LeRobot normalised 0-100 and, since 0.5, the fingertip opening in
  metres, mapped linearly: width = `width_at_0_m` + value / 100 * (`width_at_100_m` -
  `width_at_0_m`) (server config `gripper`). On XLeRobot X **0 = open (0.081 m), 100 =
  closed (0 m)**. Clients that work in metres (`width_m`, `gripper_m`) do not depend
  on this direction.
- **pose7**: `[x, y, z, qx, qy, qz, qw]`; metres; quaternion in xyzw (scipy) order,
  `qw >= 0`. Pose of the TCP frame in **that arm's own base frame** (URDF root of
  `so101_x.urdf`): +X forward, +Z up. Each arm has its own base frame; the server
  does not relate the two arms to each other or to the body.
- **TCP frame**: `tcp_frame` in `ping` (default `gripper_tip`, the fingertip centre).
  The tool axis is **+X** of the TCP frame. At the default home pose (URDF zero +
  shoulder_lift -80, elbow_flex +80) the TCP is at about (258.7, 0.3, 167.2) mm with
  the tool axis horizontal and pointing forward.

## Requests

### `ping`

No arguments. Reply:

| field | type | meaning |
|---|---|---|
| `server` | str | `"lerobot_xlerobot_server"` |
| `uptime_s` | float | seconds since the server object was created |
| `tcp_frame` | str | URDF frame used for FK/IK |
| `arms` | list[str] | `["left", "right"]` |
| `cmds` | list[str] | requests this server implements |
| `motion` | object | `hz`, `max_step_m`, `max_rot_deg` (since 0.4) |

### `get_state`

| argument | default | meaning |
|---|---|---|
| `arm` | `"both"` | which arms to read |
| `head` | `true` | also read the head joints |

Reply:

```json
{
  "arms": {
    "left":  {"q_deg": [6 floats], "gripper": 0-100, "ee_pose": [pose7], "torque": true},
    "right": {"...": "..."}
  },
  "head_q_deg": [yaw, pitch]
}
```

`q_deg` and `gripper` are measured (`Present_Position`); `ee_pose` is FK of the
measured joints, so it includes tracking error and gravity sag. It is not the
commanded target. `torque` (since 0.2) is the arm's torque state as last set by the
server.

Since 0.4, each arm also carries:

| field | meaning |
|---|---|
| `target_ee` | pose7 of the last accepted target (what `set_ee_target` deltas should be applied to) |
| `q_cmd_deg` | joints of the last waypoint written to the motors |
| `moving` | waypoints still queued |
| `error` | present only if the last waypoint write failed |
| `gripper_m` | measured gripper opening in metres (since 0.5) |
| `gripper_cmd` | last commanded 0-100 value, `null` if none since startup / torque on (since 0.5) |

### `get_obs` (since 0.3)

| argument | default | meaning |
|---|---|---|
| `cams` | `"all"` | `"all"` or a list of camera names (as configured on the server) |
| `quality` | `85` | JPEG quality, int 1..100 |
| `state` | `true` | also include the `get_state` reply for both arms + head |

Header:

```json
{
  "images": [
    {"name": "head", "shape": [480, 640, 3], "age_ms": 12.3, "bytes": 51234},
    {"name": "left_wrist", "...": "..."}
  ],
  "state": {"arms": {"...": "as get_state"}, "head_q_deg": [yaw, pitch]}
}
```

followed by one JPEG frame per entry of `images`, same order. Images are encoded from
RGB, so a standard JPEG decoder gives the true colours. `age_ms` is the time from frame
capture to encoding (each camera's background thread keeps its latest frame, so this is
at most about one frame period plus encode time). Images are captured first, then the
state. With no cameras configured, `images` is empty.

## Motion (since 0.4)

Motion requests are **non-blocking**: the reply says whether the command was accepted
and how long it will take; poll `get_state` (`moving`) to see it finish. A new command
for an arm replaces whatever is still queued for that arm.

A well-formed command the server declines is **not** an error: the reply has `ok: true`,
`accepted: false` and a `reason`; the arm does not move. Reasons:

| reason | meaning |
|---|---|
| `torque_off` | the arm's torque is off |
| `step_too_large` | target too far from the current target (`step_m`, `step_deg` in the reply) |
| `ik_not_converged` | no joint solution within tolerance (1 mm, 1 deg) |
| `urdf_joint_limit` | solution outside the URDF joint limits |
| `calibrated_range:<motors>` | solution outside the calibrated range of motion of those motors |

There is no watchdog: every command is a bounded trajectory; if the client goes away
the arm finishes what is queued and holds there (position control).

### `set_ee_target`

| argument | meaning |
|---|---|
| `arm` | `"left"` or `"right"` (not `"both"`) |
| `pose7` | absolute TCP target in that arm's base frame |

Checks, in order: torque on; translation <= `max_step_m` (default 0.05 m) and rotation
<= `max_rot_deg` (default 15 deg) from the arm's current `target_ee`; IK (seeded with the
last commanded joints, iterated to 1 mm / 1 deg, orientation weighted like position);
joint limits. Then the joints are linearly interpolated from the last commanded joints
to the solution, with `ceil(max|dq| / 3 deg) - 1` intermediate points (at most 5), sent
at `hz` (30).

Accepted reply: `accepted: true`, `q_goal_deg`, `n_waypoints`, `duration_s`,
`pos_err_mm`, `ori_err_deg`, `iters`. A rejected reply carries the IK fields when IK ran.

### `move_joints`

`arm` (`"left"`/`"right"`), `q_deg` (6), `duration_s` (default 2.0, 0.2..30). Min-jerk in
joint space from the last commanded joints. `target_ee` becomes FK(`q_deg`).

### `go_home`

`arm` (default `"both"`), `duration_s` (default `home.duration_s`). Min-jerk to the
home pose (URDF zero + `home.offset`); arms only, the head stays. Reply:
`{"arms": {"left": {accepted, ...}, ...}}`.

### `hold`

`arm` (default `"both"`). Drops queued motion; the measured joints become the goal and
`target_ee` becomes FK(measured).

### `set_gripper` (since 0.5)

| argument | meaning |
|---|---|
| `arm` | `"left"` or `"right"` |
| `value` | 0..100 (XLeRobot X: 0 = open, 100 = closed), or |
| `width_m` | opening in metres (converted with the linear map); give exactly one of the two |

A single goal write; the servo moves on its own (about 0.5-1 s for a full stroke). It is
not queued behind arm waypoints, so arm commands and gripper commands never cancel each
other. Poll `gripper` / `gripper_m` in `get_state`: when closing on an object the
reading stops at the object's width. Reply: `accepted`, `value`, `width_m`; or
`accepted: false` with `torque_off` / `calibrated_range:<motor>`. Out-of-range targets
are `bad_arg`.

### `set_torque` (since 0.2, debugging)

| argument | default | meaning |
|---|---|---|
| `arm` | `"both"` | which arms (all 7 motors of each, gripper included) |
| `enable` | required | `true` / `false` |

Reply: `{"torque": {"left": bool, "right": bool}}`.

Turning torque **off** drops queued motion and lets the arm fall under gravity; it
exists so the arm can be posed by hand to check FK. Turning it **on** first sets each
motor's goal to its present position, so the arm does not jump back to the old goal;
the commanded joints and `target_ee` are reset to the measured pose. Motion commands
are refused (`torque_off`) for an arm whose torque is off.

## Server lifecycle

1. Connect both buses with the existing calibration file (never prompts; refuses to
   start if there is no calibration).
2. Record the **start pose** (present position of both arms incl. grippers and the
   head), assumed to be a safe resting pose.
3. Connect the cameras. If one fails, the server exits before moving anything.
4. Move arms and head to the **home pose** = URDF zero + `home.offset` (min-jerk,
   `home.duration_s`). The home pose is also the default rollout begin pose.
5. Start one motion thread per arm; `target_ee` := FK(measured joints). Serve.
6. On SIGINT/SIGTERM (or a server error): stop serving and the motion threads; re-enable torque on any arm
   where it is off; move back to the start pose (min-jerk, `on_exit.duration_s`);
   disconnect. A further SIGINT during that move aborts it and the arms hold where
   they are. `on_exit.return_to_start: false` skips the move. After disconnecting,
   `on_exit.torque_off` (default `false`) decides whether the motors keep holding.
   Cameras are released.

## Planned

| version | adds |
|---|---|

## Changes

| version | change |
|---|---|
| 0.1 | `ping`, `get_state` |
| 0.2 | `set_torque`; `torque` in `get_state`; return to the start pose on exit |
| 0.3 | `get_obs`; replies are multipart (header + attachments) |
| 0.4 | `set_ee_target`, `move_joints`, `go_home`, `hold`; `target_ee`, `q_cmd_deg`, `moving` in `get_state`; `motion` in `ping` |
| 0.5 | `set_gripper`; `gripper_m`, `gripper_cmd` in `get_state` |
