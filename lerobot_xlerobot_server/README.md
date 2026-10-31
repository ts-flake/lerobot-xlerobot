# lerobot_xlerobot_server

ZMQ server that owns the XLeRobot X motor buses and serves arm state and (later)
end-effector commands. Clients such as Show-Harness talk to it only through the
protocol in [PROTOCOL.md](PROTOCOL.md); nothing here imports a client.

Same layout as the plugin packages in this repo: this folder is the distribution
(`pyproject.toml`), the inner `lerobot_xlerobot_server/` is the package. Install it into
the lerobot environment (pulls `lerobot_robot_xlerobot[kinematics]` and `pyzmq`):

```bash
pip install -e ./lerobot_xlerobot_server
```

## Run (on the robot computer)

```bash
cp lerobot_xlerobot_server/server.example.yaml lerobot_xlerobot_server/server.yaml   # edit ports / id / offset
xlerobot-server --config lerobot_xlerobot_server/server.yaml
#   --no-home     skip the startup move (arms stay where they are)
#   --no-return   skip the return to the start pose on exit
#   --fake        no hardware (fake arms + synthetic cameras)
#   --viz         show both arms (measured) and their targets in meshcat
xlerobot-server --list-cameras   # /dev/v4l/by-path capture devices, for the cameras: section
```

> Do not use `python -m ...` from the repo root. `python -m` puts the current directory
> first on `sys.path`; the repo root holds the distribution folder
> `lerobot_robot_xlerobot/` (no `__init__.py`), which Python then imports as an empty
> namespace package instead of the installed one ("cannot import name 'XLeRobotX' from
> 'lerobot_robot_xlerobot' (unknown location)"). The `xlerobot-server` entry point does
> not add the current directory.

On startup the server records where the arms and head are (the **start pose**, assumed
to be a safe resting pose), then moves them to the **home pose** (URDF zero +
`home.offset`) over 3 s. On Ctrl+C it moves them back to the start pose over 3 s, then
exits; press Ctrl+C again during that move to abort it (the arms then hold in place).
**Make sure the arms have room to move.**

## Talk to it

On the robot computer (or any machine where the package is installed):

```bash
xlerobot-server-cli --addr tcp://<robot-ip>:5560 ping
xlerobot-server-cli --addr tcp://<robot-ip>:5560 state
xlerobot-server-cli --addr tcp://<robot-ip>:5560 state --arm left --watch 5
xlerobot-server-cli --addr tcp://<robot-ip>:5560 torque off --arm left   # asks first: the arm falls
xlerobot-server-cli --addr tcp://<robot-ip>:5560 torque on --arm left
```

Cameras (configured under `cameras:` in server.yaml):

```bash
xlerobot-server-cli --addr tcp://<robot-ip>:5560 obs --out obs/        # one JPEG per camera
xlerobot-server-cli --addr tcp://<robot-ip>:5560 obs --n 50            # latency: median / p90 / max
xlerobot-server-cli --addr tcp://<robot-ip>:5560 obs --cams head --quality 70
```

Motion (end-effector targets in each arm's base frame, relative to the current target):

```bash
xlerobot-server-cli --addr ... move --arm left --dz 0.01                 # up 1 cm
xlerobot-server-cli --addr ... move --arm left --dyaw 5                  # yaw 5 deg about the TCP
xlerobot-server-cli --addr ... move --arm left --dx 0.02 --repeat 20 --back   # accuracy: travel / off-axis / rotation
xlerobot-server-cli --addr ... home            # both arms back to the home pose
xlerobot-server-cli --addr ... hold            # stop and hold
xlerobot-server-cli --addr ... gripper --arm left 0            # open (0 = open, 100 = closed)
xlerobot-server-cli --addr ... gripper --arm left --width 0.02 # 20 mm opening
```

First time on hardware: run `xlerobot-server --fake --viz` and try the same `move`
commands to see what the IK does, then on the robot start with 1 cm steps.

Checking FK (PRD criterion 3): `torque off` one arm while supporting it, pose it by
hand while `state --watch 2` runs, measure the fingertip centre from the arm's base
origin with a ruler, then `torque on`. Turning torque back on holds the arm where it is.

A client machine without lerobot needs only numpy, pyyaml and pyzmq: from **inside this
folder** (`cd lerobot_xlerobot_server`), `python -m lerobot_xlerobot_server.cli ...`.

`state` prints each TCP position in millimetres in that arm's base frame.

## Tests (no hardware)

```bash
cd lerobot_xlerobot_server && python -m pytest lerobot_xlerobot_server/tests -q
```

## Layout

| file (in `lerobot_xlerobot_server/`) | role |
|---|---|
| `server.py` | REP loop, request dispatch, lifecycle |
| `robot_io.py` | `XLeRobotIO` (buses behind one RLock per bus), `FakeRobotIO` |
| `kinematics.py` | per-arm placo FK, pose7 / quaternion conventions, URDF joint limits |
| `ik.py` | per-arm iterative IK (placo, converged to 1 mm / 1 deg) |
| `motion.py` | per-arm waypoint queue + thread, interpolation |
| `viz.py` | meshcat view (`--viz`) |
| `cameras.py` | `CameraHub` (lerobot OpenCV cameras, latest frame, JPEG), `FakeCameraHub` |
| `config.py` | defaults + YAML overlay |
| `client.py`, `cli.py` | reference client and command-line tool |
