# lerobot-xlerobot

XLeRobot X (XLeRobot V0.3 with an extra wrist-yaw joint per arm, 6+1 DoF arms, 3-omniwheel base, 2-DoF head) packaged as [LeRobot third-party plugins](https://huggingface.co/docs/lerobot/en/third_party_robots). Once installed, LeRobot discovers them automatically. You don't need to fork LeRobot.

| Package | Registers | Contents |
|---|---|---|
| `lerobot_robot_xlerobot` | `--robot.type=xlerobot_x` | robot, IK / feature-select processor steps, `RRKinematics`, `examples/` |
| `lerobot_teleoperator_xlerobot_gamepad` | `--teleop.type=xlerobot_x_gamepad` | SDL gamepad teleop (PS5 / Xbox via pygame) |
| `lerobot_teleoperator_xlerobot_vr` | `--teleop.type=xlerobot_x_vr` | Quest VR teleop. Includes the bundled XLeVR server and web UI. |

## Install

```bash
conda activate lerobot
pip install -e "./lerobot_robot_xlerobot[kinematics]" \
            -e ./lerobot_teleoperator_xlerobot_gamepad \
            -e ./lerobot_teleoperator_xlerobot_vr
```

Requires `lerobot>=0.4.4`. Tested with 0.4.4, 0.6.1 and 0.6.2 (main).

## Calibration

The calibration file is `~/.cache/huggingface/lerobot/calibration/robots/xlerobot_x/<robot.id>.json`.
Always pass `--robot.id`. Otherwise the id defaults to `None` and LeRobot looks for `None.json`.

```bash
lerobot-calibrate --robot.type=xlerobot_x --robot.id=xlerobot_x \
    --robot.port1=/dev/ttyACM0 --robot.port2=/dev/ttyACM1
```

- `port1`: left arm and head.
- `port2`: right arm and base.

## Teleoperate / record

The teleoperators output end-effector deltas. The scripts in `examples/` add the IK pipeline that turns those deltas into safe joint targets. They also handle zero-position homing.

```bash
cd lerobot_robot_xlerobot/examples

# teleoperate (swap the type for xlerobot_x_gamepad)
python teleoperate.py \
    --robot.type=xlerobot_x --robot.id=xlerobot_x --robot.port1=/dev/ttyACM0 --robot.port2=/dev/ttyACM1 \
    --teleop.type=xlerobot_x_vr --fps=30 --display_data=true

# record a dataset
python record.py \
    --robot.type=xlerobot_x --robot.id=xlerobot_x --robot.port1=/dev/ttyACM0 --robot.port2=/dev/ttyACM1 \
    --teleop.type=xlerobot_x_gamepad \
    --dataset.repo_id=<hf_user>/<dataset> --dataset.single_task="Grab the cube" --dataset.num_episodes=5

# policy rollout
python rollout.py --strategy.type=base --policy.path=<hf_user>/<policy> \
    --robot.type=xlerobot_x --robot.id=xlerobot_x --robot.port1=/dev/ttyACM0 --robot.port2=/dev/ttyACM1 \
    --task="Grab the cube" --duration=60
```

Each script's docstring lists the full set of options.

### Simulation (no hardware)

The scripts in `examples/tests/` drive a placo / meshcat visualisation of `examples/simulation/so101_x/so101_x.urdf` from a real teleop device:

```bash
cd lerobot_robot_xlerobot/examples/tests
python test_teleop_ps5.py --mode ik     # or analytical
python test_teleop_vr.py  --mode ik
```

## Cameras

By default no cameras are configured. Add them from the CLI, for example:

```bash
--robot.cameras='{
  head:        {type: opencv, index_or_path: /dev/video8, width: 640, height: 480, fps: 30, fourcc: MJPG},
  left_wrist:  {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, fourcc: MJPG},
  right_wrist: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'
```

## VR

The XLeVR server serves the web UI over HTTPS on port `8443` and the WebSocket on port `8442`. Open `https://<host-ip>:8443` in the Quest browser.

The server looks for `cert.pem` / `key.pem` in the current working directory and generates them there if they are missing.

## Known limitations

- `examples/rollout.py` with a feature-selected policy (for example, head control off) needs a patch to `lerobot/rollout/context.py` so that action keys come from the dataset action schema. Stock LeRobot doesn't have that patch.
- `lerobot-rollout` keeps only `.pos` features, so the base `.vel` action from a policy is ignored.

## Layout

```
lerobot_robot_xlerobot/
  lerobot_robot_xlerobot/      # robot package
  examples/                    # teleoperate / record / rollout, sim tests, URDF + meshes (LFS)
lerobot_teleoperator_xlerobot_gamepad/
lerobot_teleoperator_xlerobot_vr/
  lerobot_teleoperator_xlerobot_vr/xlevr/   # vendored XLeVR server
  lerobot_teleoperator_xlerobot_vr/web-ui/
```

Keep the distribution names in `pyproject.toml` underscore-only. LeRobot imports each plugin by its distribution name exactly as written.
