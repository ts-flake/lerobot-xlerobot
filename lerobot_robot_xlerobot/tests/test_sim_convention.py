"""Tests for the sim-convention processors and the arms-only robot mode (no hardware needed)."""

import math
import random

import pytest

from lerobot.processor import RobotProcessorPipeline, TransitionKey
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot_robot_xlerobot import XLeRobotX, XLeRobotXConfig
from lerobot_robot_xlerobot.sim_convention_processor import (
    ARM_JOINTS,
    URDF_LIMITS_RAD,
    DryRunAction,
    SimConventionAction,
    SimConventionObservation,
    sim_home_real,
)

ARM_KEYS = [f"{side}_arm_{j}.pos" for side in ("left", "right") for j in ARM_JOINTS + ["gripper"]]


def obs_pipeline(*steps):
    return RobotProcessorPipeline(steps=list(steps), to_transition=observation_to_transition,
                                  to_output=transition_to_observation)


def act_pipeline(*steps):
    return RobotProcessorPipeline(steps=list(steps), to_transition=robot_action_observation_to_transition,
                                  to_output=transition_to_robot_action)


def random_real_pose(rng):
    pose = {}
    for k in ARM_KEYS:
        j = k.split("_arm_")[1].removesuffix(".pos")
        if j == "gripper":
            pose[k] = rng.uniform(0, 100)
        else:
            lo, hi = URDF_LIMITS_RAD[j]
            pose[k] = math.degrees(rng.uniform(lo, hi))
    return pose


@pytest.mark.parametrize("offsets", [{}, {"left_arm_shoulder_lift": 3.0, "right_arm_wrist_roll": -2.0}])
def test_round_trip_is_identity(offsets):
    rng = random.Random(0)
    to_sim = obs_pipeline(SimConventionObservation(offset_deg=offsets))
    to_real = act_pipeline(SimConventionAction(offset_deg=offsets, clip_to_urdf=False))
    for _ in range(100):
        real = random_real_pose(rng)
        sim = to_sim(dict(real))
        back = to_real((dict(sim), dict(real)))
        for k in ARM_KEYS:
            assert back[k] == pytest.approx(real[k], abs=1e-6)


def test_known_points_and_passthrough():
    obs = {k: 0.0 for k in ARM_KEYS}
    obs["left_arm_elbow_flex.pos"] = 90.0
    obs["left_arm_gripper.pos"] = 0.0      # real: open
    obs["right_arm_gripper.pos"] = 100.0   # real: closed
    obs["head"] = "image-placeholder"
    sim = obs_pipeline(SimConventionObservation())(obs)
    assert sim["left_arm_elbow_flex.pos"] == pytest.approx(math.pi / 2)
    assert sim["left_arm_gripper.pos"] == pytest.approx(1.0)   # sim: open
    assert sim["right_arm_gripper.pos"] == pytest.approx(0.0)  # sim: closed
    assert sim["head"] == "image-placeholder"


def test_action_clipping():
    act = {k: 0.0 for k in ARM_KEYS}
    act["left_arm_wrist_yaw.pos"] = 5.0     # rad, beyond +1.22173
    act["right_arm_gripper.pos"] = 1.3      # beyond open
    act["left_arm_gripper.pos"] = -0.2      # beyond closed
    real = act_pipeline(SimConventionAction())((act, {}))
    assert real["left_arm_wrist_yaw.pos"] == pytest.approx(math.degrees(1.22173))
    assert real["right_arm_gripper.pos"] == pytest.approx(0.0)
    assert real["left_arm_gripper.pos"] == pytest.approx(100.0)


def test_sim_home_matches_robotwin_homestate():
    home = sim_home_real()
    assert home["left_arm_shoulder_lift"] == pytest.approx(math.degrees(-1.4))
    assert home["right_arm_elbow_flex"] == pytest.approx(math.degrees(1.4))
    assert home["left_arm_gripper"] == pytest.approx(0.0)  # sim open (1.0) == real 0
    assert len(home) == 14


def test_dry_run_holds_measured_pose():
    measured = {k: float(i) for i, k in enumerate(ARM_KEYS)}
    wanted = {k: v + 10.0 for k, v in measured.items()}
    out = act_pipeline(DryRunAction())((wanted, dict(measured)))
    assert out == measured


# ---------------------------------------------------------------- arms-only robot mode
class FakeBus:
    def __init__(self, motors):
        self.motors = motors
        self.is_connected = True
        self.writes = []

    def sync_write(self, register, values, **kwargs):
        self.writes.append((register, dict(values)))


def make_robot(arms_only, tmp_path):
    cfg = XLeRobotXConfig(id="test", calibration_dir=tmp_path, cameras={}, arms_only=arms_only)
    robot = XLeRobotX(cfg)
    robot.bus1 = FakeBus(robot.bus1.motors)
    robot.bus2 = FakeBus(robot.bus2.motors)
    return robot


def test_arms_only_features_and_hold_flag(tmp_path):
    robot = make_robot(True, tmp_path)
    assert list(robot.action_features) == ARM_KEYS
    assert [k for k in robot.observation_features] == ARM_KEYS
    assert robot.supports_position_hold is True
    full = make_robot(False, tmp_path)
    assert full.supports_position_hold is False
    assert "x.vel" in full.action_features and "head_pitch.pos" in full.action_features


def test_arms_only_send_action_writes_and_returns_arms_only(tmp_path):
    robot = make_robot(True, tmp_path)
    action = {k: 1.0 for k in ARM_KEYS} | {"head_pitch.pos": 30.0, "x.vel": 0.2}
    applied = robot.send_action(action)
    assert set(applied) == set(robot.action_features)
    written = [r for bus in (robot.bus1, robot.bus2) for r, _ in bus.writes]
    assert "Goal_Velocity" not in written
    head_written = any("head_pitch" in v for bus in (robot.bus1,) for _, v in bus.writes)
    assert not head_written


def test_lerobot_position_hold_contract(tmp_path):
    """The rollout RobotWrapper accepts the arms-only robot and validates its applied actions."""
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot

    robot = make_robot(True, tmp_path)
    wrapper = ThreadSafeRobot(robot)
    wrapper.configure_position_hold()
    applied = wrapper.send_action({k: 2.0 for k in ARM_KEYS})
    assert set(applied) == set(ARM_KEYS)
