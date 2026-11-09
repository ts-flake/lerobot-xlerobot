"""Convert between XLeRobotX motor units and the RoboTwin (sim) convention used for training.

Real robot (``use_degrees=True``)            RoboTwin sim (soarmX embodiment)
- arm joints: degrees, calibration zero       - arm joints: radians, URDF zero
  == URDF zero, same sign                       (``so101_x.urdf`` == ``soarmX.urdf`` kinematics)
- gripper: 0..100, 0 = open, 100 = closed     - gripper: 0..1, 0 = closed, 1 = open

The policy sees and emits the sim convention; the robot keeps its own units. Insert
``SimConventionObservation`` into the robot-observation pipeline (after any feature select)
and ``SimConventionAction`` into the robot-action pipeline, so that what leaves the robot
computer and what comes back is in sim units, and what reaches ``send_action`` is in robot units.

Mapping (per arm joint ``m``):  q_sim = deg2rad(q_real + offset_deg[m]);  inverse below.
Gripper (linear, as in the xlerobot server config):  g_sim = 1 - g_real / 100.
"""

from dataclasses import dataclass, field
import logging
import math

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ObservationProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    TransitionKey,
)

logger = logging.getLogger(__name__)

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]

# URDF joint1..joint6 limits (rad) of so101_x.urdf / soarmX.urdf, in ARM_JOINTS order.
URDF_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74115, 1.74951),
    "elbow_flex": (-1.71263, 1.54765),
    "wrist_flex": (-1.48448, 1.6222),
    "wrist_yaw": (-1.22173, 1.22173),
    "wrist_roll": (-2.6180, 2.6180),
}

# RoboTwin soarmX `homestate` (rad) with the gripper open; every training episode starts here.
SIM_HOME_RAD = {"shoulder_pan": 0.0, "shoulder_lift": -1.4, "elbow_flex": 1.4,
                "wrist_flex": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0}
SIM_HOME_GRIPPER = 1.0


def _joint_of(key: str) -> str | None:
    """'left_arm_elbow_flex.pos' -> 'elbow_flex'; 'left_arm_gripper.pos' -> 'gripper'; else None."""
    if not key.endswith(".pos"):
        return None
    name = key.removesuffix(".pos")
    for side in ("left_arm_", "right_arm_"):
        if name.startswith(side):
            return name.removeprefix(side)
    return None


def gripper_real_to_sim(g_real: float) -> float:
    return 1.0 - float(g_real) / 100.0


def gripper_sim_to_real(g_sim: float) -> float:
    return 100.0 * (1.0 - float(g_sim))


def sim_home_real(offset_deg: dict[str, float] | None = None) -> dict[str, float]:
    """The sim start pose in robot units, keyed by motor name (both arms incl. grippers)."""
    offset_deg = offset_deg or {}
    out = {}
    for side in ("left_arm_", "right_arm_"):
        for j, q in SIM_HOME_RAD.items():
            out[side + j] = math.degrees(q) - offset_deg.get(side + j, 0.0)
        out[side + "gripper"] = gripper_sim_to_real(SIM_HOME_GRIPPER)
    return out


@ProcessorStepRegistry.register("xlerobot_sim_convention_observation")
@dataclass
class SimConventionObservation(ObservationProcessorStep):
    """Robot observation (deg, gripper 0=open..100=closed) -> sim convention (rad, 0=closed..1=open).

    Only ``left_arm_*.pos`` / ``right_arm_*.pos`` entries are converted; images and other keys
    pass through. ``offset_deg`` maps motor names (e.g. ``left_arm_shoulder_lift``) to a
    calibration correction in degrees (default 0: calibration zero == URDF zero).
    """

    offset_deg: dict[str, float] = field(default_factory=dict)

    def observation(self, observation: RobotObservation) -> RobotObservation:
        for key, value in list(observation.items()):
            joint = _joint_of(key)
            if joint is None:
                continue
            if joint == "gripper":
                observation[key] = gripper_real_to_sim(value)
            elif joint in URDF_LIMITS_RAD:
                motor = key.removesuffix(".pos")
                observation[key] = math.radians(float(value) + self.offset_deg.get(motor, 0.0))
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features  # same names and types; only units change


@ProcessorStepRegistry.register("xlerobot_sim_convention_action")
@dataclass
class SimConventionAction(RobotActionProcessorStep):
    """Policy action in sim convention (rad, 0=closed..1=open) -> robot units (deg, 0=open..100=closed).

    Joints are clipped to the URDF limits (in rad, before conversion) and grippers to [0, 1];
    the robot driver applies its own calibrated-range clipping afterwards.
    """

    offset_deg: dict[str, float] = field(default_factory=dict)
    clip_to_urdf: bool = True

    def action(self, action: RobotAction) -> RobotAction:
        for key, value in list(action.items()):
            joint = _joint_of(key)
            if joint is None:
                continue
            value = float(value)
            if joint == "gripper":
                action[key] = gripper_sim_to_real(min(1.0, max(0.0, value)))
            elif joint in URDF_LIMITS_RAD:
                if self.clip_to_urdf:
                    lo, hi = URDF_LIMITS_RAD[joint]
                    value = min(hi, max(lo, value))
                motor = key.removesuffix(".pos")
                action[key] = math.degrees(value) - self.offset_deg.get(motor, 0.0)
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("xlerobot_dry_run_action")
@dataclass
class DryRunAction(RobotActionProcessorStep):
    """Log the would-be robot action and replace it by the measured pose, so the robot holds still.

    Place it last in the robot-action pipeline (after ``SimConventionAction``). It reads the raw
    robot observation carried in the transition, so every action key must be measured there.
    """

    log_every: int = 1
    _count: int = field(default=0, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION) or {}
        missing = [k for k in action if k not in observation]
        if missing:
            raise ValueError(f"DryRunAction: no measured value for {missing}")
        if self._count % max(self.log_every, 1) == 0:
            delta = {k: round(float(action[k]) - float(observation[k]), 2) for k in action}
            logger.info("[dry-run] would send %s | delta vs measured %s",
                        {k: round(float(v), 2) for k, v in action.items()}, delta)
        self._count += 1
        return {k: float(observation[k]) for k in action}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
