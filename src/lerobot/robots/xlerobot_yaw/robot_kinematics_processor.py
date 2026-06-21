from dataclasses import dataclass, field
from typing import Any
from pprint import pformat
import logging

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.model.rr_kinematics import RRKinematics
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.rotation import Rotation
from lerobot.robots.xlerobot_yaw import XLeRobotYaw

@ProcessorStepRegistry.register("inverse_kinematics_delta_to_joints")
@dataclass
class InverseKinematicsDeltaToJoints(RobotActionProcessorStep):
    """
    Computes the target arm joint positions from the delta commands using placo inverse kinematics (IK).

    Unlike AnalyticalInverseKinematicsDeltaToJoints which uses a 2-link analytical IK for (x, z) only,
    this step uses placo's full IK solver for each arm. A target end-effector pose is maintained
    per arm and updated each step by applying the delta commands; IK is then solved using the current
    joint state as the initial guess.

    Head and gripper are handled with direct delta control, identical to the analytical variant.

    Attributes:
        kinematics_left: Placo-based kinematics model for the left arm.
        kinematics_right: Placo-based kinematics model for the right arm.
        robot: The robot instance (used for unit conversions).
        motor_names: All motor names; arm joints must appear in the same order as the
            corresponding kinematics.joint_names.
    """
    kinematics_left: RobotKinematics
    kinematics_right: RobotKinematics
    motor_names: list[str]
    resync_threshold_deg: float = 20.0  # L∞ over arm joints; <=0 disables

    def __post_init__(self):
        self._target_ee_left: np.ndarray | None = None
        self._target_ee_right: np.ndarray | None = None
        self._q_left: np.ndarray | None = None
        self._q_right: np.ndarray | None = None

    def reset(self):
        self._target_ee_left = None
        self._target_ee_right = None
        self._q_left = None
        self._q_right = None

    def _arm_motor_names(self, side: str) -> list[str]:
        """Arm motor names for *side* ('left'/'right'), excluding gripper, preserving order."""
        prefix = f"{side}_arm_"
        gripper = f"{side}_arm_gripper"
        return [n for n in self.motor_names if n.startswith(prefix) and n != gripper]

    def _apply_arm_ik(
        self,
        side: str,
        q_dict: dict,
        kinematics: RobotKinematics,
        target_ee: np.ndarray | None,
        prev_q: np.ndarray | None,
        dx: float | None,
        dy: float | None,
        dz: float | None,
        droll: float | None,
        dpitch: float | None,
        dyaw: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply position/orientation deltas to *target_ee*, solve IK, update q_dict.
        Returns (target_ee, q_solution_deg).
        Uses prev_q (last IK solution) as the warm-start for placo so the solver
        stays on the same IK branch across frames, avoiding the shakiness caused
        by re-seeding from the actual (lagging) motor observation each cycle.
        """
        arm_motors = self._arm_motor_names(side)
        q_obs = np.array([q_dict[n] for n in arm_motors], dtype=float)

        # Resync warm-start AND target_ee to reality if the previous command
        # diverged from observation beyond threshold (e.g., arm blocked / lagging).
        # Without this, target_ee keeps integrating deltas while the robot is stuck,
        # causing a lurch once the obstruction clears.
        if (
            self.resync_threshold_deg > 0
            and prev_q is not None
            and np.max(np.abs(prev_q - q_obs)) > self.resync_threshold_deg
        ):
            max_delta = float(np.max(np.abs(prev_q - q_obs)))
            logging.debug(
                f"[IK resync] {side} arm: max |prev_q - q_obs| = {max_delta:.2f}° "
                f"> {self.resync_threshold_deg}°; snapping prev_q and target_ee to observation."
            )
            prev_q = q_obs.copy()
            target_ee = None  # forces FK re-init below

        # Warm-start from previous IK solution; fall back to observation on first call
        q_seed = prev_q if prev_q is not None else q_obs

        # Initialize the EE target from FK on first call or after reset
        if target_ee is None:
            target_ee = kinematics.forward_kinematics(q_obs).copy()

        # Apply position deltas
        if dx is not None:
            target_ee[0, 3] += dx
        if dy is not None:
            target_ee[1, 3] += dy
        if dz is not None:
            target_ee[2, 3] += dz

        # Apply orientation deltas in the EE local frame (right-multiply)
        droll = droll or 0.0
        dpitch = dpitch or 0.0
        dyaw = dyaw or 0.0
        if droll or dpitch or dyaw:
            dr = Rotation.from_rotvec(np.deg2rad([droll, dpitch, dyaw])).as_matrix()
            target_ee[:3, :3] = target_ee[:3, :3] @ dr

        q_target_deg = kinematics.inverse_kinematics(q_seed, target_ee)
        for i, n in enumerate(arm_motors):
            q_dict[n] = float(q_target_deg[i])

        return target_ee, q_target_deg

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is required for computing robot kinematics")

        q_dict = {
            k.removesuffix(".pos"): float(v)
            for k, v in observation.items()
            if isinstance(k, str)
            and k.endswith(".pos")
            and k.removesuffix(".pos") in self.motor_names
        }

        if not q_dict:
            raise ValueError("Joints observation is required for computing robot kinematics")

        # Head — direct delta control (same as analytical variant)
        dyaw_head = action.pop("head_delta.yaw")
        if dyaw_head is not None and "head_yaw" in q_dict:
            q_dict["head_yaw"] -= dyaw_head

        dpitch_head = action.pop("head_delta.pitch")
        if dpitch_head is not None and "head_pitch" in q_dict:
            q_dict["head_pitch"] += dpitch_head

        # Pop all arm EE delta commands
        dx_left = action.pop("left_arm_ee_delta.x")
        dy_left = action.pop("left_arm_ee_delta.y")
        dz_left = action.pop("left_arm_ee_delta.z")
        droll_left = action.pop("left_arm_ee_delta.roll")
        dpitch_left = action.pop("left_arm_ee_delta.pitch")
        dyaw_left = action.pop("left_arm_ee_delta.yaw")

        dx_right = action.pop("right_arm_ee_delta.x")
        dy_right = action.pop("right_arm_ee_delta.y")
        dz_right = action.pop("right_arm_ee_delta.z")
        droll_right = action.pop("right_arm_ee_delta.roll")
        dpitch_right = action.pop("right_arm_ee_delta.pitch")
        dyaw_right = action.pop("right_arm_ee_delta.yaw")

        # Gripper — direct delta control
        gripper_left = action.pop("left_arm_gripper.pos")
        gripper_right = action.pop("right_arm_gripper.pos")

        # Solve IK for each arm and update q_dict
        self._target_ee_left, self._q_left = self._apply_arm_ik(
            "left", q_dict, self.kinematics_left, self._target_ee_left, self._q_left,
            dx_left, dy_left, dz_left, droll_left, dpitch_left, dyaw_left,
        )
        self._target_ee_right, self._q_right = self._apply_arm_ik(
            "right", q_dict, self.kinematics_right, self._target_ee_right, self._q_right,
            dx_right, dy_right, dz_right, droll_right, dpitch_right, dyaw_right,
        )

        # Gripper — additive delta on normalized value
        if gripper_left is not None and "left_arm_gripper" in q_dict:
            q_dict["left_arm_gripper"] += gripper_left
        if gripper_right is not None and "right_arm_gripper" in q_dict:
            q_dict["right_arm_gripper"] += gripper_right

        action.update({f"{k}.pos": v for k, v in q_dict.items()})
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        no_left_wrist_yaw = "left_arm_ee_delta.yaw"  not in features[PipelineFeatureType.ACTION]
        no_right_wrist_yaw = "right_arm_ee_delta.yaw" not in features[PipelineFeatureType.ACTION]

        for feat in [
            "left_arm_ee_delta.x", "left_arm_ee_delta.y", "left_arm_ee_delta.z",
            "left_arm_ee_delta.roll", "left_arm_ee_delta.pitch", "left_arm_ee_delta.yaw",
            "left_arm_gripper.pos",
            "right_arm_ee_delta.x", "right_arm_ee_delta.y", "right_arm_ee_delta.z",
            "right_arm_ee_delta.roll", "right_arm_ee_delta.pitch", "right_arm_ee_delta.yaw",
            "right_arm_gripper.pos",
            "head_delta.yaw", "head_delta.pitch",
        ]:
            features[PipelineFeatureType.ACTION].pop(feat, None)

        for feat in [
            "left_arm_shoulder_pan", "left_arm_shoulder_lift", "left_arm_elbow_flex",
            "left_arm_wrist_flex", "left_arm_wrist_yaw", "left_arm_wrist_roll", "left_arm_gripper",
            "right_arm_shoulder_pan", "right_arm_shoulder_lift", "right_arm_elbow_flex",
            "right_arm_wrist_flex", "right_arm_wrist_yaw", "right_arm_wrist_roll", "right_arm_gripper",
            "head_yaw", "head_pitch",
        ]:
            features[PipelineFeatureType.ACTION][f"{feat}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        if no_left_wrist_yaw:
            features[PipelineFeatureType.ACTION].pop("left_arm_wrist_yaw.pos", None)
        if no_right_wrist_yaw:
            features[PipelineFeatureType.ACTION].pop("right_arm_wrist_yaw.pos", None)

        return features


@ProcessorStepRegistry.register("analytical_inverse_kinematics_delta_to_joints")
@dataclass
class AnalyticalInverseKinematicsDeltaToJoints(RobotActionProcessorStep):
    """
    Computes the target arm joint positions from the delta commands using analytical inverse kinematics (IK).
    The analytical IK is only partial, i.e., for a EE delta commmand (dx, dy, dz, droll, dpitch, dyaw),
    the kinematics model solves for the 'shoulder_lift' and 'elbow_flex' from (dx, dz),
    and uses (dx, droll, dpitch, dyaw) to directly compute 'shoulder_pan', 'wrist_roll', 'wrist_flex', and 'wrist_yaw'.

    Attributes:
        kinematics_left: The kinematic model for the left arm.
        kinematics_right: The kinematic model for the right arm.
        robot: The robot instance.
        motor_names: The names of the motors to control.
    """
    kinematics_left: RRKinematics
    kinematics_right: RRKinematics
    motor_names: list[str]

    def __post_init__(self):
        self._target_pitch_left = None
        self._target_pitch_right = None
        self._target_xz_left = None
        self._target_xz_right = None

    def reset(self):
        self._target_pitch_left = None
        self._target_pitch_right = None
        self._target_xz_left = None
        self._target_xz_right = None

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is required for computing robot kinematics")

        # q_dict values are in degrees (MotorNormMode.DEGREES)
        q_dict = {
            k.removesuffix(".pos"): float(v)
            for k, v in observation.items()
            if isinstance(k, str)
            and k.endswith(".pos")
            and k.removesuffix(".pos") in self.motor_names
        }

        if not q_dict:
            raise ValueError("Joints observation is required for computing robot kinematics")
        
        # Note that for direct degree control, the +/- sign of delta change is chosen
        # according to the motor rotation direction, by default, assuming
        # clockwise rotation is +.
    
        # Head yaw / pitch (direct degree control)
        dyaw_head = action.pop("head_delta.yaw")
        if dyaw_head is not None and q_dict.get("head_yaw") is not None:
            q_dict["head_yaw"] -= dyaw_head

        dpitch_head = action.pop("head_delta.pitch")
        if dpitch_head is not None and q_dict.get("head_pitch") is not None:
            q_dict["head_pitch"] += dpitch_head

        # Wrist roll (direct degree control)
        droll_left = action.pop("left_arm_ee_delta.roll")
        droll_right = action.pop("right_arm_ee_delta.roll")
        if droll_left is not None and q_dict.get("left_arm_wrist_roll") is not None:
            q_dict["left_arm_wrist_roll"] -= droll_left
        if droll_right is not None and q_dict.get("right_arm_wrist_roll") is not None:
            q_dict["right_arm_wrist_roll"] -= droll_right

        # Wrist yaw (direct degree control)
        dyaw_left = action.pop("left_arm_ee_delta.yaw")
        dyaw_right = action.pop("right_arm_ee_delta.yaw")
        if dyaw_left is not None and q_dict.get("left_arm_wrist_yaw") is not None:
            q_dict["left_arm_wrist_yaw"] -= dyaw_left
        if dyaw_right is not None and q_dict.get("right_arm_wrist_yaw") is not None:
            q_dict["right_arm_wrist_yaw"] -= dyaw_right

        # Shoulder pan (direct degree control via y delta)
        dy_left = action.pop("left_arm_ee_delta.y")
        dy_right = action.pop("right_arm_ee_delta.y")
        if dy_left is not None and q_dict.get("left_arm_shoulder_pan") is not None:
            q_dict["left_arm_shoulder_pan"] -= dy_left
        if dy_right is not None and q_dict.get("right_arm_shoulder_pan") is not None:
            q_dict["right_arm_shoulder_pan"] -= dy_right

        # Initialize target_pitch and target_xz from current joint state (degrees)
        if self._target_pitch_left is None:
            self._target_pitch_left = 0.0
        if self._target_pitch_right is None:
            self._target_pitch_right = 0.0
        if (
            self._target_xz_left is None
            and q_dict.get("left_arm_shoulder_lift") is not None
            and q_dict.get("left_arm_elbow_flex") is not None
        ):
            self._target_xz_left = self.kinematics_left.forward_kinematics(
                q_dict["left_arm_shoulder_lift"], q_dict["left_arm_elbow_flex"]
            )
        if (
            self._target_xz_right is None
            and q_dict.get("right_arm_shoulder_lift") is not None
            and q_dict.get("right_arm_elbow_flex") is not None
        ):
            self._target_xz_right = self.kinematics_right.forward_kinematics(
                q_dict["right_arm_shoulder_lift"], q_dict["right_arm_elbow_flex"]
            )

        # Accumulated wrist pitch
        dpitch_left = action.pop("left_arm_ee_delta.pitch")
        dpitch_right = action.pop("right_arm_ee_delta.pitch")
        if dpitch_left is not None:
            self._target_pitch_left -= dpitch_left
        if dpitch_right is not None:
            self._target_pitch_right -= dpitch_right

        # 2-link IK for shoulder_lift and elbow_flex — left arm
        dx_left = action.pop("left_arm_ee_delta.x")
        dz_left = action.pop("left_arm_ee_delta.z")
        if (
            dx_left is not None
            and dz_left is not None
            and q_dict.get("left_arm_shoulder_lift") is not None
            and q_dict.get("left_arm_elbow_flex") is not None
        ):
            self._target_xz_left[0] += dx_left
            self._target_xz_left[1] += dz_left
            self._target_xz_left = self.kinematics_left.apply_workspace_bound(*self._target_xz_left)[:2]
            jnt2, jnt3 = self.kinematics_left.inverse_kinematics(*self._target_xz_left)
            q_dict["left_arm_shoulder_lift"] = jnt2
            q_dict["left_arm_elbow_flex"] = jnt3
            if q_dict.get("left_arm_wrist_flex") is not None:
                q_dict["left_arm_wrist_flex"] = -jnt2 - jnt3 - self._target_pitch_left

        # 2-link IK for shoulder_lift and elbow_flex — right arm
        dx_right = action.pop("right_arm_ee_delta.x")
        dz_right = action.pop("right_arm_ee_delta.z")
        if (
            dx_right is not None
            and dz_right is not None
            and q_dict.get("right_arm_shoulder_lift") is not None
            and q_dict.get("right_arm_elbow_flex") is not None
        ):
            self._target_xz_right[0] += dx_right
            self._target_xz_right[1] += dz_right
            self._target_xz_right = self.kinematics_right.apply_workspace_bound(*self._target_xz_right)[:2]
            jnt2, jnt3 = self.kinematics_right.inverse_kinematics(*self._target_xz_right)
            q_dict["right_arm_shoulder_lift"] = jnt2
            q_dict["right_arm_elbow_flex"] = jnt3
            if q_dict.get("right_arm_wrist_flex") is not None:
                q_dict["right_arm_wrist_flex"] = -jnt2 - jnt3 - self._target_pitch_right

        # Gripper (direct delta on normalized [0, 100] value)
        gripper_left = action.pop("left_arm_gripper.pos")
        gripper_right = action.pop("right_arm_gripper.pos")
        if gripper_left is not None and q_dict.get("left_arm_gripper") is not None:
            q_dict["left_arm_gripper"] += gripper_left
        if gripper_right is not None and q_dict.get("right_arm_gripper") is not None:
            q_dict["right_arm_gripper"] += gripper_right

        action.update({f"{k}.pos": v for k, v in q_dict.items()})
        return action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        no_left_wrist_yaw = 'left_arm_ee_delta.yaw' not in features[PipelineFeatureType.ACTION]
        no_right_wrist_yaw = 'right_arm_ee_delta.yaw' not in features[PipelineFeatureType.ACTION]
        
        for feat in [
            "left_arm_ee_delta.x", "left_arm_ee_delta.y", "left_arm_ee_delta.z",
            "left_arm_ee_delta.roll", "left_arm_ee_delta.pitch", "left_arm_ee_delta.yaw",
            "left_arm_gripper.pos",
            "right_arm_ee_delta.x", "right_arm_ee_delta.y", "right_arm_ee_delta.z",
            "right_arm_ee_delta.roll", "right_arm_ee_delta.pitch", "right_arm_ee_delta.yaw",
            "right_arm_gripper.pos",
            "head_delta.yaw", "head_delta.pitch",
        ]:
            features[PipelineFeatureType.ACTION].pop(feat, None)
        
        for feat in [
            "left_arm_shoulder_pan", "left_arm_shoulder_lift", "left_arm_elbow_flex",
            "left_arm_wrist_flex", "left_arm_wrist_yaw", "left_arm_wrist_roll",
            "left_arm_gripper",
            "right_arm_shoulder_pan", "right_arm_shoulder_lift", "right_arm_elbow_flex",
            "right_arm_wrist_flex", "right_arm_wrist_yaw", "right_arm_wrist_roll",
            "right_arm_gripper",
            "head_yaw", "head_pitch",
        ]:
            features[PipelineFeatureType.ACTION][f"{feat}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        if no_left_wrist_yaw:
            features[PipelineFeatureType.ACTION].pop("left_arm_wrist_yaw.pos", None)
        if no_right_wrist_yaw:
            features[PipelineFeatureType.ACTION].pop("right_arm_wrist_yaw.pos", None)
        return features


@ProcessorStepRegistry.register("base_joint_action")
@dataclass
class BaseJointAction(RobotActionProcessorStep):
    """
    Converts the generic base commands to the actual base actions (e.g., 'x.vel', 'y.vel' and 'theta.vel').

    Attributes:
        robot: The robot instance.
    """
    robot: XLeRobotYaw

    def action(self, action: RobotAction) -> RobotAction:
        base_action = action.pop("base_action")
        pressed_keys = set()
        for act in base_action:
            pressed_keys.add(self.robot.teleop_keys[act])
        action.update(self.robot._from_keyboard_to_base_action(list(pressed_keys)))
        return action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION].pop("base_action", None)
        for feat in [
            "x.vel",
            "y.vel",
            "theta.vel",
        ]:
            features[PipelineFeatureType.ACTION][feat] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("joint_clip_norm_value")
@dataclass
class JointClipNormValue(RobotActionProcessorStep):
    robot: XLeRobotYaw

    def action(self, action: RobotAction) -> RobotAction:
        for k, v in action.items():
            if isinstance(k, str) and k.endswith(".pos"):
                action[k] = self.robot._clip_norm_value(k.removesuffix(".pos"), v)
        return action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("ema_joint_action")
@dataclass
class EMAJointAction(RobotActionProcessorStep):
    """
    Applies Exponential Moving Average (EMA) to the joint actions to smooth the control signals.
    
    Attributes:
        fps: The frame rate of the robot.
        ema_alpha: The alpha value for the EMA.
        base_speed_up_time: The time to speed up the base speed.
        base_speed_down_time: The time to slow down the base speed.
    """
    fps: int = 30 # Hz
    ema_alpha: float = 0.9
    base_speed_up_time: float = 3.0
    base_speed_down_time: float = 0.5

    def __post_init__(self):
        self.reset()
    
    def reset(self):
        self._prev_action = None
        self._speed_up_cnt = {}
        self._speed_down_cnt = {}

    def action(self, action: RobotAction) -> RobotAction:
        curr_action = action.copy()
        if self._prev_action is None:
            self._prev_action = curr_action
        
        calc_alpha = lambda a0, t, hz: 1.0 - a0 ** (1 / (t * hz))

        new_action = {}
        for k, v in curr_action.items():
            if not k.endswith('.vel'):
                new_action[k] = v * self.ema_alpha + self._prev_action[k] * (1 - self.ema_alpha)
            else: # Base speed EMA
                # Speed up is triggered when the robot moves from rest
                # And continues until the number of steps is completed
                if (abs(self._prev_action[k]) < 1e-3 and abs(v) > 1e-3) or self._speed_up_cnt.get(k, 0) > 0:
                    if k not in self._speed_up_cnt or self._speed_up_cnt[k] == 0:
                        self._speed_up_cnt[k] = int(self.fps * self.base_speed_up_time)
                    self._speed_up_cnt[k] -= 1
                    ema_alpha = calc_alpha(0.01, self.base_speed_up_time, self.fps)
                # Speed down is triggered when the robot is required to stop from motion
                # And continues until the number of steps is completed
                elif (abs(self._prev_action[k]) > 1e-3 and abs(v) < 1e-3) or self._speed_down_cnt.get(k, 0) > 0:
                    if k not in self._speed_down_cnt or self._speed_down_cnt[k] == 0:
                        self._speed_down_cnt[k] = int(self.fps * self.base_speed_down_time)
                    self._speed_down_cnt[k] -= 1
                    ema_alpha = calc_alpha(0.01, self.base_speed_down_time, self.fps)
                else:
                    ema_alpha = self.ema_alpha
                new_action[k] = v * ema_alpha + self._prev_action[k] * (1 - ema_alpha)
        self._prev_action.update(new_action)
        return new_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("safe_goal_position")
@dataclass
class SafeGoalPosition(RobotActionProcessorStep):
    """
    Ensures the goal position is safe by clamping the relative target magnitude.

    Attributes:
        max_relative_target: The maximum relative target magnitude for each motor.
        motor_names: The names of the motors to control.
    """
    max_relative_target: float | dict[str, float]
    motor_names: list[str]

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is require for computing safe goal position")

        present_pos = {
            k.removesuffix(".pos"):float(v)
            for k, v in observation.items()
            if isinstance(k, str)
            and k.endswith(".pos")
            and k.removesuffix(".pos") in self.motor_names
        }

        if present_pos is None:
            raise ValueError("Joints observation is require for computing safe goal position")
        
        goal_present_pos = {
            k: (goal_pos, present_pos[k.removesuffix(".pos")])
            for k, goal_pos in action.items()
            if k.removesuffix(".pos") in self.motor_names
        }
        safe_pos = ensure_safe_goal_position(goal_present_pos, self.max_relative_target)
        action.update(safe_pos)
        return action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("round_off_action")
@dataclass
class RoundOffAction(RobotActionProcessorStep):
    """
    Round off the action values to avoid shakiness.

    Attributes:
        decimal_places: The number of decimal places to round off the action values.
        min_abs_value: The minimum absolute value below which the action value is set to 0.
        filters: A list of suffixes to filter the action keys (e.g., ".pos", ".vel").
    """
    decimal_places: int = 2
    min_abs_value: float = 0.01
    filters: list[str] = field(default_factory=lambda: [".pos", ".vel"])

    def action(self, action: RobotAction) -> RobotAction:
        for k, v in action.items():
            if isinstance(k, str) and any(k.endswith(f) for f in self.filters):
                action[k] = round(float(v), self.decimal_places)
                if abs(action[k]) < self.min_abs_value:
                    action[k] = 0.0
        return action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("log_action")
@dataclass
class LogAction(RobotActionProcessorStep):
    logger: logging.Logger

    def __post_init__(self):
        self._prev_action = None

    def reset(self):
        self._prev_action = None

    def action(self, action: RobotAction) -> RobotAction:
        is_first = False
        curr_action = action.copy()
        if self._prev_action is None:
            self._prev_action = curr_action
            is_first = True
        
        log_action = {}
        for k, v in curr_action.items():
            if v != self._prev_action[k] or is_first:
                log_action[k] = round(float(v), 3)
        self._prev_action = curr_action

        if log_action:
            self.logger.info(f"\n\033[3;4mTarget action:\033[0m\n{pformat(log_action, indent=4)}")
        return action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features