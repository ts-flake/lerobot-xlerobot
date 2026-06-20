import sys
import time
import asyncio
import logging
import threading
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..teleoperator import Teleoperator
from .config_xlerobot_yaw_vr import XLeRobotYawVRConfig

# XLeVR ships its own `xlevr` package (relative imports); put it on sys.path.
_XLEVR_PATH = Path(__file__).resolve().parent / "XLeVR"
if str(_XLEVR_PATH) not in sys.path:
    sys.path.insert(0, str(_XLEVR_PATH))
from xlevr.server import VRMonitor  # noqa: E402
from xlevr.camera_stream import select_image_frames  # noqa: E402

logger = logging.getLogger(__name__)

class XLeRobotYawVR(Teleoperator):
    config_class = XLeRobotYawVRConfig
    # Loops forward the full observation to send_feedback() so the in-VR overlay
    # can display camera frames. Defaults OFF; enabled per-instance from
    # config.stream_cameras_to_vr so teleop is never affected unless requested.
    wants_observation_feedback = False
    name = "xlerobot_yaw_vr"

    def __init__(self, config: XLeRobotYawVRConfig):
        super().__init__(config)
        self.config = config
        self.wants_observation_feedback = bool(getattr(config, "stream_cameras_to_vr", False))
        self.vr_monitor: VRMonitor | None = None
        self._vr_thread: threading.Thread | None = None

        self._prev_left_active = False
        self._prev_right_active = False
        self.safe_exit = False
        
        # Per-arm previous absolute pose (pos_rb, quat_rb, ts_s) for delta calc.
        # Drifts to the latest pose while inactive so re-activation never jumps.
        self._prev_pose: dict[str, tuple | None] = {"left": None, "right": None, "headset": None}

        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "exit_teleop": False,
            "back_robot_to_zero": False,
            "gripper_toggle_close": False
        }
    
    @property
    def is_connected(self) -> bool:
        return self.vr_monitor is not None and self.vr_monitor.is_running

    def connect(self) -> None:
        # Initialize VR monitor
        try:
            vr_monitor = VRMonitor()
            vr_thread = threading.Thread(target=lambda: asyncio.run(vr_monitor.start_monitoring()), daemon=True)
            vr_thread.start()
            self._vr_thread = vr_thread
        except Exception as e:
            logger.error(f"❌ VR monitor initialization failed: {e}")
            traceback.print_exc()
            return
        self.vr_monitor = vr_monitor
        # self.calibrate() # Calibration is done in the VR monitor
    
    @property
    def is_calibrated(self) -> bool:
        # No calibration step: poses are absolute and deltas are computed live.
        return self.is_connected

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass
    
    @property
    def action_features(self) -> dict[str, type]:
        return {
            "left_arm_ee_delta.x": float,
            "left_arm_ee_delta.y": float,
            "left_arm_ee_delta.z": float,
            "left_arm_ee_delta.roll": float,
            "left_arm_ee_delta.pitch": float,
            "left_arm_ee_delta.yaw": float,
            "left_arm_gripper.pos": float,
            "right_arm_ee_delta.x": float,
            "right_arm_ee_delta.y": float,
            "right_arm_ee_delta.z": float,
            "right_arm_ee_delta.roll": float,
            "right_arm_ee_delta.pitch": float,
            "right_arm_ee_delta.yaw": float,
            "right_arm_gripper.pos": float,
            "head_delta.pitch": float,
            "head_delta.yaw": float,
            "base_action": list[str],
        }
    
    def _pose_delta(self, key: str, state, active: bool):
        """Delta of an absolute VRState vs. this arm's previous pose.

        Returns (dpos_chest [m], drpy_body [deg], dt [s]):
          - dpos_chest: position delta rotated into the chest (operator-yaw) frame
          - drpy_body : rotation increment in the previous EE/head body frame
                        (rotvec components → roll=X, pitch=Y, yaw=Z)
        Updates self._prev_pose[key] every call. While inactive, prev drifts to
        the current pose and a zero delta is returned, so re-activation never jumps.
        """
        zero = (np.zeros(3), np.zeros(3), 1e-3)
        if state is None or state.position_rb is None or state.quaternion_rb is None:
            return zero
        pos  = np.asarray(state.position_rb, dtype=float)
        quat = np.asarray(state.quaternion_rb, dtype=float)
        ts   = float(state.timestamp)
        prev = self._prev_pose.get(key)
        self._prev_pose[key] = (pos, quat, ts)
        if not active or prev is None:
            return zero
        prev_pos, prev_quat, prev_ts = prev
        R_rb_chest = R.from_quat(state.chest_quaternion_rb).as_matrix()
        dpos_chest = R_rb_chest.T @ (pos - prev_pos)
        R_rb_prev = R.from_quat(prev_quat).as_matrix()
        R_rb_curr = R.from_quat(quat).as_matrix()
        drpy = R.from_matrix(R_rb_prev.T @ R_rb_curr).as_rotvec() * (180.0 / np.pi)
        dt = max(ts - prev_ts, 1e-3)
        return dpos_chest, drpy, dt

    def get_action(self) -> dict[str, float]:
        action = dict.fromkeys(self.action_features.keys(), 0.0)
        action["base_action"] = []

        dual_goals = self.vr_monitor.get_latest_goal_nowait()
        if dual_goals is None:
            return action
        
        left_goal = dual_goals.get("left") if dual_goals else None
        right_goal = dual_goals.get("right") if dual_goals else None
        headset_goal = dual_goals.get("headset") if dual_goals else None
        base_goal = right_goal

        # :----- Check for record control events -----:
        if self.config.record_dataset and left_goal is not None:
            thumb_x = float(left_goal.thumbstick[0])
            thumb_y = float(left_goal.thumbstick[1])
            if thumb_x > 0.5:
                self.events["exit_early"] = True
                logger.info("🚫 Exiting the loop...")
                return action
            if thumb_x < -0.5:
                self.events["exit_early"] = True
                self.events["rerecord_episode"] = True
                logger.info("🔄 Exiting the loop and re-recording the last episode...")
                return action
            if thumb_y < -0.5:
                self.events["exit_early"] = True
                self.events["stop_recording"] = True
                logger.info("🛑 Stopping data recording...")
                return action

        # :----- Check for reset or exit -----:
        if left_goal is not None and left_goal.buttons.get('y', False):
            self.events["back_robot_to_zero"] = True
            logger.info("🔄 Resetting to zero position...")
            return action
        if left_goal is not None and left_goal.buttons.get('x', False):
            self.events["exit_teleop"] = True
            self.events["exit_early"] = True
            self.safe_exit = True
            logger.info("👋 Exiting the teleop loop...")
            return action

        # :----- Set action -----:
        stepsize = self.config.stepsize
        stepsize_ang = stepsize.get('ang', 10.0)
        stepsize_gripper = stepsize.get('gripper', 20.0)
        
        dead_zones = self.config.vr2robot_dead_zones
        dead_zones_head = dead_zones.get('head', {})
        dead_zones_arm = dead_zones.get('arm', {})

        factors = self.config.vr2robot_factors
        factors_head = factors.get('head', {})
        factors_arm = factors.get('arm', {})

        limits = self.config.vr2robot_limits
        limits_head = limits.get('head', {})
        limits_arm = limits.get('arm', {})

        # Activate pose tracking
        left_active  = left_goal  is not None and left_goal.buttons.get('squeeze', False)
        right_active = right_goal is not None and right_goal.buttons.get('squeeze', False)
        if not self.config.grip_to_activate: # Reverse: press the grip button to deactivate
            left_active = not left_active
            right_active = not right_active
        
        if not left_active and self._prev_left_active:
            logger.info("🧊 Left controller (head, left arm) is deactivated")
        if not right_active and self._prev_right_active:
            logger.info("🧊 Right controller (right arm) is deactivated")
        self._prev_left_active = left_active
        self._prev_right_active = right_active

        # Per-arm deltas, computed here at the teleop rate from absolute poses.
        # This is idempotent to VR/teleop frame-rate mismatch — no motion is
        # dropped or duplicated. _pose_delta also drifts prev while inactive so
        # the first active frame never jumps.
        left_arm_active = left_active and self.config.enable_left_arm_control
        head_active = (left_active and self.config.enable_head_control
                       and not self.config.use_thumbstick_for_head)
        left_d  = self._pose_delta("left",    left_goal,    left_arm_active)
        right_d = self._pose_delta("right",   right_goal,   right_active)
        head_d  = self._pose_delta("headset", headset_goal, head_active)

        # Head control
        if left_active and self.config.enable_head_control:
            if self.config.use_thumbstick_for_head:
                _, _, dt = head_d
                thumb_x = float(left_goal.thumbstick[0])
                thumb_y = float(left_goal.thumbstick[1])
                if abs(thumb_x) > 0.5:
                    action["head_delta.yaw"] = (1 if thumb_x < 0 else -1) * stepsize_ang * dt
                if abs(thumb_y) > 0.5:
                    action["head_delta.pitch"] = (1 if thumb_y < 0 else -1) * stepsize_ang * dt
            else:  # Use VR headset orientation: yaw = body Z, pitch = body Y.
                _, drpy, dt = head_d
                dyaw, dpitch = drpy[2], drpy[1]
                _limit = limits_head.get('ang', 45.0) * dt
                if abs(dyaw) > dead_zones_head.get('yaw', 10.0) * dt:
                    action["head_delta.yaw"] = np.clip(dyaw * factors_head.get('yaw', 1), -_limit, _limit)
                if abs(dpitch) > dead_zones_head.get('pitch', 10.0) * dt:
                    action["head_delta.pitch"] = np.clip(dpitch * factors_head.get('pitch', 1), -_limit, _limit)

        def _set_action_arm(delta, prefix):
            # delta = (dpos_chest [m], drpy_body [deg], dt [s]).
            # Dead-zones are velocity thresholds (m/s, deg/s) × dt → per-frame
            # cut-off, so slow steady motion isn't filtered at high frame rates.
            dpos, drpy, dt = delta
            _pos_limit = limits_arm.get('pos', 0.1) * dt # m
            _ang_limit = limits_arm.get('ang', 45.0) * dt # deg

            dx, dy, dz = dpos
            if abs(dx) > dead_zones_arm.get('x', 0.025) * dt:
                action[f"{prefix}_ee_delta.x"] = np.clip(dx * factors_arm.get('x', 1), -_pos_limit, _pos_limit)
            if abs(dy) > dead_zones_arm.get('y', 0.025) * dt:
                _limit = _pos_limit if self.config.use_placo_ik else _ang_limit
                action[f"{prefix}_ee_delta.y"] = np.clip(dy * factors_arm.get('y', 1), -_limit, _limit)
            if abs(dz) > dead_zones_arm.get('z', 0.025) * dt:
                action[f"{prefix}_ee_delta.z"] = np.clip(dz * factors_arm.get('z', 1), -_pos_limit, _pos_limit)

            droll, dpitch, dyaw = drpy
            if abs(droll) > dead_zones_arm.get('roll', 10.0) * dt:
                action[f"{prefix}_ee_delta.roll"] = np.clip(droll * factors_arm.get('roll', 1), -_ang_limit, _ang_limit)
            if abs(dpitch) > dead_zones_arm.get('pitch', 10.0) * dt:
                action[f"{prefix}_ee_delta.pitch"] = np.clip(dpitch * factors_arm.get('pitch', 1), -_ang_limit, _ang_limit)
            if abs(dyaw) > dead_zones_arm.get('yaw', 10.0) * dt:
                action[f"{prefix}_ee_delta.yaw"] = np.clip(dyaw * factors_arm.get('yaw', 1), -_ang_limit, _ang_limit)

        def _set_action_gripper(goal, dt, prefix, close_gripper):
            if goal.trigger_value > 0.5:
                if close_gripper:
                    action[f"{prefix}_gripper.pos"] = -stepsize_gripper * dt
                else:
                    action[f"{prefix}_gripper.pos"] = stepsize_gripper * dt
        
        close_gripper = right_goal is not None and right_goal.buttons.get('a', False)
        # Left hand
        if left_arm_active:
            _set_action_arm(left_d, "left_arm")
            _set_action_gripper(left_goal, left_d[-1], "left_arm", close_gripper)
        # Right hand
        if right_active:
            _set_action_arm(right_d, "right_arm")
            _set_action_gripper(right_goal, right_d[-1], "right_arm", close_gripper)

        # Base
        if base_goal is not None and self.config.enable_base_control:
            thumb_x = float(base_goal.thumbstick[0])
            thumb_y = float(base_goal.thumbstick[1])
            if abs(thumb_x) > 0.5:
                if thumb_x > 0:
                    action["base_action"].append('rotate_right')
                else:
                    action["base_action"].append('rotate_left')
            if abs(thumb_y) > 0.5:
                if thumb_y > 0:
                    action["base_action"].append('backward')
                else:
                    action["base_action"].append('forward')
        
        return action

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def send_feedback(self, feedback: dict) -> None:
        """Forward camera frames from the observation to the in-VR overlay."""
        if self.vr_monitor is None:
            return
        frames = select_image_frames(feedback)
        if frames:
            self.vr_monitor.update_camera_frames(frames)
    
    def disconnect(self) -> None:
        if self.vr_monitor is not None:
            # Signal the monitor to stop by setting is_running = False
            # The monitor's finally block in start_monitoring() will call stop_monitoring()
            # which is async and will be properly awaited within its own event loop
            self.vr_monitor.is_running = False
            
            # Wait for the VR thread to finish (with timeout)
            if self._vr_thread is not None and self._vr_thread.is_alive():
                self._vr_thread.join(timeout=5.0)
                if self._vr_thread.is_alive():
                    logger.warning("VR monitor thread did not stop within timeout")
        
        self.vr_monitor = None
        self._vr_thread = None