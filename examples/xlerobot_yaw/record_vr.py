# !/usr/bin/env python

import logging, traceback
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

# import torch

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.configs import PreTrainedConfig, parser
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.policies import (
    ActionInterpolator,
    PreTrainedPolicy,
    make_policy,
    make_pre_post_processors,
    make_robot_action,
)
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
    rename_stats,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
    observation_to_transition,
    transition_to_observation,
)
from lerobot.model.kinematics import RobotKinematics
from lerobot.model.rr_kinematics import RRKinematics

from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.utils import log_say
from lerobot.utils.color_logger import init_color_logging

from lerobot.robots.xlerobot_yaw import *
from lerobot.robots.xlerobot_yaw.robot_action_processor import (
    InverseKinematicsDeltaToJoints,
    AnalyticalInverseKinematicsDeltaToJoints,
    BaseJointAction,
    JointClipNormValue,
    SafeGoalPosition,
    EMAJointAction,
    RoundOffAction,
    LogAction
)
from lerobot.robots.xlerobot_yaw.robot_action_observation_processor import (
    RobotActionFeatureSelect,
    RobotObservationFeatureSelect,
)
from lerobot.robots.xlerobot_yaw.utils.action_utils import move_robot_to_position, move_robot_to_zero_position
from lerobot.teleoperators.xlerobot_yaw_vr import *

from lerobot.scripts.lerobot_record import RecordConfig, record_loop

@parser.wrap()
def main(cfg: RecordConfig):
    # Initialize logging
    init_color_logging()
    logger = logging.getLogger(__name__)
    logger.info("\n" + pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="xlerobot_yaw_teleop_vr", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    # Initialize the robot and teleoperator
    cfg.teleop.record_dataset = True
    cfg.dataset.fps = cfg.teleop.fps

    FPS = cfg.teleop.fps
    ZERO_POSITION_OFFSET = cfg.teleop.zero_position_offset
    NUM_EPISODES = cfg.dataset.num_episodes
    RESET_TIME_SEC = cfg.dataset.reset_time_s

    # Initialize the robot and teleoperator
    robot = XLeRobotYaw(cfg.robot)
    teleop = XLeRobotYawVR(cfg.teleop)
    if getattr(cfg.teleop, "stream_cameras_to_vr", False):
        logger.info("📷 Streaming camera frames to the in-VR overlay (stream_cameras_to_vr=true)")

    # Build pipeline to convert vr action to joint action
    if cfg.teleop.use_placo_ik:
        logger.info("Using Placo IK")
        urdf_path = str(Path(__file__).resolve().parent / "simulation/so101_yaw/so101_yaw.urdf")
        ik_step = InverseKinematicsDeltaToJoints(
            kinematics_left=RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="gripper_back",
                joint_names=[f"joint{i}" for i in range(1, 7)] # exclude the gripper joint
            ),
            kinematics_right=RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="gripper_back",
                joint_names=[f"joint{i}" for i in range(1, 7)] # exclude the gripper joint
            ),
            motor_names=robot.left_arm_motors + robot.right_arm_motors + robot.head_motors
        )
    else:
        logger.info("Using analytical IK")
        ik_step = AnalyticalInverseKinematicsDeltaToJoints(
            kinematics_left=RRKinematics(use_degrees=True, offsets=[90, 90], reversed=[True, False]),
            kinematics_right=RRKinematics(use_degrees=True, offsets=[90, 90], reversed=[True, False]),
            motor_names=robot.left_arm_motors + robot.right_arm_motors + robot.head_motors,
        )
    vr_to_robot_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            ik_step,
            BaseJointAction(robot=robot),
            JointClipNormValue(robot=robot),
            SafeGoalPosition(
                max_relative_target=robot.config.max_relative_target,
                motor_names=robot.left_arm_motors + robot.right_arm_motors + robot.head_motors
            ),
            EMAJointAction(fps=FPS),
            RoundOffAction(),
            LogAction(logger),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # _,robot_action_processor, robot_observation_processor = make_default_processors()
    features_to_ignore = []
    if not cfg.teleop.enable_left_arm_control:
        features_to_ignore += [feat for feat in robot.action_features if 'left_arm_' in feat]
    if not cfg.teleop.enable_base_control:
        features_to_ignore += ['x.vel', 'y.vel', 'theta.vel']
    if not cfg.teleop.enable_head_control:
        features_to_ignore += [feat for feat in robot.action_features if 'head_' in feat]
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[RobotActionFeatureSelect(features_to_ignore)],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[RobotObservationFeatureSelect(features_to_ignore)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None
    initial_pos = {}   # populated after connect; guarded in finally so early failures don't NameError

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # Create empty dataset or load existing saved episodes
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        # Load pretrained policy
        policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor = None
        postprocessor = None
        interpolator = None
        if cfg.policy is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": cfg.policy.device},
                    "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
                },
            )
            # Create interpolator for smoother policy control
            if cfg.interpolation_multiplier > 1:
                interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
                logging.info(f"Action interpolation enabled: {cfg.interpolation_multiplier}x control rate")

        # Connect to the robot and teleoperator
        robot.connect()
        if not robot.is_connected:
            logger.error("❌ Robot is not connected!")
            exit(1)

        teleop.connect()

        # VR connects asynchronously via the headset browser — wait for it.
        while not teleop.is_connected:
            precise_sleep(0.1)  # Wait for VR to connect
        while not teleop.is_calibrated:
            precise_sleep(0.1)  # Wait for VR to calibrate

        listener, keyboard_events = init_keyboard_listener()
        teleop_events = teleop.events
        events = teleop_events

        if not cfg.dataset.streaming_encoding:
            logger.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        # Save the initial position
        initial_obs = robot.get_observation()
        initial_pos = {k.removesuffix(".pos"): v for k, v in initial_obs.items() if k.endswith(".pos")}

        # Move to zero position
        move_robot_to_zero_position(robot, fps=FPS, duration=5.0, end_offset=ZERO_POSITION_OFFSET)

        # :----- Main teleop loop -----:
        logger.info("Starting teleop loop. Move your VR controllers to teleoperate the robot...")

        log_say('Start in 5 seconds', cfg.play_sounds)
        precise_sleep(5)
        with VideoEncodingManager(dataset):
            episode_idx = 0
            while episode_idx < NUM_EPISODES and not events["stop_recording"] and not events["exit_teleop"]:
                log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}", cfg.play_sounds)

                # Main record loop
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                    interpolator=interpolator,
                    teleop_action_processor=vr_to_robot_joints_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

                # Reset the environment without recording if not stopping or re-recording
                if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
                    log_say("Reset the robot position", cfg.play_sounds, blocking=True)
                    move_robot_to_zero_position(robot, fps=FPS, duration=5.0, end_offset=ZERO_POSITION_OFFSET)
                    vr_to_robot_joints_processor.reset()
                    log_say(f"Reset the environment for {RESET_TIME_SEC} seconds", cfg.play_sounds)
                    precise_sleep(RESET_TIME_SEC)

                # Re-record, skip saving
                if events["rerecord_episode"]:
                    log_say(f"Re-recording episode {episode_idx + 1}", cfg.play_sounds, blocking=True)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    vr_to_robot_joints_processor.reset()
                    dataset.clear_episode_buffer()
                    continue

                # Save episode
                log_say("Saving episode", cfg.play_sounds)
                dataset.save_episode()
                episode_idx += 1

    except Exception as e:
        logger.error(f"❌ Error in teleop loop: {e}")
        traceback.print_exc()

    finally:
        # Clean up
        if dataset:
            dataset.finalize()

        log_say("Stop recording", cfg.play_sounds, blocking=True)
        if initial_pos:
            log_say("Exiting the teleop loop, returning to initial position...", cfg.play_sounds)
            move_robot_to_position(robot, end_pos=initial_pos, fps=FPS, duration=5.0)

        robot.disconnect()
        teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

if __name__ == "__main__":
    main()
