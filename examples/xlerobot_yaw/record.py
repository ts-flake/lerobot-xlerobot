# !/usr/bin/env python
"""Unified dataset recording entry point for XLeRobotYaw (pure teleop data collection).

Pick the teleop device with ``--teleop.type``:

```shell
# VR controllers
uv run examples/xlerobot_yaw/record.py \
    --robot.type=xlerobot_yaw --robot.port1=/dev/ttyACM0 --robot.port2=/dev/ttyACM1 \
    --teleop.type=xlerobot_yaw_vr \
    --dataset.repo_id=<hf_user>/<dataset> --dataset.single_task="Grab the cube" \
    --dataset.num_episodes=5 --display_data=true
```

```shell
# PS5 gamepad
uv run examples/xlerobot_yaw/record.py \
    --robot.type=xlerobot_yaw --robot.port1=/dev/ttyACM0 --robot.port2=/dev/ttyACM1 \
    --teleop.type=xlerobot_yaw_gamepad \
    --dataset.repo_id=<hf_user>/<dataset> --dataset.single_task="Grab the cube" \
    --dataset.num_episodes=5 --display_data=true
```

This is data collection only — no policy inference. To deploy a trained policy,
use ``lerobot-rollout``. The device-specific logic (input -> ee-delta, control
``events``) lives in each teleoperator class; this script wires the shared
record loop (``record_loop`` / ``RecordConfig`` from lerobot) around the
XLeRobotYaw lifecycle: zero-position homing, reset between episodes, and
restoring the initial pose on exit.
"""

import logging
import traceback
from dataclasses import asdict
from pprint import pformat

from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.configs import parser
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.scripts.lerobot_record import RecordConfig, record_loop
from lerobot.robots.xlerobot_yaw import XLeRobotYaw
from lerobot.robots.xlerobot_yaw.utils.action_utils import move_robot_to_position, move_robot_to_zero_position
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.utils import log_say
from lerobot.utils.color_logger import init_color_logging

from teleop_common import (
    build_action_feature_select,
    build_ee_delta_to_joints_processor,
    build_observation_feature_select,
    features_to_ignore,
    make_teleop_device,
    wait_until_ready,
)

logger = logging.getLogger(__name__)


@parser.wrap()
def main(cfg: RecordConfig):
    # Initialize logging
    init_color_logging(cfg.teleop.console_level)
    logger.info("\n" + pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name=f"xlerobot_yaw_record_{cfg.teleop.type}", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    # The teleop devices switch input mapping when recording (e.g. thumbstick -> record control),
    # and the dataset fps must match the teleop rate.
    cfg.teleop.record_dataset = True
    cfg.dataset.fps = cfg.teleop.fps

    FPS = cfg.teleop.fps
    ZERO_POSITION_OFFSET = cfg.teleop.zero_position_offset
    NUM_EPISODES = cfg.dataset.num_episodes
    RESET_TIME_SEC = cfg.dataset.reset_time_s

    # Initialize the robot and teleoperator
    robot = XLeRobotYaw(cfg.robot)
    teleop = make_teleop_device(cfg.teleop)
    if getattr(cfg.teleop, "stream_cameras_to_vr", False):
        logger.info("📷 Streaming camera frames to the in-VR overlay (stream_cameras_to_vr=true)")

    # Two-stage action path (kept split on purpose):
    #   teleop_action_processor : ee-delta -> safe joint command (the recorded action *values*)
    #   robot_action_processor  : feature-select to the enabled DOFs (what's actually *sent*)
    teleop_action_processor = build_ee_delta_to_joints_processor(robot, cfg.teleop, FPS)
    ignore = features_to_ignore(
        robot,
        enable_left_arm_control=cfg.teleop.enable_left_arm_control,
        enable_base_control=cfg.teleop.enable_base_control,
        enable_head_control=cfg.teleop.enable_head_control,
    )
    robot_action_processor = build_action_feature_select(ignore)
    robot_observation_processor = build_observation_feature_select(ignore)

    dataset_features = combine_feature_dicts(
        # Action schema is derived from robot_action_processor (the feature selector), not the IK
        # pipeline, so the saved action == the sent action (enabled DOFs only). See record_loop.
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
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
    initial_pos = {}  # populated after connect; guarded in finally so early failures don't NameError

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # eval_ datasets are reserved for policy evaluation (lerobot-rollout).
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "This recorder is for data collection only; use lerobot-rollout for policy deployment."
                )
            cfg.dataset.stamp_repo_id()
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
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        # Connect to the robot and teleoperator
        robot.connect()
        if not robot.is_connected:
            logger.error("❌ Robot is not connected!")
            exit(1)

        teleop.connect()
        if not wait_until_ready(teleop):
            logger.error("❌ Teleop device is not connected!")
            exit(1)

        listener, _ = init_keyboard_listener()
        events = teleop.events

        if not cfg.dataset.streaming_encoding:
            logger.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.camera_encoder.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        # Save the initial position to restore on exit
        initial_obs = robot.get_observation()
        initial_pos = {k.removesuffix(".pos"): v for k, v in initial_obs.items() if k.endswith(".pos")}

        # Move to zero position
        move_robot_to_zero_position(robot, fps=FPS, duration=5.0, end_offset=ZERO_POSITION_OFFSET)

        # :----- Main record loop -----:
        logger.info("Starting record loop. Move your teleop device to teleoperate the robot...")

        log_say("Start in 5 seconds", cfg.play_sounds)
        precise_sleep(5)
        with VideoEncodingManager(dataset):
            episode_idx = 0
            while episode_idx < NUM_EPISODES and not events["stop_recording"] and not events["exit_teleop"]:
                log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}", cfg.play_sounds)

                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

                # Reset the environment without recording if not stopping or re-recording
                if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
                    log_say("Reset the robot position", cfg.play_sounds, blocking=True)
                    move_robot_to_zero_position(robot, fps=FPS, duration=5.0, end_offset=ZERO_POSITION_OFFSET)
                    teleop_action_processor.reset()
                    log_say(f"Reset the environment for {RESET_TIME_SEC} seconds", cfg.play_sounds)
                    precise_sleep(RESET_TIME_SEC)

                # Re-record, skip saving
                if events["rerecord_episode"]:
                    log_say(f"Re-recording episode {episode_idx + 1}", cfg.play_sounds, blocking=True)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    teleop_action_processor.reset()
                    dataset.clear_episode_buffer()
                    continue

                # Save episode
                log_say("Saving episode", cfg.play_sounds)
                dataset.save_episode()
                episode_idx += 1

    except Exception as e:
        logger.error(f"❌ Error in record loop: {e}")
        traceback.print_exc()

    finally:
        # Clean up
        if dataset:
            dataset.finalize()

        log_say("Stop recording", cfg.play_sounds, blocking=True)
        if initial_pos:
            log_say("Exiting the record loop, returning to initial position...", cfg.play_sounds)
            move_robot_to_position(robot, end_pos=initial_pos, fps=FPS, duration=5.0)

        robot.disconnect()
        teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub and dataset:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)


if __name__ == "__main__":
    main()
