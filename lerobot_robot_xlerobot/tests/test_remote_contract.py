"""Offline contract checks for remote inference (no robot, no GPU, no network).

1. The server YAML parses into lerobot's ServerConfig.
2. Its features match a SmolVLA checkpoint config (set CHECKPOINT_CONFIG to a config.json;
   skipped otherwise) with the same rule the policy server applies.
3. The client-side admission rules of `build_remote_rollout_context` accept the arms-only
   robot + rollout.py processors + rename_map against that YAML.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.remote_inference.configs import ServerConfig
from lerobot_robot_xlerobot import XLeRobotX, XLeRobotXConfig

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples"
SERVER_YAML = EXAMPLES / "remote_inference" / "server_smolvla_robotwin.yaml"
RENAME = {
    "observation.images.head": "observation.images.camera1",
    "observation.images.left_wrist": "observation.images.camera2",
    "observation.images.right_wrist": "observation.images.camera3",
}


def load_server_cfg() -> ServerConfig:
    with open(SERVER_YAML) as f:
        return draccus.load(ServerConfig, f)


def test_server_yaml_parses():
    cfg = load_server_cfg()
    assert cfg.execution.action_fps == pytest.approx(16.67)
    assert cfg.action_feature.names[0] == "left_arm_shoulder_pan.pos"


def test_server_features_match_checkpoint():
    path = os.environ.get("CHECKPOINT_CONFIG")
    if not path:
        pytest.skip("set CHECKPOINT_CONFIG=/path/to/pretrained_model/config.json")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pretrained import PreTrainedPolicy

    policy_cfg = PreTrainedConfig.from_pretrained(str(Path(path).parent))
    cfg = load_server_cfg()
    # Same check the policy server runs at load time.
    PreTrainedPolicy.validate_chunk_input_features(SimpleNamespace(config=policy_cfg), tuple(cfg.features))
    assert tuple(policy_cfg.action_feature.shape) == tuple(cfg.action_feature.shape)


def test_client_admission_rules(tmp_path):
    from lerobot.rollout import context as rc

    sys.path.insert(0, str(EXAMPLES))
    import rollout as xl_rollout  # examples/rollout.py

    cams = {name: OpenCVCameraConfig(index_or_path=f"/dev/null-{name}", width=640, height=480, fps=30)
            for name in ("head", "left_wrist", "right_wrist")}
    robot = XLeRobotX(XLeRobotXConfig(id="t", calibration_dir=tmp_path, cameras=cams, arms_only=True))
    cfg = SimpleNamespace(sim_convention=True, dry_run=False, joint_offset_deg={})
    ignore = xl_rollout.features_to_ignore(robot, enable_base_control=False)
    obs_proc, act_proc = xl_rollout.build_robot_processors(cfg, ignore)
    processors = rc._resolve_robot_processors(xl_rollout.build_action_feature_select(ignore), act_proc, obs_proc)

    server = load_server_cfg()
    state = next(f for f in server.features if f.name == "observation.state")
    observation_hw = {k: v for k, v in robot.observation_features.items() if isinstance(v, tuple) or v is float}
    observation_hw = rc._align_to_checkpoint_order(observation_hw, list(state.names), what="state")
    action_hw = rc._align_to_checkpoint_order(robot.action_features, list(server.action_feature.names), what="action")
    assert list(action_hw) == list(server.action_feature.names)
    features = rc._aggregate_rollout_features(processors, action_hw, observation_hw, use_videos=True)
    mapped = {RENAME.get(n, n): f for n, f in features.items() if n.startswith("observation.")}
    assert {f.name for f in server.features} <= set(mapped)
    assert list(features["observation.state"]["names"]) == list(state.names)
    assert tuple(features["observation.images.head"]["shape"]) == (480, 640, 3)
