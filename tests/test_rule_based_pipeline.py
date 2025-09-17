import json
from pathlib import Path

from app.services.pipeline import InferencePipeline
from app.utils.manifest import Manifest


def make_rbc_manifest(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "rbc_bundle"
    bundle_dir.mkdir()

    policy = {
        "default_actions": {"action": 1.0},
        "rules": [
            {"if": {"mode": "idle"}, "actions": {"action": 0.0}}
        ],
    }
    (bundle_dir / "policies").mkdir()
    policy_path = bundle_dir / "policies" / "agent_0.json"
    with policy_path.open("w", encoding="utf-8") as handle:
        json.dump(policy, handle)

    manifest = {
        "manifest_version": 1,
        "metadata": {"experiment_name": "rbc", "run_name": "rbc"},
        "simulator": {},
        "training": {},
        "topology": {"num_agents": 1},
        "algorithm": {"name": "RuleBasedPolicy", "hyperparameters": {}},
        "environment": {
            "observation_names": [["mode"]],
            "encoders": [[{"type": "NoNormalization", "params": {}}]],
            "action_bounds": [[{"low": [0], "high": [1]}]],
            "action_names": ["action"],
            "reward_function": {"name": "RewardFunction", "params": {}},
        },
        "agent": {
            "format": "rule_based",
            "artifacts": [
                {
                    "agent_index": 0,
                    "path": "policies/agent_0.json",
                    "format": "rule_based",
                    "config": {},
                }
            ],
        },
    }

    manifest_path = bundle_dir / "artifact_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return manifest_path


def test_rule_based_runtime(tmp_path):
    manifest_path = make_rbc_manifest(tmp_path)
    manifest = Manifest.parse_file(manifest_path)
    pipeline = InferencePipeline(manifest=manifest, artifacts_root=manifest_path.parent, agent_index=0)

    result = pipeline.inference({"mode": "idle"})
    assert result["0"]["action"] == 0.0

    result = pipeline.inference({"mode": "active"})
    assert result["0"]["action"] == 1.0
