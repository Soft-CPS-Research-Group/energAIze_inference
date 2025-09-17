from pathlib import Path

import pytest

from app.state import store
from app.services.pipeline import InferencePipeline
from app.utils.manifest import Manifest


def build_manifest(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    manifest = {
        "manifest_version": 1,
        "metadata": {},
        "simulator": {},
        "training": {},
        "topology": {"num_agents": 1},
        "algorithm": {"name": "MADDPG", "hyperparameters": {}},
        "environment": {
            "observation_names": [["feat"]],
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
                    "path": "policy.json",
                    "format": "rule_based",
                    "config": {"default_actions": {"action": 1.0}},
                }
            ],
        },
    }
    (bundle_dir / "policy.json").write_text('{"default_actions": {"action": 1.0}}')
    manifest_path = bundle_dir / "artifact_manifest.json"
    manifest_path.write_text(
        __import__("json").dumps(manifest),
        encoding="utf-8",
    )
    return manifest_path


def test_store_load_unload(tmp_path):
    manifest_path = build_manifest(tmp_path)
    record = store.load(manifest_path, None, 0)
    assert record.pipeline is not None
    assert store.is_configured()
    store.unload()
    assert not store.is_configured()
    with pytest.raises(RuntimeError):
        store.get_pipeline()
