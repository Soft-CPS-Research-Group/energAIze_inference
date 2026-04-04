import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.settings import settings
from app.state import store


def _build_rule_based_bundle(bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    policy = {
        "default_actions": {"hvac": 1.0},
        "rules": [{"if": {"mode": "standby"}, "actions": {"hvac": 0.0}}],
    }
    (bundle_dir / "policy_agent_0.json").write_text(json.dumps(policy), encoding="utf-8")

    manifest = {
        "manifest_version": 1,
        "metadata": {},
        "simulator": {},
        "training": {},
        "topology": {"num_agents": 1},
        "algorithm": {"name": "RuleBasedPolicy", "hyperparameters": {}},
        "environment": {
            "observation_names": [["mode"]],
            "encoders": [[{"type": "NoNormalization", "params": {}}]],
            "action_bounds": [[{"low": [0], "high": [1]}]],
            "action_names": ["hvac"],
            "reward_function": {"name": "RewardFunction", "params": {}},
        },
        "agent": {
            "format": "rule_based",
            "artifacts": [
                {
                    "agent_index": 0,
                    "path": "policy_agent_0.json",
                    "format": "rule_based",
                    "config": {},
                }
            ],
        },
    }

    manifest_path = bundle_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_admin_load_accepts_manifest_inside_allowed_bundle_root(tmp_path: Path):
    client = TestClient(app)
    allowed_root = tmp_path / "allowed"
    manifest_path = _build_rule_based_bundle(allowed_root / "bundle")

    prior_root = settings.allowed_bundle_root

    try:
        if store.is_configured():
            store.unload()

        settings.allowed_bundle_root = allowed_root
        response = client.post(
            "/admin/load",
            json={
                "manifest_path": str(manifest_path),
                "artifacts_dir": str(manifest_path.parent),
            },
        )
        assert response.status_code == 200, response.text
    finally:
        settings.allowed_bundle_root = prior_root
        if store.is_configured():
            store.unload()


def test_admin_load_rejects_manifest_outside_allowed_bundle_root(tmp_path: Path):
    client = TestClient(app)
    allowed_root = tmp_path / "allowed"
    disallowed_manifest = _build_rule_based_bundle(tmp_path / "outside" / "bundle")

    prior_root = settings.allowed_bundle_root

    try:
        if store.is_configured():
            store.unload()

        settings.allowed_bundle_root = allowed_root
        response = client.post(
            "/admin/load",
            json={
                "manifest_path": str(disallowed_manifest),
                "artifacts_dir": str(disallowed_manifest.parent),
            },
        )
        assert response.status_code == 400
        assert "outside ALLOWED_BUNDLE_ROOT" in response.json().get("detail", "")
    finally:
        settings.allowed_bundle_root = prior_root
        if store.is_configured():
            store.unload()
