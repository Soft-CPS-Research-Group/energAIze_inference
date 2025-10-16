import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store


def _build_rule_based_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    policy = {
        "default_actions": {"hvac": 1.0},
        "rules": [
            {"if": {"mode": "standby"}, "actions": {"hvac": 0.0}},
            {"if": {"mode": "cooling"}, "actions": {"hvac": 0.5}},
        ],
    }
    policy_path = bundle_dir / "policy_agent_0.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

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


def test_api_accepts_string_features(tmp_path):
    manifest_path = _build_rule_based_bundle(tmp_path)
    if store.is_configured():
        store.unload()
    store.load(manifest_path, None, 0)

    client = TestClient(app)

    try:
        response = client.post("/inference", json={"features": {"mode": "cooling"}})
        assert response.status_code == 200
        body = response.json()
        assert body["actions"]["0"]["hvac"] == 0.5
    finally:
        store.unload()


def test_api_flattens_nested_payload(tmp_path):
    bundle_dir = tmp_path / "identity"
    (bundle_dir / "onnx_models").mkdir(parents=True)

    manifest = {
        "manifest_version": 1,
        "metadata": {},
        "simulator": {},
        "training": {},
        "topology": {"num_agents": 1},
        "algorithm": {"name": "Identity", "hyperparameters": {}},
        "environment": {
            "observation_names": [["session.AC.power"]],
            "encoders": [[{"type": "NoNormalization", "params": {}}]],
            "action_bounds": [[{"low": [0], "high": [1]}]],
            "action_names": ["hvac"],
            "reward_function": {"name": "RewardFunction", "params": {}},
        },
        "agent": {
            "format": "onnx",
            "artifacts": [
                {
                    "agent_index": 0,
                    "path": "onnx_models/agent_0.onnx",
                    "observation_dimension": 1,
                    "action_dimension": 1,
                }
            ],
        },
    }

    import onnx
    from onnx import TensorProto, helper

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "IdentityGraph", [input_tensor], [output_tensor])
    model = helper.make_model(graph, producer_name="flatten-test", opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.save(model, bundle_dir / "onnx_models" / "agent_0.onnx")

    manifest_path = bundle_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    alias_file = bundle_dir / "aliases.json"
    alias_file.write_text('{"charging.sessions.AC0001.power": "session.AC.power"}', encoding="utf-8")

    if store.is_configured():
        store.unload()
    store.load(manifest_path, bundle_dir, 0, alias_file)

    client = TestClient(app)
    nested_payload = {
        "charging": {
            "sessions": {
                "AC0001": {
                    "power": 0.8,
                }
            }
        }
    }

    try:
        response = client.post("/inference", json={"features": nested_payload})
        assert response.status_code == 200
        body = response.json()
        assert body["actions"]["0"]["hvac"] == pytest.approx(0.8, rel=1e-6)
    finally:
        store.unload()


def test_health_reports_metadata(tmp_path):
    manifest_path = _build_rule_based_bundle(tmp_path)
    if store.is_configured():
        store.unload()
    store.load(manifest_path, None, 0)

    client = TestClient(app)
    try:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["configured"] is True
        assert body["manifest_path"].endswith("artifact_manifest.json")
        assert isinstance(body.get("providers"), list)
        assert "gpu_available" in body
    finally:
        store.unload()
