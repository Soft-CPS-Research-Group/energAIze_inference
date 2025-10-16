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


def test_breaker_allocation_strategy():
    manifest_path = Path("examples/fleet_rule_based/artifact_manifest.json")
    artifacts_dir = manifest_path.parent
    if store.is_configured():
        store.unload()
    store.load(manifest_path, artifacts_dir, 0)

    client = TestClient(app)
    charger_ids = [
        "AC000001_1", "AC000002_1", "AC000003_1", "AC000004_1", "AC000005_1",
        "AC000006_1", "AC000007_1", "AC000008_1", "AC000009_1", "AC000010_1",
        "AC000011_1", "AC000012_1", "AC000013_1", "AC000014_1",
        "ACEXT001_1", "ACEXT002_1", "ACEXT003_1", "ACEXT004_1"
    ]
    charging_sessions = {cid: {"power": 0, "electric_vehicle": ""} for cid in charger_ids}
    charging_sessions["AC000001_1"]["electric_vehicle"] = "11823"
    charging_sessions["AC000004_1"]["electric_vehicle"] = "11837"
    charging_sessions["AC000002_1"] = {"power": 4.0, "electric_vehicle": "NF-1"}
    charging_sessions["AC000003_1"] = {"power": 4.0, "electric_vehicle": "NF-2"}

    payload = {
        "timestamp": "2025-10-16T09:13:10",
        "non_shiftable_load": 30,
        "solar_generation": 2,
        "electricity_pricing": 0.15,
        "charging_sessions": charging_sessions,
        "electric_vehicles": {
            "11823": {
                "SoC": 0.2,
                "flexibility": {
                    "charger_id": "AC000001_1",
                    "departure_time": "2025-10-16T18:00:00"
                }
            },
            "11837": {
                "SoC": 0.3,
                "flexibility": {
                    "charger_id": "AC000004_1",
                    "departure_time": "2025-10-16T19:00:00"
                }
            }
        },
        "meters": {
            "m_1": {"energy_in": 4},
            "m_2": {"energy_in": 2}
        },
        "batteries": {
            "b_1": {"energy_in": 1, "last_soc": 0.1},
            "b_2": {"energy_in": 1, "last_soc": 0.4}
        },
        "pv_panels": {
            "pv_1": {"energy": 0.5},
            "pv_2": {"energy": 0.5}
        }
    }

    try:
        response = client.post("/inference", json={"features": payload})
        assert response.status_code == 200
        actions = response.json()["actions"]["0"]

        assert actions["AC000002_1"] == pytest.approx(4.6, rel=1e-6)
        assert actions["AC000003_1"] == pytest.approx(4.6, rel=1e-6)
        assert actions["AC000001_1"] <= 4.6 + 1e-6
        assert actions["AC000004_1"] <= 4.6 + 1e-6

        inactive = set(charger_ids) - {"AC000001_1", "AC000002_1", "AC000003_1", "AC000004_1"}
        for cid in inactive:
            assert actions[cid] == pytest.approx(0.0, abs=1e-6)

        board_total = 0.0
        board_total += max(payload["non_shiftable_load"] - payload["solar_generation"], 0)
        board_total += sum(actions[cid] for cid in actions if cid not in {"b_1", "b_2"})
        board_total += sum(actions[cid] for cid in ["b_1", "b_2"])
        assert board_total <= 43.0 + 1e-6

        line_groups = {
            "L1": ["ACEXT003_1", "ACEXT002_1", "AC000014_1", "AC000011_1", "AC000008_1", "AC000005_1", "AC000002_1", "ACEXT001_1"],
            "L2": ["ACEXT004_1", "AC000012_1", "AC000009_1", "AC000006_1", "AC000003_1"],
            "L3": ["AC000013_1", "AC000010_1", "AC000007_1", "AC000004_1", "AC000001_1"]
        }
        for group, chargers in line_groups.items():
            total = sum(actions.get(cid, 0.0) for cid in chargers)
            assert total <= 11.0 + 1e-6

        available_headroom = max(43.0 - (payload["non_shiftable_load"] - payload["solar_generation"]) - sum(actions[cid] for cid in actions if cid not in {"b_1", "b_2"}), 0.0)
        assert actions["b_1"] <= available_headroom + 1e-6
        assert actions["b_2"] <= available_headroom + 1e-6
    finally:
        store.unload()
