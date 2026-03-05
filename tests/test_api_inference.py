import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from loguru import logger
from pydantic import ValidationError

from app.logging import get_logger, init_logging
from app.main import app
from app.services.rbc.icharging import ChargerState, IchargingBreakerRuntime, IchargingRuntimeConfig
from app.settings import settings
from app.state import store
from app.utils.manifest import load_manifest


ICHARGING_BOARD_LIMIT_KW = 55.0
BREAKER_ONLY_BASE_BOARD_LIMIT_KW = 55.0
ICHARGING_BOAVISTA_WITH_FLEX_DIR = Path("examples/icharging_boavista_with_flex")
ICHARGING_BOAVISTA_WITHOUT_FLEX_DIR = Path("examples/icharging_boavista_without_flex")


def _to_bundle_features(payload: dict) -> dict:
    if isinstance(payload.get("observations"), dict):
        return payload

    require_observations = False
    if store.is_configured():
        try:
            pipeline = store.get_pipeline()
            artifact_cfg = pipeline.manifest.get_artifact(pipeline.agent_index).config or {}
            require_observations = bool(artifact_cfg.get("require_observations_envelope", False))
        except Exception:  # noqa: BLE001
            require_observations = False

    if not require_observations:
        return payload

    features = {"observations": dict(payload), "forecasts": {}}
    timestamp = payload.get("timestamp")
    timestamp_date = payload.get("timestamp.$date")
    features["observations"].pop("timestamp", None)
    features["observations"].pop("timestamp.$date", None)
    if timestamp is not None:
        features["timestamp"] = timestamp
    elif timestamp_date is not None:
        features["timestamp.$date"] = timestamp_date
    return features


def _post_payload(client: TestClient, payload: dict):
    return client.post("/inference", json={"features": _to_bundle_features(payload)})


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


def _build_rule_based_alias_bundle(tmp_path: Path) -> tuple[Path, Path]:
    bundle_dir = tmp_path / "alias_bundle"
    bundle_dir.mkdir()

    policy = {
        "default_actions": {"hvac": 0.0},
        "rules": [
            {"if": {"electricity_pricing": 5}, "actions": {"hvac": 0.9}},
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
            "observation_names": [["electricity_pricing"]],
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
    alias_path = bundle_dir / "aliases.json"
    alias_path.write_text('{"energy_price": "electricity_pricing"}', encoding="utf-8")
    return manifest_path, alias_path


def _build_icharging_subminute_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "icharging_subminute"
    bundle_dir.mkdir()

    chargers = {
        "AC000001_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6, "allow_flex_when_ev": True},
        "AC000004_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6, "allow_flex_when_ev": True},
        "AC000007_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6, "allow_flex_when_ev": True},
        "AC000010_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6, "allow_flex_when_ev": True},
        "AC000013_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6, "allow_flex_when_ev": True},
    }
    action_names = list(chargers.keys())
    policy = {"default_actions": {name: 0.0 for name in action_names}, "rules": []}
    policy_path = bundle_dir / "policy_agent_0.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    manifest = {
        "manifest_version": 1,
        "metadata": {},
        "simulator": {},
        "training": {},
        "topology": {"num_agents": 1},
        "algorithm": {"name": "RuleBasedBreaker", "hyperparameters": {}},
        "environment": {
            "observation_names": [["timestamp"]],
            "encoders": [[{"type": "NoNormalization", "params": {}}]],
            "action_bounds": [[{"low": [0.0] * len(action_names), "high": [4.6] * len(action_names)}]],
            "action_names": action_names,
            "reward_function": {"name": "RewardFunction", "params": {}},
        },
        "agent": {
            "format": "rule_based",
            "artifacts": [
                {
                    "agent_index": 0,
                    "path": "policy_agent_0.json",
                    "format": "rule_based",
                    "config": {
                        "use_preprocessor": False,
                        "strategy": "icharging_breaker",
                        "control_interval_minutes": 0.0833,
                        "max_board_kw": 11.0,
                        "charger_limit_kw": 4.6,
                        "min_connected_kw": 1.6,
                        "chargers": chargers,
                        "line_limits": {"L1": {"limit_kw": 11.0}},
                        "vehicle_capacities": {"2": 50},
                    },
                }
            ],
        },
    }

    manifest_path = bundle_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _build_breaker_only_headroom_bundle(tmp_path: Path, per_phase_headroom_kw: float) -> Path:
    bundle_dir = tmp_path / "breaker_only_headroom"
    bundle_dir.mkdir()

    chargers = {
        "AC000001_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6},
        "AC000002_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6},
        "AC000003_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6},
        "AC000004_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6},
    }
    action_names = list(chargers.keys())
    policy = {"default_actions": {name: 0.0 for name in action_names}, "rules": []}
    policy_path = bundle_dir / "policy_agent_0.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    manifest = {
        "manifest_version": 1,
        "metadata": {},
        "simulator": {},
        "training": {},
        "topology": {"num_agents": 1},
        "algorithm": {"name": "RuleBasedBreaker", "hyperparameters": {}},
        "environment": {
            "observation_names": [["timestamp"]],
            "encoders": [[{"type": "NoNormalization", "params": {}}]],
            "action_bounds": [[{"low": [0.0] * len(action_names), "high": [4.6] * len(action_names)}]],
            "action_names": action_names,
            "reward_function": {"name": "RewardFunction", "params": {}},
        },
        "agent": {
            "format": "rule_based",
            "artifacts": [
                {
                    "agent_index": 0,
                    "path": "policy_agent_0.json",
                    "format": "rule_based",
                    "config": {
                        "use_preprocessor": False,
                        "strategy": "breaker_only",
                        "control_interval_minutes": 1,
                        "max_board_kw": BREAKER_ONLY_BASE_BOARD_LIMIT_KW,
                        "charger_limit_kw": 4.6,
                        "min_connected_kw": 1.6,
                        "per_phase_headroom_kw": per_phase_headroom_kw,
                        "chargers": chargers,
                        "line_limits": {
                            "L1": {"limit_kw": BREAKER_ONLY_BASE_BOARD_LIMIT_KW / 3.0},
                            "L2": {"limit_kw": BREAKER_ONLY_BASE_BOARD_LIMIT_KW / 3.0},
                            "L3": {"limit_kw": BREAKER_ONLY_BASE_BOARD_LIMIT_KW / 3.0},
                        },
                        "action_order": action_names,
                    },
                }
            ],
        },
    }

    manifest_path = bundle_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _build_icharging_headroom_bundle(tmp_path: Path, per_phase_headroom_kw: float) -> Path:
    bundle_dir = tmp_path / "icharging_headroom"
    bundle_dir.mkdir()

    chargers = {
        "AC000001_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6, "allow_flex_when_ev": True},
    }
    action_names = list(chargers.keys())
    policy = {"default_actions": {name: 0.0 for name in action_names}, "rules": []}
    policy_path = bundle_dir / "policy_agent_0.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    manifest = {
        "manifest_version": 1,
        "metadata": {},
        "simulator": {},
        "training": {},
        "topology": {"num_agents": 1},
        "algorithm": {"name": "RuleBasedBreaker", "hyperparameters": {}},
        "environment": {
            "observation_names": [["timestamp"]],
            "encoders": [[{"type": "NoNormalization", "params": {}}]],
            "action_bounds": [[{"low": [0.0] * len(action_names), "high": [4.6] * len(action_names)}]],
            "action_names": action_names,
            "reward_function": {"name": "RewardFunction", "params": {}},
        },
        "agent": {
            "format": "rule_based",
            "artifacts": [
                {
                    "agent_index": 0,
                    "path": "policy_agent_0.json",
                    "format": "rule_based",
                    "config": {
                        "use_preprocessor": False,
                        "strategy": "icharging_breaker",
                        "control_interval_minutes": 1,
                        "max_board_kw": ICHARGING_BOARD_LIMIT_KW,
                        "charger_limit_kw": 4.6,
                        "min_connected_kw": 1.6,
                        "per_phase_headroom_kw": per_phase_headroom_kw,
                        "chargers": chargers,
                        "line_limits": {
                            "L1": {"limit_kw": ICHARGING_BOARD_LIMIT_KW / 3.0},
                            "L2": {"limit_kw": ICHARGING_BOARD_LIMIT_KW / 3.0},
                            "L3": {"limit_kw": ICHARGING_BOARD_LIMIT_KW / 3.0},
                        },
                        "action_order": action_names,
                    },
                }
            ],
        },
    }

    manifest_path = bundle_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _build_icharging_pv_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "icharging_pv"
    bundle_dir.mkdir()

    base_board_kw = 9.0
    chargers = {
        "AC000001_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6, "allow_flex_when_ev": True},
        "AC000002_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6, "allow_flex_when_ev": True},
        "AC000003_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6, "allow_flex_when_ev": True},
        "AC000004_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6, "allow_flex_when_ev": True},
        "AC000005_1": {"line": "L1", "min_kw": 0.0, "max_kw": 4.6, "allow_flex_when_ev": True},
    }
    action_names = list(chargers.keys())
    policy = {"default_actions": {name: 0.0 for name in action_names}, "rules": []}
    policy_path = bundle_dir / "policy_agent_0.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    manifest = {
        "manifest_version": 1,
        "metadata": {},
        "simulator": {},
        "training": {},
        "topology": {"num_agents": 1},
        "algorithm": {"name": "RuleBasedBreaker", "hyperparameters": {}},
        "environment": {
            "observation_names": [["timestamp"]],
            "encoders": [[{"type": "NoNormalization", "params": {}}]],
            "action_bounds": [[{"low": [0.0] * len(action_names), "high": [4.6] * len(action_names)}]],
            "action_names": action_names,
            "reward_function": {"name": "RewardFunction", "params": {}},
        },
        "agent": {
            "format": "rule_based",
            "artifacts": [
                {
                    "agent_index": 0,
                    "path": "policy_agent_0.json",
                    "format": "rule_based",
                    "config": {
                        "use_preprocessor": False,
                        "strategy": "icharging_breaker",
                        "control_interval_minutes": 1,
                        "max_board_kw": base_board_kw,
                        "charger_limit_kw": 4.6,
                        "min_connected_kw": 1.6,
                        "chargers": chargers,
                        "line_limits": {
                            "L1": {"limit_kw": base_board_kw / 3.0},
                            "L2": {"limit_kw": base_board_kw / 3.0},
                            "L3": {"limit_kw": base_board_kw / 3.0},
                        },
                        "action_order": action_names,
                    },
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


def test_alias_mapping_applies_to_rule_based(tmp_path):
    manifest_path, alias_path = _build_rule_based_alias_bundle(tmp_path)
    if store.is_configured():
        store.unload()
    store.load(manifest_path, None, 0, alias_path)

    client = TestClient(app)
    try:
        response = client.post("/inference", json={"features": {"energy_price": 5}})
        assert response.status_code == 200
        body = response.json()
        assert body["actions"]["0"]["hvac"] == pytest.approx(0.9, rel=1e-6)
    finally:
        store.unload()


def test_icharging_replay_full_dataset():
    _replay_log_dataset(
        _load_icharging_bundle,
        lambda payload: ICHARGING_BOARD_LIMIT_KW + payload.get("solar_generation", 0.0),
        lambda payload: (ICHARGING_BOARD_LIMIT_KW + payload.get("solar_generation", 0.0)) / 3.0,
        lambda cid: 10.0 if cid == "BB000018_1" else 4.6,
    )


def test_breaker_only_replay_full_dataset():
    _replay_log_dataset(
        _load_breaker_only_bundle,
        lambda payload: BREAKER_ONLY_BASE_BOARD_LIMIT_KW + payload.get("solar_generation", 0.0),
        lambda payload: (BREAKER_ONLY_BASE_BOARD_LIMIT_KW + payload.get("solar_generation", 0.0)) / 3.0,
        lambda cid: 10.0 if cid == "BB000018_1" else 4.6,
    )


def test_onnx_icharging_sample_bundle():
    manifest_path = Path("examples/ichargingusecase_onnx/artifact_manifest.json")
    artifacts_dir = manifest_path.parent
    if store.is_configured():
        store.unload()
    store.load(manifest_path, artifacts_dir, 0)
    client = TestClient(app)
    all_chargers = [
        "AC000001_1","AC000002_1","AC000003_1","AC000004_1","AC000005_1","AC000006_1",
        "AC000007_1","AC000008_1","AC000009_1","AC000010_1","AC000011_1","AC000012_1",
        "AC000013_1","AC000014_1","ACEXT001_1","ACEXT002_1","ACEXT003_1","ACEXT004_1","BB000018_1",
    ]
    charging_sessions = {cid: {"power": 0.0, "electric_vehicle": ""} for cid in all_chargers}
    charging_sessions["AC000001_1"]["power"] = 1.0
    charging_sessions["AC000001_1"]["electric_vehicle"] = 1
    charging_sessions["AC000002_1"]["power"] = 2.0
    charging_sessions["AC000002_1"]["electric_vehicle"] = 2
    charging_sessions["AC000003_1"]["power"] = 3.0
    charging_sessions["AC000003_1"]["electric_vehicle"] = 3
    payload = {
        "timestamp": "2026-03-04T12:00:00Z",
        "observations": {
            "charging_sessions": charging_sessions,
        },
        "forecasts": {},
    }
    try:
        response = _post_payload(client, payload)
        assert response.status_code == 200
        body = response.json()
        assert "actions" in body and "0" in body["actions"]
    finally:
        store.unload()

def _replay_log_dataset(client_loader, board_limit_fn, line_limit_fn, max_kw_fn):
    client = client_loader()
    dataset_paths = [
        ICHARGING_BOAVISTA_WITH_FLEX_DIR / "datasets" / "dados_de_inferência_IC_11.11.2025_a_14.11.2025.json",
        ICHARGING_BOAVISTA_WITH_FLEX_DIR / "datasets" / "dados_de_inferência_IC_14.11.2025_a_18.11.2025.json",
    ]
    records = []
    for dataset_path in dataset_paths:
        if dataset_path.exists():
            records.extend(json.loads(dataset_path.read_text(encoding="utf-8")))

    line_groups = {
        "L1": ["ACEXT003_1", "ACEXT002_1", "AC000014_1", "AC000011_1", "AC000008_1", "AC000005_1", "AC000002_1", "ACEXT001_1", "BB000018_1"],
        "L2": ["ACEXT004_1", "AC000012_1", "AC000009_1", "AC000006_1", "AC000003_1", "BB000018_1"],
        "L3": ["AC000013_1", "AC000010_1", "AC000007_1", "AC000004_1", "AC000001_1", "BB000018_1"],
    }
    phase_counts = {"BB000018_1": 3}
    try:
        for record in records:
            payload = {
                "timestamp": record.get("timestamp") or record.get("timestamp.$date"),
                "non_shiftable_load": record.get("non_shiftable_load", 0.0),
                "solar_generation": record.get("solar_generation", 0.0),
                "energy_price": record.get("energy_price", 0.0),
                "charging_sessions": record.get("charging_sessions", {}),
                "electric_vehicles": record.get("electric_vehicles", {}),
                "pv_panels": record.get("pv_panels", {"PV01": {"energy": record.get("solar_generation", 0.0)}}),
            }
            resp = _post_payload(client, payload)
            assert resp.status_code == 200
            actions = resp.json()["actions"]["0"]
            connected = {
                cid for cid, info in payload["charging_sessions"].items() if str(info.get("electric_vehicle") or "").strip()
            }

            board_total = sum(actions.get(cid, 0.0) for cid in actions if cid in connected)
            board_limit = board_limit_fn(payload)
            assert board_total <= board_limit + 1e-6
            line_limit = line_limit_fn(payload)
            for chargers in line_groups.values():
                total = sum(
                    actions.get(cid, 0.0) / phase_counts.get(cid, 1)
                    for cid in chargers
                    if cid in connected
                )
                assert total <= line_limit + 1e-6

            for cid, session in payload["charging_sessions"].items():
                ev_id = str(session.get("electric_vehicle") or "").strip()
                action_val = actions.get(cid, 0.0)
                if ev_id and action_val > 1e-6:
                    assert action_val >= 1.6 - 1e-6
                assert actions.get(cid, 0.0) <= max_kw_fn(cid) + 1e-6
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


def test_manifest_validation_rejects_missing_fields(tmp_path):
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(json.dumps({"manifest_version": 1}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_manifest(manifest_path)


def test_bundle_missing_artifact_raises(tmp_path):
    bundle_dir = tmp_path / "missing_artifact"
    bundle_dir.mkdir()

    manifest = {
        "manifest_version": 1,
        "metadata": {},
        "simulator": {},
        "training": {},
        "topology": {"num_agents": 1},
        "algorithm": {"name": "OnnxPolicy", "hyperparameters": {}},
        "environment": {
            "observation_names": [["feature"]],
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
                    "path": "missing.onnx",
                    "format": "onnx",
                }
            ],
        },
    }

    manifest_path = bundle_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    if store.is_configured():
        store.unload()
    with pytest.raises(FileNotFoundError):
        store.load(manifest_path, bundle_dir, 0)


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


def test_file_logging_writes_to_disk(tmp_path):
    log_path = tmp_path / "logs" / "app.log"
    prior_log_file = settings.log_file
    prior_log_json = settings.log_json
    prior_log_level = settings.log_level
    prior_rotation = settings.log_file_rotation
    prior_retention = settings.log_file_retention

    settings.log_file = log_path
    settings.log_json = False
    settings.log_level = "INFO"
    settings.log_file_rotation = None
    settings.log_file_retention = None
    try:
        init_logging()
        get_logger().info("file.log.test")
        logger.complete()
        assert log_path.exists()
        assert "file.log.test" in log_path.read_text(encoding="utf-8")
    finally:
        settings.log_file = prior_log_file
        settings.log_json = prior_log_json
        settings.log_level = prior_log_level
        settings.log_file_rotation = prior_rotation
        settings.log_file_retention = prior_retention
        init_logging()


def test_breaker_allocation_strategy():
    manifest_path = ICHARGING_BOAVISTA_WITH_FLEX_DIR / "artifact_manifest.json"
    artifacts_dir = manifest_path.parent
    if store.is_configured():
        store.unload()
    store.load(manifest_path, artifacts_dir, 0, manifest_path.parent / "aliases.json")

    client = TestClient(app)
    charger_ids = [
        "AC000001_1", "AC000002_1", "AC000003_1", "AC000004_1", "AC000005_1",
        "AC000006_1", "AC000007_1", "AC000008_1", "AC000009_1", "AC000010_1",
        "AC000011_1", "AC000012_1", "AC000013_1", "AC000014_1",
        "ACEXT001_1", "ACEXT002_1", "ACEXT003_1", "ACEXT004_1", "BB000018_1"
    ]
    charging_sessions = {cid: {"power": 0.0, "electric_vehicle": ""} for cid in charger_ids}
    charging_sessions.update(
        {
            "AC000004_1": {"power": 1.8, "electric_vehicle": 11824},
            "AC000007_1": {"power": 1.8, "electric_vehicle": 11823},
            "AC000005_1": {"power": 1.8, "electric_vehicle": 11833},
            "AC000006_1": {"power": 1.8, "electric_vehicle": 11825},
            "AC000008_1": {"power": 1.8, "electric_vehicle": 11822},
            "AC000009_1": {"power": 1.8, "electric_vehicle": 11832},
            "AC000014_1": {"power": 1.8, "electric_vehicle": 11838},
            "ACEXT004_1": {"power": 1.8, "electric_vehicle": 11821},
        }
    )

    timestamp_iso = "2025-11-04T09:36:14Z"
    payload = {
        "timestamp": timestamp_iso,
        "non_shiftable_load": 12,
        "solar_generation": 2.8,
        "energy_price": 0.0,
        "charging_sessions": charging_sessions,
        "electric_vehicles": {
            "11824": {
                "SoC": 0.5,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
            "11823": {
                "SoC": 0.2,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
            "11833": {
                "SoC": 0.55,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
            "11825": {
                "SoC": 0.7,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
        },
        "pv_panels": {"PV01": {"energy": 2.8}},
        "grid_meters": {},
        "batteries": {},
    }

    line_groups = {
        "L1": [
            "ACEXT003_1",
            "ACEXT002_1",
            "AC000014_1",
            "AC000011_1",
            "AC000008_1",
            "AC000005_1",
            "AC000002_1",
            "ACEXT001_1",
            "BB000018_1",
        ],
        "L2": ["ACEXT004_1", "AC000012_1", "AC000009_1", "AC000006_1", "AC000003_1", "BB000018_1"],
        "L3": ["AC000013_1", "AC000010_1", "AC000007_1", "AC000004_1", "AC000001_1", "BB000018_1"],
    }

    try:
        response = _post_payload(client, payload)
        assert response.status_code == 200
        actions = response.json()["actions"]["0"]

        per_phase_limit = (ICHARGING_BOARD_LIMIT_KW + payload.get("solar_generation", 0.0)) / 3.0
        for line, chargers in line_groups.items():
            total = 0.0
            for cid in chargers:
                if not payload["charging_sessions"][cid]["electric_vehicle"]:
                    continue
                action = actions.get(cid, 0.0)
                if cid == "BB000018_1":
                    total += action / 3.0
                else:
                    total += action
            assert total <= per_phase_limit + 1e-6
            for cid in chargers:
                action = actions.get(cid, 0.0)
                max_kw = 10.0 if cid == "BB000018_1" else 4.6
                assert 0.0 <= action <= max_kw + 1e-6

        board_total = sum(
            actions.get(cid, 0.0) for cid in charger_ids if payload["charging_sessions"][cid]["electric_vehicle"]
        )
        board_limit = ICHARGING_BOARD_LIMIT_KW + payload.get("solar_generation", 0.0)
        assert board_total <= board_limit + 1e-6
    finally:
        store.unload()


def _load_icharging_bundle():
    manifest_path = ICHARGING_BOAVISTA_WITH_FLEX_DIR / "artifact_manifest.json"
    artifacts_dir = manifest_path.parent
    alias_path = artifacts_dir / "aliases.json"
    if store.is_configured():
        store.unload()
    store.load(manifest_path, artifacts_dir, 0, alias_path)
    return TestClient(app)


def _load_breaker_only_bundle():
    manifest_path = ICHARGING_BOAVISTA_WITHOUT_FLEX_DIR / "artifact_manifest.json"
    artifacts_dir = manifest_path.parent
    alias_path = artifacts_dir / "aliases.json"
    if store.is_configured():
        store.unload()
    store.load(manifest_path, artifacts_dir, 0, alias_path)
    return TestClient(app)


def test_breaker_allocation_prioritises_urgent_ev():
    client = _load_icharging_bundle()
    timestamp = "2025-11-04T09:00:00Z"
    charging_sessions = {
        "AC000004_1": {"power": 1.8, "electric_vehicle": 11824},
        "AC000007_1": {"power": 1.8, "electric_vehicle": 11823},
        "AC000005_1": {"power": 1.8, "electric_vehicle": 11833},
    }
    payload = {
        "timestamp": timestamp,
        "non_shiftable_load": 5.0,
        "solar_generation": 1.5,
        "energy_price": 0.0,
        "charging_sessions": charging_sessions,
        "electric_vehicles": {
            "11824": {
                "SoC": 0.2,
                "flexibility": {
                    "estimated_soc_at_departure": 0.95,
                    "estimated_time_at_departure": "2025-11-04T09:15:00Z",
                },
            },
            "11823": {
                "SoC": 0.6,
                "flexibility": {
                    "estimated_soc_at_departure": 0.8,
                    "estimated_time_at_departure": "2025-11-04T11:00:00Z",
                },
            },
        },
        "pv_panels": {"PV01": {"energy": 2.8}},
    }

    try:
        resp = _post_payload(client, payload)
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        assert actions["AC000004_1"] == pytest.approx(4.6, rel=1e-6)
        assert actions["AC000007_1"] <= 4.6 + 1e-6
        total = sum(actions[cid] for cid in charging_sessions)
        board_limit = ICHARGING_BOARD_LIMIT_KW + payload.get("solar_generation", 0.0)
        assert total <= board_limit + 1e-6
    finally:
        store.unload()


def test_nonflex_distribution_balances_phase():
    client = _load_icharging_bundle()
    charging_sessions = {
        "AC000004_1": {"power": 2.0, "electric_vehicle": 11824},
        "AC000007_1": {"power": 2.0, "electric_vehicle": 11823},
        "AC000013_1": {"power": 2.0, "electric_vehicle": 30001},
    }
    payload = {
        "timestamp": "2025-11-04T08:00:00Z",
        "non_shiftable_load": 0.0,
        "solar_generation": 0.0,
        "energy_price": 0.0,
        "charging_sessions": charging_sessions,
        "electric_vehicles": {
            "11824": {
                "SoC": 0.2,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
            "11823": {
                "SoC": 0.6,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
            "30001": {
                "SoC": 0.4,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
        },
        "pv_panels": {"PV01": {"energy": 0.0}},
    }

    try:
        resp = _post_payload(client, payload)
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        total_l3 = (
            actions["AC000004_1"]
            + actions["AC000007_1"]
            + actions["AC000013_1"]
        )
        per_phase_limit = (ICHARGING_BOARD_LIMIT_KW + payload.get("solar_generation", 0.0)) / 3.0
        assert total_l3 <= per_phase_limit + 1e-6
        for cid in ["AC000004_1", "AC000007_1", "AC000013_1"]:
            assert actions[cid] >= 1.6 - 1e-6
            assert actions[cid] <= 4.6 + 1e-6
        board_total = sum(actions.get(cid, 0.0) for cid in charging_sessions)
        board_limit = ICHARGING_BOARD_LIMIT_KW + payload.get("solar_generation", 0.0)
        assert board_total <= board_limit + 1e-6
    finally:
        store.unload()


def test_flexible_ev_consumes_solar_headroom():
    client = _load_icharging_bundle()
    charging_sessions = {
        "AC000004_1": {"power": 1.8, "electric_vehicle": 11824},
    }
    payload = {
        "timestamp": "2025-11-05T10:00:00Z",
        "non_shiftable_load": 0.0,
        "solar_generation": 5.0,
        "energy_price": 0.0,
        "charging_sessions": charging_sessions,
        "electric_vehicles": {
            "11824": {
                "SoC": 0.4,
                "flexibility": {
                    "estimated_soc_at_departure": 0.9,
                    "estimated_time_at_departure": "2025-11-05T12:00:00Z",
                },
            }
        },
        "pv_panels": {"PV01": {"energy": 5.0}},
    }

    try:
        resp = _post_payload(client, payload)
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        assert actions["AC000004_1"] > 4.0
        total = sum(actions[cid] for cid in charging_sessions)
        board_limit = ICHARGING_BOARD_LIMIT_KW + payload.get("solar_generation", 0.0)
        assert total <= board_limit + 1e-6
    finally:
        store.unload()


def test_nonflex_minimum_power_enforced():
    client = _load_icharging_bundle()
    charging_sessions = {
        "AC000005_1": {"power": 1.8, "electric_vehicle": 50001},
        "AC000006_1": {"power": 1.8, "electric_vehicle": 11825},
        "AC000009_1": {"power": 1.8, "electric_vehicle": 11832},
    }
    payload = {
        "timestamp": "2025-11-06T07:00:00Z",
        "non_shiftable_load": 0.0,
        "solar_generation": 0.0,
        "energy_price": 0.0,
        "charging_sessions": charging_sessions,
        "electric_vehicles": {
            "11825": {
                "SoC": 0.3,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
            "11832": {
                "SoC": 0.3,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
        },
        "pv_panels": {"PV01": {"energy": 0.0}},
    }

    try:
        resp = _post_payload(client, payload)
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        assert actions["AC000005_1"] >= 1.6 - 1e-6
        assert actions["AC000005_1"] <= 4.6 + 1e-6
        assert actions["AC000006_1"] <= 4.6 + 1e-6
        assert actions["AC000009_1"] <= 4.6 + 1e-6
        board_total = sum(actions.get(cid, 0.0) for cid in charging_sessions)
        board_limit = ICHARGING_BOARD_LIMIT_KW + payload.get("solar_generation", 0.0)
        assert board_total <= board_limit + 1e-6
    finally:
        store.unload()


def test_breaker_only_enforces_limits_and_minimums():
    client = _load_breaker_only_bundle()
    charging_sessions = {
        "AC000004_1": {"power": 2.5, "electric_vehicle": 11824},
        "AC000007_1": {"power": 2.5, "electric_vehicle": 11823},
        "AC000005_1": {"power": 2.5, "electric_vehicle": 11833},
        "AC000006_1": {"power": 2.5, "electric_vehicle": 11825},
        "AC000008_1": {"power": 2.5, "electric_vehicle": 11822},
        "AC000009_1": {"power": 2.5, "electric_vehicle": 11832},
    }
    payload = {
        "timestamp": "2025-11-07T07:00:00Z",
        "non_shiftable_load": 0.0,
        "solar_generation": 0.0,
        "energy_price": 0.0,
        "charging_sessions": charging_sessions,
        "pv_panels": {"PV01": {"energy": 0.0}},
    }

    try:
        resp = _post_payload(client, payload)
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        for cid in charging_sessions:
            assert actions[cid] == pytest.approx(4.6, rel=1e-6)
        board_total = sum(actions.get(cid, 0.0) for cid in charging_sessions)
        board_limit = BREAKER_ONLY_BASE_BOARD_LIMIT_KW + payload.get("solar_generation", 0.0)
        assert board_total <= board_limit + 1e-6
    finally:
        store.unload()


def test_breaker_only_skips_idle_sessions():
    client = _load_breaker_only_bundle()
    charging_sessions = {
        "AC000004_1": {"power": 0.1, "electric_vehicle": 11824},
        "AC000007_1": {"power": 2.0, "electric_vehicle": 11823},
    }
    payload = {
        "timestamp": "2025-11-07T09:00:00Z",
        "non_shiftable_load": 0.0,
        "solar_generation": 0.0,
        "energy_price": 0.0,
        "charging_sessions": charging_sessions,
        "pv_panels": {"PV01": {"energy": 0.0}},
    }

    try:
        resp = _post_payload(client, payload)
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        assert actions["AC000004_1"] >= 1.6 - 1e-6
        assert actions["AC000004_1"] <= 4.6 + 1e-6
        assert actions["AC000007_1"] >= 1.6 - 1e-6
        assert actions["AC000007_1"] <= 4.6 + 1e-6
    finally:
        store.unload()


def test_breaker_only_bb_tri_phase_limits():
    client = _load_breaker_only_bundle()
    payload = {
        "timestamp": "2025-11-07T10:00:00Z",
        "non_shiftable_load": 0.0,
        "solar_generation": 0.0,
        "energy_price": 0.0,
        "charging_sessions": {
            "BB000018_1": {"power": 0.0, "electric_vehicle": ""},
        },
        "pv_panels": {"PV01": {"energy": 0.0}},
    }

    try:
        resp = _post_payload(client, payload)
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        assert actions["BB000018_1"] == pytest.approx(8.0, rel=1e-6)

        payload["charging_sessions"]["BB000018_1"]["electric_vehicle"] = 9001
        resp = _post_payload(client, payload)
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        assert actions["BB000018_1"] >= 8.0 - 1e-6
        assert actions["BB000018_1"] <= 10.0 + 1e-6
    finally:
        store.unload()


def test_icharging_bb_tri_phase_limits():
    client = _load_icharging_bundle()
    payload = {
        "timestamp": "2025-11-07T10:00:00Z",
        "non_shiftable_load": 0.0,
        "solar_generation": 0.0,
        "energy_price": 0.0,
        "charging_sessions": {
            "BB000018_1": {"power": 0.0, "electric_vehicle": ""},
        },
        "electric_vehicles": {},
        "pv_panels": {"PV01": {"energy": 0.0}},
    }

    try:
        resp = _post_payload(client, payload)
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        assert actions["BB000018_1"] == pytest.approx(8.0, rel=1e-6)

        payload["charging_sessions"]["BB000018_1"]["electric_vehicle"] = 9001
        resp = _post_payload(client, payload)
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        assert actions["BB000018_1"] >= 8.0 - 1e-6
        assert actions["BB000018_1"] <= 10.0 + 1e-6
    finally:
        store.unload()


def test_bb_treated_as_flexible_with_targets():
    cfg = IchargingRuntimeConfig.from_dict(
        {
            "max_board_kw": ICHARGING_BOARD_LIMIT_KW,
            "charger_limit_kw": 4.6,
            "line_limits": {"L1": {"limit_kw": 18.333}, "L2": {"limit_kw": 18.333}, "L3": {"limit_kw": 18.333}},
            "chargers": {
                "BB000018_1": {
                    "phases": ["L1", "L2", "L3"],
                    "min_kw": 8.0,
                    "max_kw": 10.0,
                    "allow_flex_when_ev": True,
                }
            },
        }
    )
    runtime = IchargingBreakerRuntime(cfg)
    state = ChargerState(
        id="BB000018_1",
        min_kw=8.0,
        max_kw=10.0,
        line=None,
        phases=["L1", "L2", "L3"],
        n_phases=3,
        ev_id="9001",
        connected=True,
        allow_flex=True,
        session_power=0.0,
    )
    payload = {
        "timestamp": "2025-11-07T12:00:00Z",
        "electric_vehicles.9001.SoC": 0.2,
        "electric_vehicles.9001.flexibility.estimated_soc_at_departure": 0.9,
        "electric_vehicles.9001.flexibility.estimated_time_at_departure": "2025-11-07T14:00:00Z",
    }

    now = runtime._current_timestamp(payload)
    ok = runtime._populate_flexible_state(
        cfg,
        payload,
        state,
        now,
        control_minutes=1.0,
        min_minutes=1.0,
        whitelist=None,
    )
    assert ok is True
    assert state.flexible is True
    assert 8.0 - 1e-6 <= state.required_kw <= 10.0 + 1e-6


def test_breaker_only_phase_headroom_respects_pv(tmp_path):
    manifest_path = _build_breaker_only_headroom_bundle(tmp_path, per_phase_headroom_kw=1.0)
    if store.is_configured():
        store.unload()
    store.load(manifest_path, None, 0)

    client = TestClient(app)
    charging_sessions = {
        "AC000001_1": {"power": 0.0, "electric_vehicle": 1001},
        "AC000002_1": {"power": 0.0, "electric_vehicle": 1002},
        "AC000003_1": {"power": 0.0, "electric_vehicle": 1003},
        "AC000004_1": {"power": 0.0, "electric_vehicle": 1004},
    }
    payload = {
        "timestamp": "2025-11-07T11:00:00Z",
        "non_shiftable_load": 0.0,
        "solar_generation": 6.0,
        "energy_price": 0.0,
        "charging_sessions": charging_sessions,
    }

    try:
        resp = _post_payload(client, payload)
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        total = sum(actions.values())
        expected_limit = (BREAKER_ONLY_BASE_BOARD_LIMIT_KW + 6.0) / 3.0 - 1.0
        assert total <= expected_limit + 1e-6
    finally:
        store.unload()


def test_icharging_phase_headroom_respects_pv(tmp_path):
    manifest_path = _build_icharging_headroom_bundle(tmp_path, per_phase_headroom_kw=1.0)
    if store.is_configured():
        store.unload()
    store.load(manifest_path, None, 0)

    client = TestClient(app)
    charging_sessions = {
        "AC000001_1": {"power": 0.0, "electric_vehicle": 1001},
        "AC000002_1": {"power": 0.0, "electric_vehicle": 1002},
        "AC000003_1": {"power": 0.0, "electric_vehicle": 1003},
        "AC000004_1": {"power": 0.0, "electric_vehicle": 1004},
        "AC000005_1": {"power": 0.0, "electric_vehicle": 1005},
    }
    payload = {
        "timestamp": "2025-11-07T11:30:00Z",
        "non_shiftable_load": 0.0,
        "solar_generation": 6.0,
        "energy_price": 0.0,
        "charging_sessions": charging_sessions,
    }

    try:
        resp = _post_payload(client, payload)
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        total = sum(actions.values())
        expected_limit = (ICHARGING_BOARD_LIMIT_KW + 6.0) / 3.0 - 1.0
        assert total <= expected_limit + 1e-6
    finally:
        store.unload()


def test_icharging_pv_increases_flex_capacity(tmp_path):
    manifest_path = _build_icharging_pv_bundle(tmp_path)
    if store.is_configured():
        store.unload()
    store.load(manifest_path, None, 0)

    client = TestClient(app)

    def run_with_pv(pv_kw: float) -> float:
        charging_sessions = {
            "AC000001_1": {"power": 0.0, "electric_vehicle": 1001},
        }
        electric_vehicles = {
            "1001": {
                "SoC": 0.2,
                "flexibility": {
                    "estimated_soc_at_departure": 0.25,
                    "estimated_time_at_departure": "2025-11-07T14:00:00Z",
                },
            },
        }
        payload = {
            "timestamp": "2025-11-07T12:00:00Z",
            "non_shiftable_load": 0.0,
            "solar_generation": pv_kw,
            "energy_price": 0.0,
            "charging_sessions": charging_sessions,
            "electric_vehicles": electric_vehicles,
        }
        resp = _post_payload(client, payload)
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        return actions.get("AC000001_1", 0.0)

    try:
        total_no_pv = run_with_pv(0.0)
        total_with_pv = run_with_pv(6.0)
        assert total_no_pv <= 3.1
        assert total_with_pv >= total_no_pv + 1.0
    finally:
        store.unload()


def test_subminute_control_interval_allows_lower_required_kw(tmp_path):
    manifest_path = _build_icharging_subminute_bundle(tmp_path)
    if store.is_configured():
        store.unload()
    store.load(manifest_path, None, 0)
    client = TestClient(app)
    timestamp = "2025-11-04T09:00:00Z"
    departure = "2025-11-04T09:00:30Z"
    charging_sessions = {
        "AC000001_1": {"power": 0.0, "electric_vehicle": 11831},
        "AC000004_1": {"power": 0.0, "electric_vehicle": 2},
        "AC000007_1": {"power": 0.0, "electric_vehicle": 11832},
        "AC000010_1": {"power": 0.0, "electric_vehicle": 11833},
        "AC000013_1": {"power": 0.0, "electric_vehicle": 11834},
    }
    payload = {
        "timestamp": timestamp,
        "non_shiftable_load": 0.0,
        "solar_generation": 0.0,
        "energy_price": 0.0,
        "charging_sessions": charging_sessions,
        "electric_vehicles": {
            "2": {
                "SoC": 0.5,
                "flexibility": {
                    "estimated_soc_at_departure": 0.5005,
                    "estimated_time_at_departure": departure,
                },
            },
            "11831": {
                "SoC": 0.5,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
            "11832": {
                "SoC": 0.5,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
            "11833": {
                "SoC": 0.5,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
            "11834": {
                "SoC": 0.5,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
        },
    }

    try:
        resp = _post_payload(client, payload)
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        assert actions["AC000004_1"] < 4.6 - 1e-6
    finally:
        store.unload()


def test_icharging_multi_step_sequence():
    client = _load_icharging_bundle()

    def _required_kw(now_iso: str, soc: float, target: float, departure_iso: str, capacity_kwh: float = 75.0) -> float:
        now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        departure = datetime.fromisoformat(departure_iso.replace("Z", "+00:00"))
        minutes_remaining = max((departure - now).total_seconds() / 60.0, 15.0)
        energy_gap = max(target - soc, 0.0) * capacity_kwh
        if energy_gap <= 1e-6:
            return 0.0
        required = energy_gap / (minutes_remaining / 60.0)
        return min(required, 4.6)

    line_groups = {
        "L1": ["ACEXT003_1", "ACEXT002_1", "AC000014_1", "AC000011_1", "AC000008_1", "AC000005_1", "AC000002_1", "ACEXT001_1", "BB000018_1"],
        "L2": ["ACEXT004_1", "AC000012_1", "AC000009_1", "AC000006_1", "AC000003_1", "BB000018_1"],
        "L3": ["AC000013_1", "AC000010_1", "AC000007_1", "AC000004_1", "AC000001_1", "BB000018_1"],
    }

    scenarios = [
        {
            "description": "Day 1 morning, ample headroom, two flexible vehicles",
            "timestamp": "2025-11-04T08:00:00Z",
            "non_shiftable_load": 6.0,
            "solar_generation": 2.5,
            "energy_price": 0.0,
            "charging_sessions": {
                "AC000004_1": {"power": 4.0, "electric_vehicle": 11824},
                "AC000007_1": {"power": 2.5, "electric_vehicle": 11823},
                "AC000005_1": {"power": 2.0, "electric_vehicle": 11833},
                "AC000006_1": {"power": 2.0, "electric_vehicle": 11825},
            },
            "electric_vehicles": {
                "11824": {
                    "SoC": 0.25,
                    "flexibility": {
                        "estimated_soc_at_departure": 0.9,
                        "estimated_time_at_departure": "2025-11-04T09:00:00Z",
                    },
                },
                "11823": {
                    "SoC": 0.4,
                    "flexibility": {
                        "estimated_soc_at_departure": 0.85,
                        "estimated_time_at_departure": "2025-11-04T10:30:00Z",
                    },
                },
                "11833": {
                    "SoC": 0.6,
                    "flexibility": {
                        "estimated_soc_at_departure": -1,
                        "estimated_time_at_departure": "",
                    },
                },
                "11825": {
                    "SoC": 0.5,
                    "flexibility": {
                        "estimated_soc_at_departure": -1,
                        "estimated_time_at_departure": "",
                    },
                },
            },
            "expected_min": ["AC000004_1", "AC000007_1"],
        },
        {
            "description": "Day 1 evening, tight board limit forces shedding",
            "timestamp": "2025-11-04T18:00:00Z",
            "non_shiftable_load": 28.0,
            "solar_generation": 0.5,
            "energy_price": 0.1,
            "charging_sessions": {
                "AC000004_1": {"power": 4.6, "electric_vehicle": 11824},
                "AC000007_1": {"power": 4.6, "electric_vehicle": 11823},
                "AC000005_1": {"power": 2.5, "electric_vehicle": 11833},
                "AC000006_1": {"power": 2.5, "electric_vehicle": 11825},
            },
            "electric_vehicles": {
                "11824": {
                    "SoC": 0.5,
                    "flexibility": {
                        "estimated_soc_at_departure": 0.95,
                        "estimated_time_at_departure": "2025-11-04T18:30:00Z",
                    },
                },
                "11823": {
                    "SoC": 0.7,
                    "flexibility": {
                        "estimated_soc_at_departure": 0.9,
                        "estimated_time_at_departure": "2025-11-04T19:00:00Z",
                    },
                },
                "11833": {
                    "SoC": 0.6,
                    "flexibility": {
                        "estimated_soc_at_departure": -1,
                        "estimated_time_at_departure": "",
                    },
                },
                "11825": {
                    "SoC": 0.55,
                    "flexibility": {
                        "estimated_soc_at_departure": -1,
                        "estimated_time_at_departure": "",
                    },
                },
            },
        },
        {
            "description": "Day 2 midday, new flexible EV arrives",
            "timestamp": "2025-11-05T12:15:00Z",
            "non_shiftable_load": 10.0,
            "solar_generation": 4.0,
            "energy_price": 0.05,
            "charging_sessions": {
                "AC000004_1": {"power": 3.0, "electric_vehicle": 11824},
                "AC000007_1": {"power": 4.0, "electric_vehicle": 11823},
                "AC000005_1": {"power": 3.2, "electric_vehicle": 11833},
                "ACEXT004_1": {"power": 3.0, "electric_vehicle": 11821},
            },
            "electric_vehicles": {
                "11824": {
                    "SoC": 0.6,
                    "flexibility": {
                        "estimated_soc_at_departure": 0.85,
                        "estimated_time_at_departure": "2025-11-05T14:00:00Z",
                    },
                },
                "11823": {
                    "SoC": 0.3,
                    "flexibility": {
                        "estimated_soc_at_departure": 0.8,
                        "estimated_time_at_departure": "2025-11-05T13:00:00Z",
                    },
                },
                "11833": {
                    "SoC": 0.65,
                    "flexibility": {
                        "estimated_soc_at_departure": -1,
                        "estimated_time_at_departure": "",
                    },
                },
                "11821": {
                    "SoC": 0.2,
                    "flexibility": {
                        "estimated_soc_at_departure": 0.8,
                        "estimated_time_at_departure": "2025-11-05T16:00:00Z",
                    },
                },
            },
            "expected_min": ["AC000004_1", "AC000007_1", "ACEXT004_1"],
        },
        {
            "description": "Day 3 early morning, no flexibility data available",
            "timestamp": "2025-11-06T05:30:00Z",
            "non_shiftable_load": 4.0,
            "solar_generation": 0.0,
            "energy_price": 0.0,
            "charging_sessions": {
                "AC000004_1": {"power": 1.8, "electric_vehicle": 11824},
                "AC000007_1": {"power": 1.8, "electric_vehicle": 11823},
                "AC000005_1": {"power": 1.8, "electric_vehicle": 11833},
            },
            "electric_vehicles": {
                "11824": {
                    "SoC": 0.75,
                    "flexibility": {
                        "estimated_soc_at_departure": -1,
                        "estimated_time_at_departure": "",
                    },
                },
                "11823": {
                    "SoC": 0.8,
                    "flexibility": {
                        "estimated_soc_at_departure": -1,
                        "estimated_time_at_departure": "",
                    },
                },
                "11833": {
                    "SoC": 0.7,
                    "flexibility": {
                        "estimated_soc_at_departure": -1,
                        "estimated_time_at_departure": "",
                    },
                },
            },
        },
        {
            "description": "Day 3 afternoon, flexible EV with null SoC treated as non-flex",
            "timestamp": "2025-11-06T15:00:00Z",
            "non_shiftable_load": 0.0,
            "solar_generation": 1.0,
            "energy_price": 0.0,
            "charging_sessions": {
                "AC000004_1": {"power": 2.2, "electric_vehicle": 11824},
                "AC000007_1": {"power": 2.2, "electric_vehicle": 11823},
            },
            "electric_vehicles": {
                "11824": {
                    "SoC": None,
                    "flexibility": {
                        "estimated_soc_at_departure": 0.9,
                        "estimated_time_at_departure": "2025-11-06T17:00:00Z",
                    },
                },
                "11823": {
                    "SoC": 0.4,
                    "flexibility": {
                        "estimated_soc_at_departure": 0.8,
                        "estimated_time_at_departure": "2025-11-06T18:00:00Z",
                    },
                },
            },
            "pv_panels": {"PV01": {"energy": 1.0}},
        },
    ]

    try:
        for scenario in scenarios:
            payload = {
                "timestamp": scenario["timestamp"],
                "non_shiftable_load": scenario["non_shiftable_load"],
                "solar_generation": scenario["solar_generation"],
                "energy_price": scenario["energy_price"],
                "charging_sessions": scenario["charging_sessions"],
                "electric_vehicles": scenario["electric_vehicles"],
                "pv_panels": {"PV01": {"energy": scenario["solar_generation"]}},
            }
            resp = _post_payload(client, payload)
            assert resp.status_code == 200, scenario["description"]
            actions = resp.json()["actions"]["0"]

            connected = {
                cid for cid, info in scenario["charging_sessions"].items() if str(info.get("electric_vehicle") or "").strip()
            }

            board_total = sum(actions.get(cid, 0.0) for cid in connected)
            board_limit = ICHARGING_BOARD_LIMIT_KW + scenario["solar_generation"]
            assert board_total <= board_limit + 1e-6, scenario["description"]
            per_phase_limit = board_limit / 3.0
            for chargers in line_groups.values():
                total = 0.0
                for cid in chargers:
                    if cid not in connected:
                        continue
                    action = actions.get(cid, 0.0)
                    if cid == "BB000018_1":
                        total += action / 3.0
                    else:
                        total += action
                assert total <= per_phase_limit + 1e-6, scenario["description"]

            if "null SoC treated as non-flex" in scenario["description"]:
                for cid in scenario["charging_sessions"]:
                    assert actions[cid] >= 1.6 - 1e-6, scenario["description"]
                    assert actions[cid] <= 4.6 + 1e-6, scenario["description"]

            for cid in scenario.get("expected_min", []):
                session = scenario["charging_sessions"][cid]
                ev_id = str(session["electric_vehicle"])
                ev_data = scenario["electric_vehicles"].get(ev_id)
                if not ev_data:
                    continue
                required = _required_kw(
                    scenario["timestamp"],
                    ev_data["SoC"],
                    ev_data["flexibility"]["estimated_soc_at_departure"],
                    ev_data["flexibility"]["estimated_time_at_departure"],
                )
                assert actions[cid] + 1e-6 >= required, scenario["description"]

    finally:
        store.unload()
