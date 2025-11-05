import json
from datetime import datetime
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
    manifest_path = Path("examples/ichargingusecase_rule_based/artifact_manifest.json")
    artifacts_dir = manifest_path.parent
    if store.is_configured():
        store.unload()
    store.load(manifest_path, artifacts_dir, 0, manifest_path.parent / "aliases.json")

    client = TestClient(app)
    charger_ids = [
        "AC000001_1", "AC000002_1", "AC000003_1", "AC000004_1", "AC000005_1",
        "AC000006_1", "AC000007_1", "AC000008_1", "AC000009_1", "AC000010_1",
        "AC000011_1", "AC000012_1", "AC000013_1", "AC000014_1",
        "ACEXT001_1", "ACEXT002_1", "ACEXT003_1", "ACEXT004_1"
    ]
    charging_sessions = {cid: {"power": 0.0, "electric_vehicle": ""} for cid in charger_ids}
    charging_sessions.update(
        {
            "AC000004_1": {"power": 4.078, "electric_vehicle": 11824},
            "AC000007_1": {"power": 4.274, "electric_vehicle": 11823},
            "AC000005_1": {"power": 2.724, "electric_vehicle": 11833},
            "AC000006_1": {"power": 2.807, "electric_vehicle": 11825},
            "AC000008_1": {"power": 2.819, "electric_vehicle": 11822},
            "AC000009_1": {"power": 2.788, "electric_vehicle": 11832},
            "AC000014_1": {"power": 3.035, "electric_vehicle": 11838},
            "ACEXT004_1": {"power": 2.979, "electric_vehicle": 11821},
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

    try:
        response = client.post("/inference", json={"features": payload})
        assert response.status_code == 200
        actions = response.json()["actions"]["0"]

        active_chargers = {
            "AC000004_1": 4.078,
            "AC000007_1": 4.274,
            "AC000005_1": 2.724,
            "AC000006_1": 2.807,
            "AC000008_1": 2.819,
            "AC000009_1": 2.788,
            "AC000014_1": 3.035,
            "ACEXT004_1": 2.979,
        }
        for cid, baseline in active_chargers.items():
            assert actions[cid] == pytest.approx(baseline, rel=1e-6, abs=1e-6)

        inactive = set(charger_ids) - set(active_chargers.keys())
        for cid in inactive:
            assert actions[cid] == pytest.approx(0.0, abs=1e-6)

        base_load = max(payload["non_shiftable_load"] - payload["solar_generation"], 0.0)
        board_total = base_load + sum(actions.get(cid, 0.0) for cid in charger_ids)
        assert board_total <= 43.0 + 1e-6

        line_groups = {
            "L1": ["ACEXT003_1", "ACEXT002_1", "AC000014_1", "AC000011_1", "AC000008_1", "AC000005_1", "AC000002_1", "ACEXT001_1"],
            "L2": ["ACEXT004_1", "AC000012_1", "AC000009_1", "AC000006_1", "AC000003_1"],
            "L3": ["AC000013_1", "AC000010_1", "AC000007_1", "AC000004_1", "AC000001_1"],
        }
        for chargers in line_groups.values():
            total = sum(actions.get(cid, 0.0) for cid in chargers)
            assert total <= 11.0 + 1e-6
    finally:
        store.unload()


def _load_icharging_bundle():
    manifest_path = Path("examples/ichargingusecase_rule_based/artifact_manifest.json")
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
        "AC000004_1": {"power": 0.0, "electric_vehicle": 11824},
        "AC000007_1": {"power": 2.0, "electric_vehicle": 11823},
        "AC000005_1": {"power": 2.0, "electric_vehicle": 11833},
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
                    "capacity_kwh": 80,
                },
            },
            "11823": {
                "SoC": 0.6,
                "flexibility": {
                    "estimated_soc_at_departure": 0.8,
                    "estimated_time_at_departure": "2025-11-04T11:00:00Z",
                    "capacity_kwh": 70,
                },
            },
            "11833": {
                "SoC": 0.5,
                "flexibility": {
                    "estimated_soc_at_departure": 0.6,
                    "estimated_time_at_departure": "2025-11-04T12:00:00Z",
                },
            },
        },
        "pv_panels": {"PV01": {"energy": 2.8}},
        "grid_meters": {},
        "batteries": {},
    }

    try:
        resp = client.post("/inference", json={"features": payload})
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        assert actions["AC000004_1"] == pytest.approx(4.6, rel=1e-6)
        assert actions["AC000007_1"] <= 4.6 + 1e-6

        line_l3 = ["AC000013_1", "AC000010_1", "AC000007_1", "AC000004_1", "AC000001_1"]
        total_l3 = sum(actions.get(cid, 0.0) for cid in line_l3)
        assert total_l3 <= 11.0 + 1e-6
    finally:
        store.unload()


def test_breaker_allocation_keeps_non_flexible_baseline():
    client = _load_icharging_bundle()
    charging_sessions = {
        "AC000004_1": {"power": 3.0, "electric_vehicle": 11824},
        "AC000008_1": {"power": 2.5, "electric_vehicle": 20001},
    }
    payload = {
        "timestamp": "2025-11-04T08:00:00Z",
        "non_shiftable_load": 8.0,
        "solar_generation": 2.0,
        "energy_price": 0.0,
        "charging_sessions": charging_sessions,
        "electric_vehicles": {
            "11824": {
                "SoC": 0.3,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
            "20001": {
                "SoC": 0.4,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            },
        },
        "pv_panels": {"PV01": {"energy": 1.5}},
        "grid_meters": {},
        "batteries": {},
    }

    try:
        resp = client.post("/inference", json={"features": payload})
        assert resp.status_code == 200
        actions = resp.json()["actions"]["0"]
        assert actions["AC000008_1"] == pytest.approx(2.5, rel=1e-6)
        assert actions["AC000004_1"] <= 4.6 + 1e-6
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
        "L1": ["ACEXT003_1", "ACEXT002_1", "AC000014_1", "AC000011_1", "AC000008_1", "AC000005_1", "AC000002_1", "ACEXT001_1"],
        "L2": ["ACEXT004_1", "AC000012_1", "AC000009_1", "AC000006_1", "AC000003_1"],
        "L3": ["AC000013_1", "AC000010_1", "AC000007_1", "AC000004_1", "AC000001_1"],
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
            "baseline_equal": {"AC000005_1", "AC000006_1"},
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
            "expect_reduced": ["AC000004_1", "AC000007_1"],
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
                "AC000004_1": {"power": 2.0, "electric_vehicle": 11824},
                "AC000007_1": {"power": 1.5, "electric_vehicle": 11823},
                "AC000005_1": {"power": 1.0, "electric_vehicle": 11833},
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
            "baseline_equal": {"AC000004_1", "AC000007_1", "AC000005_1"},
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
            resp = client.post("/inference", json={"features": payload})
            assert resp.status_code == 200, scenario["description"]
            actions = resp.json()["actions"]["0"]

            base_load = max(scenario["non_shiftable_load"] - scenario["solar_generation"], 0.0)
            board_total = base_load + sum(actions.get(cid, 0.0) for cid in actions if not cid.startswith("b_"))
            assert board_total <= 43.0 + 1e-6, scenario["description"]
            for chargers in line_groups.values():
                total = sum(actions.get(cid, 0.0) for cid in chargers)
                assert total <= 11.0 + 1e-6, scenario["description"]

            for cid in scenario.get("baseline_equal", set()):
                baseline = scenario["charging_sessions"][cid]["power"]
                assert actions[cid] == pytest.approx(baseline, rel=1e-6, abs=1e-6), scenario["description"]

        for cid in scenario.get("expected_min", []):
            session = scenario["charging_sessions"][cid]
            ev_id = str(session["electric_vehicle"])
            ev_data = scenario["electric_vehicles"][ev_id]
            required = _required_kw(
                scenario["timestamp"],
                ev_data["SoC"],
                ev_data["flexibility"]["estimated_soc_at_departure"],
                ev_data["flexibility"]["estimated_time_at_departure"],
            )
            assert actions[cid] + 1e-6 >= required, scenario["description"]

        if scenario.get("expect_reduced"):
            reduced = False
            for cid in scenario["expect_reduced"]:
                baseline = scenario["charging_sessions"][cid]["power"]
                if actions[cid] < baseline - 1e-6:
                    reduced = True
                    break
            assert reduced, scenario["description"]
    finally:
        store.unload()
