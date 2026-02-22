from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.rbc.icharging import IchargingBreakerRuntime, IchargingRuntimeConfig
from app.state import store


BUNDLE_DIR = Path("examples/icharging_sao_mamede")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
ALIAS_PATH = BUNDLE_DIR / "aliases.json"
SEQUENCE_PATH = BUNDLE_DIR / "three_day_sequence.json"
MESSAGE_PATH = BUNDLE_DIR / "message_example.json"


def _run(client: TestClient, payload: dict) -> dict[str, float]:
    response = client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    return response.json()["actions"]["0"]


@pytest.fixture
def sao_mamede_client():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0, ALIAS_PATH)
    client = TestClient(app)
    try:
        yield client
    finally:
        if store.is_configured():
            store.unload()


def test_bundle_loads_sao_mamede(sao_mamede_client):
    pipeline = store.get_pipeline()
    assert pipeline.agent.strategy == "icharging_breaker"
    assert pipeline.agent._icharging_runtime is not None  # noqa: SLF001


def test_actions_contract_only_bb000smi(sao_mamede_client):
    payload = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))["features"]
    actions = _run(sao_mamede_client, payload)
    assert set(actions.keys()) == {"BB000SMI"}


def test_respects_pt_available_headroom(sao_mamede_client):
    payload = {
        "timestamp": "2026-02-22T09:00:00Z",
        "site": {"pt_available_kw": 18.0},
        "solar_generation": 10.0,
        "charging_sessions": {
            "BB000SMI": {"power": 0.0, "electric_vehicle": "SM_EV_01"},
            "M1123089-5": {"power": 4.0, "electric_vehicle": ""},
            "M1123089-6": {"power": 1.0, "electric_vehicle": ""},
        },
        "electric_vehicles": {
            "SM_EV_01": {
                "SoC": 0.40,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            }
        },
    }
    actions = _run(sao_mamede_client, payload)
    assert 0.0 <= actions["BB000SMI"] <= 18.0 + 1e-6


def test_missing_pt_headroom_fails_safe_to_zero(sao_mamede_client):
    payload = {
        "timestamp": "2026-02-22T10:00:00Z",
        "solar_generation": 14.0,
        "charging_sessions": {
            "BB000SMI": {"power": 0.0, "electric_vehicle": "SM_EV_01"},
            "M1123089-5": {"power": 7.0, "electric_vehicle": ""},
            "M1123089-6": {"power": 3.0, "electric_vehicle": ""},
        },
        "electric_vehicles": {
            "SM_EV_01": {
                "SoC": 0.38,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            }
        },
    }
    actions = _run(sao_mamede_client, payload)
    assert actions["BB000SMI"] == pytest.approx(0.0, rel=1e-6)


def test_no_ev_connected_returns_zero(sao_mamede_client):
    payload = {
        "timestamp": "2026-02-23T07:00:00Z",
        "site": {"pt_available_kw": 70.0},
        "solar_generation": 0.0,
        "charging_sessions": {
            "BB000SMI": {"power": 0.0, "electric_vehicle": ""},
            "M1123089-5": {"power": 4.0, "electric_vehicle": ""},
            "M1123089-6": {"power": 6.0, "electric_vehicle": ""},
        },
        "electric_vehicles": {},
    }
    actions = _run(sao_mamede_client, payload)
    assert actions["BB000SMI"] == pytest.approx(0.0, rel=1e-6)


def test_flexibility_is_applied_with_deadline(sao_mamede_client):
    payload = {
        "timestamp": "2026-02-22T11:30:00Z",
        "site": {"pt_available_kw": 35.0},
        "solar_generation": 17.5,
        "charging_sessions": {
            "BB000SMI": {"power": 0.0, "electric_vehicle": "SM_EV_02"},
            "M1123089-5": {"power": 0.0, "electric_vehicle": ""},
            "M1123089-6": {"power": 0.0, "electric_vehicle": ""},
        },
        "electric_vehicles": {
            "SM_EV_02": {
                "SoC": 0.25,
                "flexibility": {
                    "estimated_soc_at_departure": 0.80,
                    "estimated_time_at_departure": "2026-02-22T14:30:00Z",
                },
            }
        },
    }

    actions = _run(sao_mamede_client, payload)

    now = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
    departure = datetime.fromisoformat(
        payload["electric_vehicles"]["SM_EV_02"]["flexibility"]["estimated_time_at_departure"].replace(
            "Z", "+00:00"
        )
    )
    minutes_remaining = max((departure - now).total_seconds() / 60.0, 15.0)
    energy_gap_kwh = max(0.80 - 0.25, 0.0) * 60.0
    required_kw = min(50.0, energy_gap_kwh / (minutes_remaining / 60.0))

    assert actions["BB000SMI"] + 0.11 >= required_kw


def test_non_controllable_m_chargers_are_observational_only(sao_mamede_client):
    payload_a = {
        "timestamp": "2026-02-22T12:00:00Z",
        "site": {"pt_available_kw": 25.0},
        "solar_generation": 8.0,
        "charging_sessions": {
            "BB000SMI": {"power": 0.0, "electric_vehicle": "SM_EV_03"},
            "M1123089-5": {"power": 0.0, "electric_vehicle": ""},
            "M1123089-6": {"power": 0.0, "electric_vehicle": ""},
        },
        "electric_vehicles": {
            "SM_EV_03": {
                "SoC": 0.40,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            }
        },
    }
    payload_b = {
        "timestamp": "2026-02-22T12:00:00Z",
        "site": {"pt_available_kw": 25.0},
        "solar_generation": 8.0,
        "charging_sessions": {
            "BB000SMI": {"power": 0.0, "electric_vehicle": "SM_EV_03"},
            "M1123089-5": {"power": 7.4, "electric_vehicle": ""},
            "M1123089-6": {"power": 7.4, "electric_vehicle": ""},
        },
        "electric_vehicles": {
            "SM_EV_03": {
                "SoC": 0.40,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            }
        },
    }

    actions_a = _run(sao_mamede_client, payload_a)
    actions_b = _run(sao_mamede_client, payload_b)
    assert actions_a["BB000SMI"] == pytest.approx(actions_b["BB000SMI"], rel=1e-6)


def test_replay_sequence_sao_mamede(sao_mamede_client):
    scenarios = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))

    for scenario in scenarios:
        actions = _run(sao_mamede_client, scenario)
        action_value = actions["BB000SMI"]

        pt_available = scenario.get("site", {}).get("pt_available_kw", 0.0)
        ev_connected = bool(
            str(scenario.get("charging_sessions", {}).get("BB000SMI", {}).get("electric_vehicle", "")).strip()
        )

        if not ev_connected:
            assert action_value == pytest.approx(0.0, rel=1e-6), scenario["description"]
            continue

        assert 0.0 <= action_value <= 50.0 + 1e-6, scenario["description"]
        assert action_value <= float(pt_available) + 1e-6, scenario["description"]


def test_legacy_behavior_without_site_headroom_key():
    cfg = IchargingRuntimeConfig.from_dict(
        {
            "max_board_kw": 10.0,
            "charger_limit_kw": 50.0,
            "min_connected_kw": 0.0,
            "control_interval_minutes": 15.0,
            "chargers": {
                "BB000SMI": {
                    "line": "L1",
                    "min_kw": 0.0,
                    "max_kw": 50.0,
                    "allow_flex_when_ev": True
                }
            },
            "line_limits": {
                "L1": {"limit_kw": 10.0}
            }
        }
    )
    runtime = IchargingBreakerRuntime(cfg)

    payload = {
        "timestamp": "2026-02-22T12:00:00Z",
        "solar_generation": 5.0,
        "charging_sessions.BB000SMI.electric_vehicle": "SM_EV_99",
        "charging_sessions.BB000SMI.power": 0.0,
        "electric_vehicles.SM_EV_99.SoC": 0.40,
        "electric_vehicles.SM_EV_99.flexibility.estimated_soc_at_departure": -1,
        "electric_vehicles.SM_EV_99.flexibility.estimated_time_at_departure": "",
    }

    actions = runtime.allocate(payload)
    assert actions["BB000SMI"] == pytest.approx(15.0, rel=1e-6)
