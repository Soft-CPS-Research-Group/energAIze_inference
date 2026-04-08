from __future__ import annotations

import copy
from pathlib import Path
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store


BUNDLE_DIR = Path("examples/icharging_sao_mamede_with_virtual_battery_community")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
ALIAS_PATH = BUNDLE_DIR / "aliases.json"
MESSAGE_PATH = BUNDLE_DIR / "exemplos_mensagem_SaoMamede_2303.json"

ACTION_BATTERY = "B01"
DECISION_INTERVAL_HOURS = 15.0 / 3600.0


def _kwh_for_interval(power_kw: float) -> float:
    return power_kw * DECISION_INTERVAL_HOURS


@pytest.fixture
def sao_mamede_with_battery_community_client():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0, ALIAS_PATH)
    client = TestClient(app)
    try:
        yield client
    finally:
        if store.is_configured():
            store.unload()


def _base_payload() -> dict:
    payload = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))[0]
    payload = copy.deepcopy(payload)
    payload["timestamp"] = "2026-03-01T10:00:00Z"
    payload["observations"]["charging_sessions"] = {
        "BB000SMI_1": {"power": 0.0, "electric_vehicle": ""},
        "BB000SMI_2": {"power": 0.0, "electric_vehicle": ""},
        "M1123089-5_1": {"power": 0.0, "electric_vehicle": ""},
        "M1123089-5_2": {"power": 0.0, "electric_vehicle": ""},
        "M1123089-6_1": {"power": 0.0, "electric_vehicle": ""},
        "M1123089-6_2": {"power": 0.0, "electric_vehicle": ""},
    }
    payload["observations"]["electric_vehicles"] = {}
    payload["observations"]["solar_generation"] = 0.0
    payload["observations"]["non_shiftable_load"] = 0.0
    payload["observations"]["batteries"] = {
        "B01": {
            "energy_in": 0.0,
            "energy_out": 0.0,
            "SoC": 0.70,
        }
    }
    payload["community"] = {
        "energy_in_total": 0.0,
        "energy_out_total": 0.0,
    }
    payload["forecasts"] = {}
    return payload


def _run(client: TestClient, payload: dict) -> dict[str, float]:
    response = client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    return response.json()["actions"]["0"]


def test_bundle_loads_sao_mamede_with_virtual_battery_community(
    sao_mamede_with_battery_community_client,
):
    pipeline = store.get_pipeline()
    assert pipeline.agent.strategy == "icharging_breaker"
    assert pipeline.agent._icharging_runtime is not None  # noqa: SLF001


def test_missing_required_community_fields_returns_400(
    sao_mamede_with_battery_community_client,
):
    missing_in = _base_payload()
    del missing_in["community"]["energy_in_total"]
    response = sao_mamede_with_battery_community_client.post(
        "/inference", json={"features": missing_in}
    )
    assert response.status_code == 400

    missing_out = _base_payload()
    del missing_out["community"]["energy_out_total"]
    response = sao_mamede_with_battery_community_client.post(
        "/inference", json={"features": missing_out}
    )
    assert response.status_code == 400


def test_community_deficit_forces_virtual_battery_discharge(
    sao_mamede_with_battery_community_client,
):
    payload = _base_payload()
    payload["community"]["energy_in_total"] = _kwh_for_interval(40.0)
    payload["community"]["energy_out_total"] = 0.0

    actions = _run(sao_mamede_with_battery_community_client, payload)
    assert actions[ACTION_BATTERY] < -0.1


def test_community_surplus_forces_virtual_battery_charge(
    sao_mamede_with_battery_community_client,
):
    payload = _base_payload()
    payload["community"]["energy_in_total"] = 0.0
    payload["community"]["energy_out_total"] = _kwh_for_interval(40.0)

    actions = _run(sao_mamede_with_battery_community_client, payload)
    assert actions[ACTION_BATTERY] > 0.1


def test_small_community_energy_interval_converts_to_kw(
    sao_mamede_with_battery_community_client,
):
    payload = _base_payload()
    payload["community"]["energy_in_total"] = _kwh_for_interval(1.774152)
    payload["community"]["energy_out_total"] = 0.0

    actions = _run(sao_mamede_with_battery_community_client, payload)
    assert actions[ACTION_BATTERY] < -0.1


def test_low_soc_does_not_force_aggressive_discharge_under_community_deficit(
    sao_mamede_with_battery_community_client,
):
    payload = _base_payload()
    payload["observations"]["batteries"]["B01"]["SoC"] = 0.11
    payload["community"]["energy_in_total"] = _kwh_for_interval(40.0)
    payload["community"]["energy_out_total"] = 0.0

    actions = _run(sao_mamede_with_battery_community_client, payload)
    assert actions[ACTION_BATTERY] >= -0.1


def test_low_soc_can_recharge_with_persistent_community_deficit_when_price_favors_charge(
    sao_mamede_with_battery_community_client,
):
    payload = _base_payload()
    payload["observations"]["batteries"]["B01"]["SoC"] = 0.15
    payload["community"]["energy_in_total"] = _kwh_for_interval(35.0)
    payload["community"]["energy_out_total"] = 0.0
    payload["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.05] + [0.30] * 95

    actions = _run(sao_mamede_with_battery_community_client, payload)
    assert actions[ACTION_BATTERY] > 0.1


def test_deadband_reduces_near_zero_dispatch_chattering(
    sao_mamede_with_battery_community_client,
):
    base = _base_payload()
    base["observations"]["batteries"]["B01"]["SoC"] = 0.50

    deficit_small = copy.deepcopy(base)
    deficit_small["community"]["energy_in_total"] = _kwh_for_interval(0.4)
    deficit_small["community"]["energy_out_total"] = 0.0

    surplus_small = copy.deepcopy(base)
    surplus_small["community"]["energy_in_total"] = 0.0
    surplus_small["community"]["energy_out_total"] = _kwh_for_interval(0.4)

    neutral = copy.deepcopy(base)

    a_deficit = _run(sao_mamede_with_battery_community_client, deficit_small)[ACTION_BATTERY]
    a_surplus = _run(sao_mamede_with_battery_community_client, surplus_small)[ACTION_BATTERY]
    a_neutral = _run(sao_mamede_with_battery_community_client, neutral)[ACTION_BATTERY]

    assert abs(a_deficit) <= 0.1
    assert abs(a_surplus) <= 0.1
    assert abs(a_neutral) <= 0.1
