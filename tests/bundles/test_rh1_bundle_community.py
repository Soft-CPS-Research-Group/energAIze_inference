from __future__ import annotations

import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store


BUNDLE_DIR = Path("examples/rh1_bundle_community")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
ALIAS_PATH = BUNDLE_DIR / "aliases.json"
ACTION_EV = "EVC01"
ACTION_BATTERY = "B01"
DECISION_INTERVAL_HOURS = 15.0 / 3600.0


def _kwh_for_interval(power_kw: float) -> float:
    return power_kw * DECISION_INTERVAL_HOURS


def _price_curve(now: float, h1: float, h2: float, h6: float, h12: float, h24: float) -> dict:
    values = [now] * 96
    values[4] = h1
    values[8] = h2
    values[24] = h6
    values[48] = h12
    values[95] = h24
    return {
        "values": values,
        "measurement_unit": "€/kWh",
        "frequency_seconds": 900,
        "horizon_seconds": 86400,
    }


@pytest.fixture
def rh1_community_client():
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
    return {
        "timestamp": "2026-03-01T10:00:00Z",
        "community": {
            "energy_in_total": 0.0,
            "energy_out_total": 0.0,
        },
        "observations": {
            "non_shiftable_load": 0.2,
            "solar_generation": 0.0,
            "energy_price": _price_curve(0.05, 0.25, 0.25, 0.25, 0.25, 0.25),
            "batteries": {"B01": {"SoC": 0.6}},
            "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
            "electric_vehicles": {},
        },
        "forecasts": {},
    }


def _run(client: TestClient, payload: dict) -> dict[str, float]:
    response = client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    return response.json()["actions"]["0"]


def _net_grid_kw(payload: dict, actions: dict[str, float]) -> float:
    obs = payload["observations"]
    return (
        float(obs.get("non_shiftable_load", 0.0))
        + float(actions.get(ACTION_EV, 0.0))
        + float(actions.get(ACTION_BATTERY, 0.0))
        - float(obs.get("solar_generation", 0.0))
    )


def test_bundle_loads_rh1_community(rh1_community_client):
    pipeline = store.get_pipeline()
    assert pipeline.agent.strategy == "rh1_house_rbc_v1"


def test_missing_required_community_fields_returns_400(rh1_community_client):
    missing_in = _base_payload()
    del missing_in["community"]["energy_in_total"]
    response = rh1_community_client.post("/inference", json={"features": missing_in})
    assert response.status_code == 400

    missing_out = _base_payload()
    del missing_out["community"]["energy_out_total"]
    response = rh1_community_client.post("/inference", json={"features": missing_out})
    assert response.status_code == 400


def test_community_deficit_bias_reduces_import_vs_surplus(rh1_community_client):
    deficit = _base_payload()
    deficit["community"]["energy_in_total"] = _kwh_for_interval(30.0)
    deficit["community"]["energy_out_total"] = 0.0

    surplus = copy.deepcopy(deficit)
    surplus["community"]["energy_in_total"] = 0.0
    surplus["community"]["energy_out_total"] = _kwh_for_interval(30.0)

    deficit_actions = _run(rh1_community_client, deficit)
    surplus_actions = _run(rh1_community_client, surplus)

    deficit_net = _net_grid_kw(deficit, deficit_actions)
    surplus_net = _net_grid_kw(surplus, surplus_actions)

    assert deficit_actions[ACTION_BATTERY] <= surplus_actions[ACTION_BATTERY]
    assert deficit_net <= surplus_net


def test_small_community_energy_interval_converts_to_kw(rh1_community_client):
    neutral = _base_payload()
    neutral_actions = _run(rh1_community_client, neutral)

    deficit_small = _base_payload()
    deficit_small["community"]["energy_in_total"] = _kwh_for_interval(1.774152)
    deficit_small["community"]["energy_out_total"] = 0.0
    deficit_small_actions = _run(rh1_community_client, deficit_small)

    assert deficit_small_actions[ACTION_BATTERY] < neutral_actions[ACTION_BATTERY]
