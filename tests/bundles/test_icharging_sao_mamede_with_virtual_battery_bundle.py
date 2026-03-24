from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store


BUNDLE_DIR = Path("examples/icharging_sao_mamede_with_virtual_battery")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
ALIAS_PATH = BUNDLE_DIR / "aliases.json"
MESSAGE_PATH = BUNDLE_DIR / "exemplos_mensagem_SaoMamede_2303.json"


@pytest.fixture
def sao_mamede_with_battery_client():
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
    payload["observations"]["virtual_battery"] = {"soc": 0.5}
    payload["forecasts"] = {}
    return payload


def _run(client: TestClient, payload: dict) -> dict[str, float]:
    response = client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    return response.json()["actions"]["0"]


def test_actions_contract_includes_virtual_battery(sao_mamede_with_battery_client):
    actions = _run(sao_mamede_with_battery_client, _base_payload())
    assert set(actions.keys()) == {"BB000SMI_1", "BB000SMI_2", "virtual_battery_kw"}


def test_virtual_battery_solar_first_charge(sao_mamede_with_battery_client):
    payload = _base_payload()
    payload["observations"]["solar_generation"] = 20.0
    payload["observations"]["non_shiftable_load"] = 0.0
    payload["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.20] + [0.10] * 95
    actions = _run(sao_mamede_with_battery_client, payload)
    assert 0.0 <= actions["virtual_battery_kw"] <= 15.0


def test_virtual_battery_can_discharge_when_no_solar_and_price_unfavorable(
    sao_mamede_with_battery_client,
):
    payload = _base_payload()
    payload["observations"]["solar_generation"] = 0.0
    payload["observations"]["non_shiftable_load"] = 0.0
    payload["observations"]["charging_sessions"]["BB000SMI_2"] = {
        "power": 8.0,
        "electric_vehicle": "SM_EV_1",
    }
    payload["observations"]["electric_vehicles"]["SM_EV_1"] = {
        "SoC": 0.5,
        "flexibility": {
            "estimated_soc_at_departure": 0.8,
            "estimated_time_at_departure": "2026-03-01T12:00:00Z",
        },
    }
    payload["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.25] + [0.08] * 95
    actions = _run(sao_mamede_with_battery_client, payload)
    assert -15.0 <= actions["virtual_battery_kw"] <= 0.0


def test_virtual_battery_soc_guards(sao_mamede_with_battery_client):
    high_soc = _base_payload()
    high_soc["observations"]["virtual_battery"]["soc"] = 1.0
    high_soc["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.08] + [0.22] * 95
    high_actions = _run(sao_mamede_with_battery_client, high_soc)
    assert high_actions["virtual_battery_kw"] == pytest.approx(0.0, rel=1e-6)

    low_soc = _base_payload()
    low_soc["observations"]["virtual_battery"]["soc"] = 0.05
    low_soc["observations"]["solar_generation"] = 0.0
    low_soc["observations"]["non_shiftable_load"] = 0.0
    low_soc["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.25] + [0.08] * 95
    low_actions = _run(sao_mamede_with_battery_client, low_soc)
    assert low_actions["virtual_battery_kw"] >= 0.0
