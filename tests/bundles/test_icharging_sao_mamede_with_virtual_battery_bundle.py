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
MESSAGE_PATH = BUNDLE_DIR / "exemplos_mensagem_SaoMamede.json"


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


def test_virtual_battery_price_arbitrage_direction(sao_mamede_with_battery_client):
    charge_case = _base_payload()
    charge_case["observations"]["energy_price"]["values"] = [0.08] + [0.22] * 95
    charge_actions = _run(sao_mamede_with_battery_client, charge_case)
    assert 0.0 <= charge_actions["virtual_battery_kw"] <= 15.0

    discharge_case = _base_payload()
    discharge_case["observations"]["energy_price"]["values"] = [0.25] + [0.08] * 95
    discharge_actions = _run(sao_mamede_with_battery_client, discharge_case)
    assert -15.0 <= discharge_actions["virtual_battery_kw"] <= 0.0


def test_virtual_battery_soc_guards(sao_mamede_with_battery_client):
    high_soc = _base_payload()
    high_soc["observations"]["virtual_battery"]["soc"] = 1.0
    high_soc["observations"]["energy_price"]["values"] = [0.08] + [0.22] * 95
    high_actions = _run(sao_mamede_with_battery_client, high_soc)
    assert high_actions["virtual_battery_kw"] == pytest.approx(0.0, rel=1e-6)

    low_soc = _base_payload()
    low_soc["observations"]["virtual_battery"]["soc"] = 0.05
    low_soc["observations"]["energy_price"]["values"] = [0.25] + [0.08] * 95
    low_actions = _run(sao_mamede_with_battery_client, low_soc)
    assert low_actions["virtual_battery_kw"] == pytest.approx(0.0, rel=1e-6)
