from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store


BUNDLE_DIR = Path("examples/icharging_community_boavista_sao_mamede_rh1")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
MESSAGE_PATH = BUNDLE_DIR / "community_message_example.json"
SEQUENCE_PATH = BUNDLE_DIR / "community_sequence.json"

BOAVISTA_ACTIONS = {
    "AC000001_1",
    "AC000002_1",
    "AC000003_1",
    "AC000004_1",
    "AC000005_1",
    "AC000006_1",
    "AC000007_1",
    "AC000008_1",
    "AC000009_1",
    "AC000010_1",
    "AC000011_1",
    "AC000012_1",
    "AC000013_1",
    "AC000014_1",
    "ACEXT001_1",
    "ACEXT002_1",
    "ACEXT003_1",
    "ACEXT004_1",
    "BB000018_1",
}
SAO_MAMEDE_ACTIONS = {"BB000SMI", "virtual_battery_kw"}
RH1_ACTIONS = {"ev_charge_kw", "battery_kw", "cooling_kw", "dhw_heater_kw"}


@pytest.fixture
def community_rh1_client():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0)
    client = TestClient(app)
    try:
        yield client
    finally:
        if store.is_configured():
            store.unload()


def test_bundle_loads_community_boavista_sao_mamede_rh1(community_rh1_client):
    record = store.get_record()
    assert record is not None
    assert record.default_agent_index == 0
    assert set(record.loaded_agent_indices) == {0, 1, 2}


def test_action_contract_per_agent(community_rh1_client):
    message = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))

    response_0 = community_rh1_client.post(
        "/inference", json={"agent_index": 0, "features": message["features"]}
    )
    assert response_0.status_code == 200
    actions_0 = response_0.json()["actions"]["0"]
    assert set(actions_0.keys()) == BOAVISTA_ACTIONS

    response_1 = community_rh1_client.post(
        "/inference", json={"agent_index": 1, "features": message["features"]}
    )
    assert response_1.status_code == 200
    actions_1 = response_1.json()["actions"]["1"]
    assert set(actions_1.keys()) == SAO_MAMEDE_ACTIONS

    response_2 = community_rh1_client.post(
        "/inference", json={"agent_index": 2, "features": message["features"]}
    )
    assert response_2.status_code == 200
    actions_2 = response_2.json()["actions"]["2"]
    assert set(actions_2.keys()) == RH1_ACTIONS


def test_missing_rh1_site_payload_returns_400(community_rh1_client):
    payload = {
        "agent_index": 2,
        "features": {
            "timestamp": "2026-02-22T12:00:00Z",
            "sites": {
                "boavista": {
                    "non_shiftable_load": 5.0,
                    "solar_generation": 1.0,
                    "energy_price": 0.1,
                    "charging_sessions": {},
                    "electric_vehicles": {},
                },
                "sao_mamede": {
                    "site": {"pt_available_kw": 50.0},
                    "charging_sessions": {"BB000SMI": {"power": 0.0, "electric_vehicle": ""}},
                    "electric_vehicles": {},
                    "electrical_storage": {"soc": 0.5},
                    "virtual_battery": {"setpoint_kw": 0.0},
                },
            },
        },
    }
    response = community_rh1_client.post("/inference", json=payload)
    assert response.status_code == 400


def test_virtual_battery_action_present_and_bounded(community_rh1_client):
    payload = {
        "agent_index": 1,
        "features": {
            "timestamp": "2026-02-22T12:00:00Z",
            "sites": {
                "sao_mamede": {
                    "site": {"pt_available_kw": 100.0},
                    "solar_generation": 14.0,
                    "charging_sessions": {
                        "BB000SMI": {"power": 0.0, "electric_vehicle": ""}
                    },
                    "electric_vehicles": {},
                    "electrical_storage": {"soc": 0.5},
                    "virtual_battery": {"setpoint_kw": 40.0},
                }
            },
        },
    }

    response = community_rh1_client.post("/inference", json=payload)
    assert response.status_code == 200
    actions = response.json()["actions"]["1"]
    assert "virtual_battery_kw" in actions
    assert -30.0 <= actions["virtual_battery_kw"] <= 30.0


def test_sequence_smoke_replay(community_rh1_client):
    sequence = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))
    assert sequence

    for step in sequence:
        response = community_rh1_client.post(
            "/inference",
            json={
                "agent_index": step["agent_index"],
                "features": step["features"],
            },
        )
        assert response.status_code == 200
