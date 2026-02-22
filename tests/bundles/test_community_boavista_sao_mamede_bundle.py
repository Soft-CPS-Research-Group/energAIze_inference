from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store


BUNDLE_DIR = Path("examples/icharging_community_boavista_sao_mamede")
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


@pytest.fixture
def community_client():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0)
    client = TestClient(app)
    try:
        yield client
    finally:
        if store.is_configured():
            store.unload()


def test_bundle_loads_community_boavista_sao_mamede(community_client):
    record = store.get_record()
    assert record is not None
    assert record.default_agent_index == 0
    assert set(record.loaded_agent_indices) == {0, 1}


def test_action_contract_per_agent(community_client):
    message = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))

    response_boavista = community_client.post(
        "/inference", json={"agent_index": 0, "features": message["features"]}
    )
    assert response_boavista.status_code == 200
    actions_0 = response_boavista.json()["actions"]["0"]
    assert set(actions_0.keys()) == BOAVISTA_ACTIONS

    response_sao_mamede = community_client.post(
        "/inference", json={"agent_index": 1, "features": message["features"]}
    )
    assert response_sao_mamede.status_code == 200
    actions_1 = response_sao_mamede.json()["actions"]["1"]
    assert set(actions_1.keys()) == SAO_MAMEDE_ACTIONS


def test_missing_site_payload_returns_400(community_client):
    payload = {
        "agent_index": 1,
        "features": {
            "timestamp": "2026-02-22T12:00:00Z",
            "sites": {
                "boavista": {
                    "non_shiftable_load": 4.0,
                    "solar_generation": 1.0,
                    "energy_price": 0.1,
                    "charging_sessions": {},
                    "electric_vehicles": {},
                }
            },
        },
    }
    response = community_client.post("/inference", json=payload)
    assert response.status_code == 400


def test_virtual_battery_action_limits_and_soc_guard(community_client):
    base_features = {
        "timestamp": "2026-02-22T13:00:00Z",
        "sites": {
            "sao_mamede": {
                "site": {"pt_available_kw": 120.0},
                "solar_generation": 10.0,
                "charging_sessions": {
                    "BB000SMI": {"power": 0.0, "electric_vehicle": ""}
                },
                "electric_vehicles": {},
                "electrical_storage": {"soc": 0.5},
                "virtual_battery": {"setpoint_kw": 100.0},
            }
        },
    }

    charge_resp = community_client.post(
        "/inference", json={"agent_index": 1, "features": base_features}
    )
    assert charge_resp.status_code == 200
    charge_actions = charge_resp.json()["actions"]["1"]
    assert 0.0 <= charge_actions["virtual_battery_kw"] <= 30.0

    low_headroom_features = json.loads(json.dumps(base_features))
    low_headroom_features["sites"]["sao_mamede"]["site"]["pt_available_kw"] = 5.0
    low_headroom_resp = community_client.post(
        "/inference", json={"agent_index": 1, "features": low_headroom_features}
    )
    assert low_headroom_resp.status_code == 200
    low_headroom_actions = low_headroom_resp.json()["actions"]["1"]
    assert 0.0 <= low_headroom_actions["virtual_battery_kw"] <= 5.0 + 1e-6

    high_soc_features = json.loads(json.dumps(base_features))
    high_soc_features["sites"]["sao_mamede"]["electrical_storage"]["soc"] = 0.95
    high_soc_features["sites"]["sao_mamede"]["virtual_battery"]["setpoint_kw"] = 10.0
    high_soc_resp = community_client.post(
        "/inference", json={"agent_index": 1, "features": high_soc_features}
    )
    assert high_soc_resp.status_code == 200
    high_soc_actions = high_soc_resp.json()["actions"]["1"]
    assert high_soc_actions["virtual_battery_kw"] == pytest.approx(0.0, rel=1e-6)

    low_soc_features = json.loads(json.dumps(base_features))
    low_soc_features["sites"]["sao_mamede"]["electrical_storage"]["soc"] = 0.05
    low_soc_features["sites"]["sao_mamede"]["virtual_battery"]["setpoint_kw"] = -10.0
    low_soc_resp = community_client.post(
        "/inference", json={"agent_index": 1, "features": low_soc_features}
    )
    assert low_soc_resp.status_code == 200
    low_soc_actions = low_soc_resp.json()["actions"]["1"]
    assert low_soc_actions["virtual_battery_kw"] == pytest.approx(0.0, rel=1e-6)


def test_sequence_smoke_replay(community_client):
    sequence = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))
    assert sequence

    for step in sequence:
        response = community_client.post(
            "/inference",
            json={
                "agent_index": step["agent_index"],
                "features": step["features"],
            },
        )
        assert response.status_code == 200
