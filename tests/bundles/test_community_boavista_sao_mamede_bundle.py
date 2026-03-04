from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store


BUNDLE_DIR = Path("examples/icharging_community_boavista_sao_mamede_with_virtual_battery")
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
SAO_MAMEDE_ACTIONS = {"BB000SMI_1", "BB000SMI_2", "virtual_battery_kw"}


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
            "timestamp": "2026-03-01T12:00:00Z",
            "sites": {
                "boavista": {
                    "timestamp": "2026-03-01T12:00:00Z",
                    "observations": {},
                    "forecasts": {},
                }
            },
        },
    }
    response = community_client.post("/inference", json=payload)
    assert response.status_code == 400


def test_missing_observations_returns_400(community_client):
    payload = {
        "agent_index": 1,
        "features": {
            "timestamp": "2026-03-01T12:00:00Z",
            "sites": {
                "sao_mamede": {
                    "timestamp": "2026-03-01T12:00:00Z",
                    "forecasts": {},
                }
            },
        },
    }
    response = community_client.post("/inference", json=payload)
    assert response.status_code == 400


def test_virtual_battery_action_limits_and_soc_guard(community_client):
    message = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))
    features = copy.deepcopy(message["features"])
    features["sites"]["sao_mamede"]["observations"]["virtual_battery"] = {"soc": 0.5}

    discharge_case = copy.deepcopy(features)
    discharge_case["community"]["target_net_import_kw"] = 100.0
    discharge_case["community"]["current_net_import_kw"] = 145.0
    discharge_resp = community_client.post(
        "/inference", json={"agent_index": 1, "features": discharge_case}
    )
    assert discharge_resp.status_code == 200
    discharge_kw = discharge_resp.json()["actions"]["1"]["virtual_battery_kw"]
    assert -15.0 <= discharge_kw <= 15.0
    assert discharge_kw <= 0.0

    charge_case = copy.deepcopy(features)
    charge_case["community"]["target_net_import_kw"] = 120.0
    charge_case["community"]["current_net_import_kw"] = 80.0
    charge_resp = community_client.post(
        "/inference", json={"agent_index": 1, "features": charge_case}
    )
    assert charge_resp.status_code == 200
    charge_kw = charge_resp.json()["actions"]["1"]["virtual_battery_kw"]
    assert -15.0 <= charge_kw <= 15.0
    assert charge_kw >= 0.0

    high_soc = copy.deepcopy(features)
    high_soc["sites"]["sao_mamede"]["observations"]["virtual_battery"]["soc"] = 1.0
    high_soc["community"]["target_net_import_kw"] = 120.0
    high_soc["community"]["current_net_import_kw"] = 80.0
    high_resp = community_client.post(
        "/inference", json={"agent_index": 1, "features": high_soc}
    )
    assert high_resp.status_code == 200
    assert high_resp.json()["actions"]["1"]["virtual_battery_kw"] == pytest.approx(0.0, rel=1e-6)

    low_soc = copy.deepcopy(features)
    low_soc["sites"]["sao_mamede"]["observations"]["virtual_battery"]["soc"] = 0.05
    low_soc["community"]["target_net_import_kw"] = 100.0
    low_soc["community"]["current_net_import_kw"] = 150.0
    low_resp = community_client.post(
        "/inference", json={"agent_index": 1, "features": low_soc}
    )
    assert low_resp.status_code == 200
    assert low_resp.json()["actions"]["1"]["virtual_battery_kw"] == pytest.approx(0.0, rel=1e-6)


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
