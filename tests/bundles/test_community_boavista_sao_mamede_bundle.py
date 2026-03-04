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

    response = community_client.post(
        "/inference", json={"agent_index": 0, "features": message["features"]}
    )
    assert response.status_code == 200
    actions = response.json()["actions"]
    assert set(actions.keys()) == {"0", "1"}
    assert set(actions["0"].keys()) == BOAVISTA_ACTIONS
    assert set(actions["1"].keys()) == SAO_MAMEDE_ACTIONS


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
    discharge_case["sites"]["boavista"]["observations"]["solar_generation"] = 0.0
    discharge_case["sites"]["boavista"]["observations"]["non_shiftable_load"] = 14.0
    discharge_case["sites"]["sao_mamede"]["observations"]["solar_generation"] = 0.0
    discharge_case["sites"]["sao_mamede"]["observations"]["non_shiftable_load"] = 6.0
    discharge_case["community"]["price_signal"]["values"] = [0.24] + [0.10] * 95
    discharge_resp = community_client.post(
        "/inference", json={"agent_index": 1, "features": discharge_case}
    )
    assert discharge_resp.status_code == 200
    discharge_kw = discharge_resp.json()["actions"]["1"]["virtual_battery_kw"]
    assert -15.0 <= discharge_kw <= 15.0
    assert discharge_kw <= 0.0

    charge_case = copy.deepcopy(features)
    charge_case["sites"]["sao_mamede"]["observations"]["solar_generation"] = 35.0
    charge_case["sites"]["sao_mamede"]["observations"]["non_shiftable_load"] = 0.0
    charge_case["sites"]["boavista"]["observations"]["solar_generation"] = 35.0
    charge_case["sites"]["boavista"]["observations"]["non_shiftable_load"] = 0.0
    charge_case["sites"]["boavista"]["observations"]["charging_sessions"] = {}
    charge_case["sites"]["boavista"]["observations"]["electric_vehicles"] = {}
    charge_case["community"]["price_signal"]["values"] = [0.08] + [0.20] * 95
    charge_resp = community_client.post(
        "/inference", json={"agent_index": 1, "features": charge_case}
    )
    assert charge_resp.status_code == 200
    charge_kw = charge_resp.json()["actions"]["1"]["virtual_battery_kw"]
    assert -15.0 <= charge_kw <= 15.0
    assert charge_kw >= 0.0

    high_soc = copy.deepcopy(features)
    high_soc["sites"]["sao_mamede"]["observations"]["virtual_battery"]["soc"] = 1.0
    high_soc["sites"]["sao_mamede"]["observations"]["solar_generation"] = 35.0
    high_resp = community_client.post(
        "/inference", json={"agent_index": 1, "features": high_soc}
    )
    assert high_resp.status_code == 200
    assert -15.0 <= high_resp.json()["actions"]["1"]["virtual_battery_kw"] <= 0.0

    low_soc = copy.deepcopy(features)
    low_soc["sites"]["sao_mamede"]["observations"]["virtual_battery"]["soc"] = 0.05
    low_soc["sites"]["boavista"]["observations"]["solar_generation"] = 0.0
    low_soc["sites"]["boavista"]["observations"]["non_shiftable_load"] = 18.0
    low_soc["community"]["price_signal"]["values"] = [0.24] + [0.10] * 95
    low_resp = community_client.post(
        "/inference", json={"agent_index": 1, "features": low_soc}
    )
    assert low_resp.status_code == 200
    assert low_resp.json()["actions"]["1"]["virtual_battery_kw"] == pytest.approx(0.0, rel=1e-6)


def test_sequence_smoke_replay(community_client):
    sequence = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))
    assert sequence
    base_features = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))["features"]

    for step in sequence:
        features = copy.deepcopy(base_features)
        step_features = step.get("features", {})
        step_sites = step_features.get("sites", {})
        if isinstance(step_sites, dict):
            for site_key, site_payload in step_sites.items():
                if isinstance(site_payload, dict):
                    features.setdefault("sites", {})[site_key] = copy.deepcopy(site_payload)
        if isinstance(step_features.get("community"), dict):
            features["community"] = copy.deepcopy(step_features["community"])
        if step_features.get("timestamp") is not None:
            features["timestamp"] = step_features["timestamp"]
        response = community_client.post(
            "/inference",
            json={
                "agent_index": step["agent_index"],
                "features": features,
            },
        )
        assert response.status_code == 200
        assert set(response.json()["actions"].keys()) == {"0", "1"}
