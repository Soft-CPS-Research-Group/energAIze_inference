from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store
from app.utils.flatten import flatten_payload

pytestmark = pytest.mark.skip(reason="Community em standby")


BUNDLE_DIR = Path("examples/icharging_community_boavista_sao_mamede_rh1_with_virtual_battery")
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
RH1_ACTIONS = {"ev_charge_kw", "battery_kw"}



def _inference(client: TestClient, agent_index: int, features: dict):
    return client.post("/inference", json={"agent_index": agent_index, "features": features})



def _load_message() -> dict:
    return json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))



def _load_sequence() -> list[dict]:
    return json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))


def _normalize_ev_id(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"none", "nan", "null"}:
        return ""
    if text in {"0", "0.0"}:
        return ""
    return text


def _normalize_site_payload(site_payload: dict, top_timestamp, top_community) -> dict:
    selected = copy.deepcopy(site_payload)
    if isinstance(top_community, dict) and "community" not in selected:
        selected["community"] = copy.deepcopy(top_community)
    if "timestamp" not in selected and top_timestamp is not None:
        selected["timestamp"] = top_timestamp
    observations = selected.get("observations")
    if not isinstance(observations, dict):
        return {}
    normalized = dict(observations)
    if isinstance(selected.get("community"), dict):
        normalized["community"] = copy.deepcopy(selected["community"])
    if "timestamp" not in normalized and selected.get("timestamp") is not None:
        normalized["timestamp"] = selected["timestamp"]
    return normalized


def _site_net(features: dict, actions: dict) -> float:
    total = 0.0
    sites = features["sites"]

    for site_key, site_payload in sites.items():
        obs = site_payload["observations"]
        non_shiftable = float(obs.get("non_shiftable_load", 0.0))
        solar = float(obs.get("solar_generation", 0.0))
        if site_key == "rh1":
            ev_kw = float(actions["2"].get("ev_charge_kw", 0.0))
            battery_kw = float(actions["2"].get("battery_kw", 0.0))
            total += non_shiftable + ev_kw + battery_kw - solar
            continue

        site_actions = actions["0"] if site_key == "boavista" else actions["1"]
        chargers = obs.get("charging_sessions", {})
        connected_kw = 0.0
        for charger_id, session in chargers.items():
            ev_id = _normalize_ev_id((session or {}).get("electric_vehicle"))
            if not ev_id:
                continue
            connected_kw += float(site_actions.get(charger_id, 0.0))
        battery_kw = float(site_actions.get("virtual_battery_kw", 0.0))
        total += non_shiftable + connected_kw + battery_kw - solar
    return total


def _external_cost(total_net_kw: float, price: float, export_factor: float = 0.8) -> float:
    return max(total_net_kw, 0.0) * price - max(-total_net_kw, 0.0) * price * export_factor


def _baseline_actions_from_local_pipelines(features: dict) -> dict[str, dict[str, float]]:
    record = store.get_record()
    assert record is not None
    actions: dict[str, dict[str, float]] = {}
    top_timestamp = features.get("timestamp")
    top_community = features.get("community")
    for agent_index in record.loaded_agent_indices:
        pipeline = record.pipelines[agent_index]
        cfg = pipeline.manifest.get_artifact(agent_index).config or {}
        site_key = cfg.get("input_site_key")
        if not site_key:
            continue
        site_payload = features["sites"][site_key]
        normalized = _normalize_site_payload(site_payload, top_timestamp, top_community)
        flattened = flatten_payload(normalized)
        out = pipeline.inference(flattened)
        actions[str(agent_index)] = out[str(agent_index)]
    return actions


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
    message = _load_message()

    response = _inference(community_rh1_client, 0, message["features"])
    assert response.status_code == 200
    actions = response.json()["actions"]
    assert set(actions.keys()) == {"0", "1", "2"}
    assert set(actions["0"].keys()) == BOAVISTA_ACTIONS
    assert set(actions["1"].keys()) == SAO_MAMEDE_ACTIONS
    assert set(actions["2"].keys()) == RH1_ACTIONS



def test_missing_rh1_site_payload_returns_400(community_rh1_client):
    payload = {
        "agent_index": 2,
        "features": {
            "timestamp": "2026-03-01T12:00:00Z",
            "sites": {
                "boavista": {
                    "timestamp": "2026-03-01T12:00:00Z",
                    "observations": {},
                    "forecasts": {},
                },
                "sao_mamede": {
                    "timestamp": "2026-03-01T12:00:00Z",
                    "observations": {},
                    "forecasts": {},
                },
            },
        },
    }
    response = community_rh1_client.post("/inference", json=payload)
    assert response.status_code == 400



def test_virtual_battery_action_present_and_bounded(community_rh1_client):
    message = _load_message()
    features = copy.deepcopy(message["features"])
    features["sites"]["sao_mamede"]["observations"]["virtual_battery"] = {"soc": 0.5}
    features["sites"]["boavista"]["observations"]["solar_generation"] = 0.0
    features["sites"]["boavista"]["observations"]["non_shiftable_load"] = 16.0
    features["sites"]["sao_mamede"]["observations"]["solar_generation"] = 0.0
    features["sites"]["sao_mamede"]["observations"]["non_shiftable_load"] = 7.0
    features["community"]["price_signal"]["values"] = [0.24] + [0.10] * 95

    response = _inference(community_rh1_client, 1, features)
    assert response.status_code == 200
    actions = response.json()["actions"]
    assert set(actions.keys()) == {"0", "1", "2"}
    assert "virtual_battery_kw" in actions["1"]
    assert -15.0 <= actions["1"]["virtual_battery_kw"] <= 15.0
    assert actions["1"]["virtual_battery_kw"] <= 0.0


def test_community_optimization_reduces_external_cost_vs_local_baseline(community_rh1_client):
    message = _load_message()
    features = copy.deepcopy(message["features"])
    features["sites"]["sao_mamede"]["observations"]["virtual_battery"] = {"soc": 0.9}
    features["sites"]["rh1"]["observations"]["batteries"] = {"B01": {"SoC": 90}}
    features["sites"]["boavista"]["observations"]["solar_generation"] = 0.0
    features["sites"]["boavista"]["observations"]["non_shiftable_load"] = 15.0
    features["sites"]["sao_mamede"]["observations"]["solar_generation"] = 35.0
    features["sites"]["sao_mamede"]["observations"]["non_shiftable_load"] = 1.0
    features["community"]["price_signal"]["values"] = [0.23] + [0.10] * 95

    baseline_actions = _baseline_actions_from_local_pipelines(features)
    baseline_net = _site_net(features, baseline_actions)
    baseline_cost = _external_cost(baseline_net, 0.23)

    response = _inference(community_rh1_client, 2, features)
    assert response.status_code == 200
    coordinated_actions = response.json()["actions"]
    coordinated_net = _site_net(features, coordinated_actions)
    coordinated_cost = _external_cost(coordinated_net, 0.23)

    assert coordinated_cost <= baseline_cost + 1e-6



def test_sequence_smoke_replay(community_rh1_client):
    sequence = _load_sequence()
    assert sequence
    base_features = _load_message()["features"]

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
        response = community_rh1_client.post(
            "/inference",
            json={
                "agent_index": step["agent_index"],
                "features": features,
            },
        )
        assert response.status_code == 200
        assert set(response.json()["actions"].keys()) == {"0", "1", "2"}
