from __future__ import annotations

import copy
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.state import store
from app.utils.flatten import flatten_payload


BUNDLE_DIR = Path("examples/icharging_community_boavista_sao_mamede_rh1_without_virtual_battery")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
MESSAGE_PATH = BUNDLE_DIR / "community_message_example.json"
SEQUENCE_PATH = BUNDLE_DIR / "community_sequence.json"


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


def _site_net(features: dict, actions: dict) -> float:
    total = 0.0
    for site_key, site_payload in features["sites"].items():
        obs = site_payload["observations"]
        non_shiftable = float(obs.get("non_shiftable_load", 0.0))
        solar = float(obs.get("solar_generation", 0.0))
        if site_key == "rh1":
            total += non_shiftable + float(actions["2"]["ev_charge_kw"]) + float(actions["2"]["battery_kw"]) - solar
            continue
        site_actions = actions["0"] if site_key == "boavista" else actions["1"]
        connected_kw = 0.0
        for charger_id, session in obs.get("charging_sessions", {}).items():
            if not _normalize_ev_id((session or {}).get("electric_vehicle")):
                continue
            connected_kw += float(site_actions.get(charger_id, 0.0))
        total += non_shiftable + connected_kw - solar
    return total


def _external_cost(total_net_kw: float, price: float, export_factor: float = 0.8) -> float:
    return max(total_net_kw, 0.0) * price - max(-total_net_kw, 0.0) * price * export_factor


def _local_baseline_actions(features: dict) -> dict[str, dict[str, float]]:
    record = store.get_record()
    assert record is not None
    baseline = {}
    for agent_index in record.loaded_agent_indices:
        pipeline = record.pipelines[agent_index]
        cfg = pipeline.manifest.get_artifact(agent_index).config or {}
        site_key = cfg.get("input_site_key")
        if not site_key:
            continue
        selected = copy.deepcopy(features["sites"][site_key])
        if "timestamp" not in selected and features.get("timestamp") is not None:
            selected["timestamp"] = features["timestamp"]
        observations = selected.get("observations", {})
        normalized = dict(observations)
        if "timestamp" not in normalized and selected.get("timestamp") is not None:
            normalized["timestamp"] = selected["timestamp"]
        flattened = flatten_payload(normalized)
        out = pipeline.inference(flattened)
        baseline[str(agent_index)] = out[str(agent_index)]
    return baseline


def test_bundle_loads_and_agent_contracts_without_virtual_battery():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0)
    client = TestClient(app)
    try:
        message = _load_message()
        resp_sm = _inference(client, 1, message["features"])
        assert resp_sm.status_code == 200
        actions = resp_sm.json()["actions"]
        assert set(actions.keys()) == {"0", "1", "2"}
        sm_actions = actions["1"]
        assert set(sm_actions.keys()) == {"BB000SMI_1", "BB000SMI_2"}

        assert set(actions["2"].keys()) == RH1_ACTIONS
    finally:
        if store.is_configured():
            store.unload()


def test_sequence_smoke_replay_without_virtual_battery():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0)
    client = TestClient(app)
    try:
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
            response = _inference(client, step["agent_index"], features)
            assert response.status_code == 200
            assert set(response.json()["actions"].keys()) == {"0", "1", "2"}
            assert "virtual_battery_kw" not in response.json()["actions"]["1"]
    finally:
        if store.is_configured():
            store.unload()


def test_cost_reduction_vs_local_baseline_without_virtual_battery():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0)
    client = TestClient(app)
    try:
        message = _load_message()
        features = copy.deepcopy(message["features"])
        features["sites"]["boavista"]["observations"]["solar_generation"] = 0.0
        features["sites"]["boavista"]["observations"]["non_shiftable_load"] = 14.0
        features["sites"]["rh1"]["observations"]["batteries"] = {"B01": {"SoC": 90}}
        features["sites"]["rh1"]["observations"]["solar_generation"] = 0.0
        community = features.setdefault("community", {})
        community["price_signal"] = {
            "values": [0.24] + [0.10] * 95,
            "measurement_unit": "€/kWh",
            "frequency_seconds": 900,
        }

        baseline_actions = _local_baseline_actions(features)
        baseline_cost = _external_cost(_site_net(features, baseline_actions), 0.24)

        response = _inference(client, 2, features)
        assert response.status_code == 200
        coordinated_actions = response.json()["actions"]
        coordinated_cost = _external_cost(_site_net(features, coordinated_actions), 0.24)
        assert coordinated_cost <= baseline_cost + 1e-6
    finally:
        if store.is_configured():
            store.unload()
