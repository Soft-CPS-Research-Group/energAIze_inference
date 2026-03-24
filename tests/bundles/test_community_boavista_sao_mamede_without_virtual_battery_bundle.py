from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store

pytestmark = pytest.mark.skip(reason="Community em standby")


BUNDLE_DIR = Path("examples/icharging_community_boavista_sao_mamede_without_virtual_battery")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
MESSAGE_PATH = BUNDLE_DIR / "community_message_example.json"
SEQUENCE_PATH = BUNDLE_DIR / "community_sequence.json"


def _inference(client: TestClient, agent_index: int, features: dict):
    return client.post("/inference", json={"agent_index": agent_index, "features": features})


def _load_message() -> dict:
    return json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))


def _load_sequence() -> list[dict]:
    return json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))


def test_bundle_loads_and_actions_contract_without_virtual_battery():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0)
    client = TestClient(app)
    try:
        message = _load_message()
        resp = _inference(client, 1, message["features"])
        assert resp.status_code == 200
        actions = resp.json()["actions"]
        assert set(actions.keys()) == {"0", "1"}
        assert set(actions["1"].keys()) == {"BB000SMI_1", "BB000SMI_2"}
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
            features = json.loads(json.dumps(base_features))
            step_features = step.get("features", {})
            step_sites = step_features.get("sites", {})
            if isinstance(step_sites, dict):
                for site_key, site_payload in step_sites.items():
                    if isinstance(site_payload, dict):
                        features.setdefault("sites", {})[site_key] = site_payload
            if isinstance(step_features.get("community"), dict):
                features["community"] = step_features["community"]
            if step_features.get("timestamp") is not None:
                features["timestamp"] = step_features["timestamp"]
            response = _inference(client, step["agent_index"], features)
            assert response.status_code == 200
            assert set(response.json()["actions"].keys()) == {"0", "1"}
            assert "virtual_battery_kw" not in response.json()["actions"]["1"]
    finally:
        if store.is_configured():
            store.unload()
