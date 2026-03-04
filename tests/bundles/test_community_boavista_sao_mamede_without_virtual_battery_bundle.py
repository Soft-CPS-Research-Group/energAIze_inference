from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.state import store


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
        actions = resp.json()["actions"]["1"]
        assert set(actions.keys()) == {"BB000SMI_1", "BB000SMI_2"}
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
        for step in sequence:
            response = _inference(client, step["agent_index"], step["features"])
            assert response.status_code == 200
            if step["agent_index"] == 1:
                assert "virtual_battery_kw" not in response.json()["actions"]["1"]
    finally:
        if store.is_configured():
            store.unload()
