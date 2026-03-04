from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.state import store


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


def test_bundle_loads_and_agent_contracts_without_virtual_battery():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0)
    client = TestClient(app)
    try:
        message = _load_message()
        resp_sm = _inference(client, 1, message["features"])
        assert resp_sm.status_code == 200
        sm_actions = resp_sm.json()["actions"]["1"]
        assert set(sm_actions.keys()) == {"BB000SMI_1", "BB000SMI_2"}

        resp_rh1 = _inference(client, 2, message["features"])
        assert resp_rh1.status_code == 200
        assert set(resp_rh1.json()["actions"]["2"].keys()) == RH1_ACTIONS
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
