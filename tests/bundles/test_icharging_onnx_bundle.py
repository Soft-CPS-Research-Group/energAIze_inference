from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store


BUNDLE_DIR = Path("examples/ichargingusecase_onnx")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
ALIAS_PATH = BUNDLE_DIR / "aliases.json"
MESSAGE_PATH = BUNDLE_DIR / "exemplos_mensagem_i-charging_headquarters_onnx.json"


@pytest.fixture
def onnx_client():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0, ALIAS_PATH)
    client = TestClient(app)
    try:
        yield client
    finally:
        if store.is_configured():
            store.unload()


def _run(client: TestClient, payload: dict) -> dict[str, float]:
    response = client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    return response.json()["actions"]["0"]


def test_bundle_loads_onnx_sample(onnx_client):
    pipeline = store.get_pipeline()
    action_names = pipeline.manifest.get_action_names(0)
    assert len(action_names) == 21
    assert action_names[0] == "AC000001_1"
    assert action_names[-1] == "b_2"


def test_onnx_replay_new_envelope_messages(onnx_client):
    scenarios = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))
    assert scenarios
    for scenario in scenarios:
        actions = _run(onnx_client, scenario)
        assert len(actions) == 21


def test_onnx_actions_match_contract_and_bounds(onnx_client):
    scenario = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))[0]
    actions = _run(onnx_client, scenario)

    pipeline = store.get_pipeline()
    action_names = pipeline.manifest.get_action_names(0)
    bounds = pipeline.manifest.get_action_bounds(0) or {}
    lows = [float(v) for v in bounds.get("low", [])]
    highs = [float(v) for v in bounds.get("high", [])]

    assert set(actions.keys()) == set(action_names)
    for idx, name in enumerate(action_names):
        value = float(actions[name])
        if idx < len(lows):
            assert value >= lows[idx] - 1e-6
        if idx < len(highs):
            assert value <= highs[idx] + 1e-6


def test_onnx_missing_observations_returns_400(onnx_client):
    response = onnx_client.post(
        "/inference",
        json={"features": {"timestamp": "2026-03-04T12:00:00Z", "charging_sessions": {}}},
    )
    assert response.status_code == 400
    detail = str(response.json().get("detail", ""))
    assert "observations" in detail
