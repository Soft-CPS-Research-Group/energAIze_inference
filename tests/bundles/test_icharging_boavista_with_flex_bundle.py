from __future__ import annotations

from pathlib import Path
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store
from tests.bundles._icharging_shared import (
    BASE_BOARD_LIMIT_KW,
    assert_board_and_phase_limits,
    assert_connected_charger_action_bounds,
    load_json,
    normalize_record,
    post_inference,
)


BUNDLE_DIR = Path("examples/icharging_boavista_with_flex")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
ALIAS_PATH = BUNDLE_DIR / "aliases.json"
MESSAGE_PATH = BUNDLE_DIR / "exemplos_mensagem_i-charging_headquarters.json"


@pytest.fixture
def boavista_with_flex_client():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0, ALIAS_PATH)
    client = TestClient(app)
    try:
        yield client
    finally:
        if store.is_configured():
            store.unload()


def test_bundle_loads_boavista_with_flex(boavista_with_flex_client):
    pipeline = store.get_pipeline()
    assert pipeline.agent.strategy == "icharging_breaker"
    assert pipeline.agent._icharging_runtime is not None  # noqa: SLF001


def test_replay_new_headquarters_messages(boavista_with_flex_client):
    scenarios = load_json(MESSAGE_PATH)
    assert scenarios
    for scenario in scenarios:
        payload = normalize_record(scenario)
        actions = post_inference(boavista_with_flex_client, payload)
        board_limit = BASE_BOARD_LIMIT_KW + float(
            payload.get("observations", {}).get("solar_generation", 0.0)
        )
        assert_board_and_phase_limits(actions, payload, board_limit_kw=board_limit)
        assert_connected_charger_action_bounds(actions, payload)


def test_price_vector_does_not_change_dispatch(boavista_with_flex_client):
    record = load_json(MESSAGE_PATH)[0]
    payload_a = normalize_record(record)
    payload_b = json.loads(json.dumps(payload_a))
    payload_b["observations"]["energy_price"]["values"] = [9.99] * 96
    payload_b["observations"]["energy_price"]["measurement_unit"] = "€/kWh"

    actions_a = post_inference(boavista_with_flex_client, payload_a)
    actions_b = post_inference(boavista_with_flex_client, payload_b)
    assert actions_a == actions_b
