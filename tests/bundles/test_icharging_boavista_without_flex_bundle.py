from __future__ import annotations

from pathlib import Path

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


BUNDLE_DIR = Path("examples/icharging_boavista_without_flex")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
ALIAS_PATH = BUNDLE_DIR / "aliases.json"
SEQUENCE_PATH = BUNDLE_DIR / "three_day_sequence.json"


@pytest.fixture
def boavista_without_flex_client():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0, ALIAS_PATH)
    client = TestClient(app)
    try:
        yield client
    finally:
        if store.is_configured():
            store.unload()


def test_bundle_loads_boavista_without_flex(boavista_without_flex_client):
    pipeline = store.get_pipeline()
    assert pipeline.agent.strategy == "breaker_only"
    assert pipeline.agent._breaker_runtime is not None  # noqa: SLF001


def test_sequence_replay_boavista_without_flex(boavista_without_flex_client):
    scenarios = load_json(SEQUENCE_PATH)
    assert scenarios

    for scenario in scenarios:
        payload = normalize_record(scenario)
        actions = post_inference(boavista_without_flex_client, payload)
        board_limit = BASE_BOARD_LIMIT_KW + payload.get("solar_generation", 0.0)
        assert_board_and_phase_limits(actions, payload, board_limit_kw=board_limit)
        assert_connected_charger_action_bounds(actions, payload)
