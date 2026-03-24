from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import random

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
MESSAGE_PATH = BUNDLE_DIR / "exemplos_mensagem_i-charging_headquarters_2303.json"

L1_CHARGERS = [
    "AC000002_1",
    "AC000005_1",
    "AC000008_1",
    "AC000011_1",
    "AC000014_1",
    "ACEXT001_1",
    "ACEXT002_1",
    "ACEXT003_1",
]
L2_CHARGERS = ["AC000003_1", "AC000006_1", "AC000009_1", "AC000012_1", "ACEXT004_1"]
L3_CHARGERS = ["AC000001_1", "AC000004_1", "AC000007_1", "AC000010_1", "AC000013_1"]


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
    payload_b["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [9.99] * 96
    payload_b["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["measurement_unit"] = "€/kWh"

    actions_a = post_inference(boavista_with_flex_client, payload_a)
    actions_b = post_inference(boavista_with_flex_client, payload_b)
    assert actions_a == actions_b


def _empty_hq_payload() -> dict:
    record = load_json(MESSAGE_PATH)[0]
    payload = normalize_record(record)
    payload["timestamp"] = "2026-03-01T10:00:00Z"
    observations = payload.setdefault("observations", {})
    observations["solar_generation"] = 0.0

    sessions = observations.setdefault("charging_sessions", {})
    for charger_id in list(sessions.keys()):
        sessions[charger_id] = {"power": 0.0, "electric_vehicle": ""}
    observations["electric_vehicles"] = {}
    return payload


def _connect_chargers(payload: dict, charger_ids: list[str]) -> dict[str, str]:
    sessions = payload["observations"]["charging_sessions"]
    ev_by_charger: dict[str, str] = {}
    for idx, charger_id in enumerate(charger_ids):
        ev_id = f"EV_SYN_{idx}_{charger_id}"
        sessions[charger_id] = {"power": 0.0, "electric_vehicle": ev_id}
        ev_by_charger[charger_id] = ev_id
    return ev_by_charger


def _set_flex(
    payload: dict,
    *,
    ev_id: str,
    soc: float,
    target_soc: float,
    departure_minutes_from_now: int,
) -> None:
    now = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    departure = now + timedelta(minutes=departure_minutes_from_now)
    payload["observations"]["electric_vehicles"][ev_id] = {
        "SoC": soc,
        "flexibility": {
            "estimated_soc_at_arrival": None,
            "estimated_soc_at_departure": target_soc,
            "estimated_time_at_arrival": "",
            "estimated_time_at_departure": departure.isoformat().replace("+00:00", "Z"),
            "charger": "",
            "mode": "",
        },
    }


def test_real_flex_prioritizes_urgent_vehicle(boavista_with_flex_client):
    base_payload = _empty_hq_payload()
    connected_l1 = ["AC000002_1", "AC000005_1", "AC000008_1", "AC000011_1", "AC000014_1"]
    ev_map = _connect_chargers(base_payload, connected_l1)

    payload_without_flex = copy.deepcopy(base_payload)
    payload_with_flex = copy.deepcopy(base_payload)

    urgent_charger = "AC000002_1"
    _set_flex(
        payload_with_flex,
        ev_id=ev_map[urgent_charger],
        soc=0.10,
        target_soc=0.90,
        departure_minutes_from_now=30,
    )

    actions_without_flex = post_inference(boavista_with_flex_client, payload_without_flex)
    actions_with_flex = post_inference(boavista_with_flex_client, payload_with_flex)

    assert_board_and_phase_limits(actions_without_flex, payload_without_flex, board_limit_kw=55.0)
    assert_board_and_phase_limits(actions_with_flex, payload_with_flex, board_limit_kw=55.0)
    assert_connected_charger_action_bounds(actions_without_flex, payload_without_flex)
    assert_connected_charger_action_bounds(actions_with_flex, payload_with_flex)

    assert actions_with_flex[urgent_charger] >= 4.5
    assert actions_with_flex[urgent_charger] > actions_without_flex[urgent_charger] + 0.5


def test_real_flex_extensive_activation_across_l1_contention(boavista_with_flex_client):
    rng = random.Random(42)
    changed_cases = 0
    total_cases = 24

    for _ in range(total_cases):
        payload_with_flex = _empty_hq_payload()
        selected_l1 = rng.sample(L1_CHARGERS, 5)
        selected_l2 = rng.sample(L2_CHARGERS, 1)
        selected_l3 = rng.sample(L3_CHARGERS, 1)
        ev_map = _connect_chargers(payload_with_flex, selected_l1 + selected_l2 + selected_l3)

        urgent_charger = selected_l1[0]
        _set_flex(
            payload_with_flex,
            ev_id=ev_map[urgent_charger],
            soc=0.10,
            target_soc=0.90,
            departure_minutes_from_now=30,
        )

        second_flex_charger = selected_l1[1]
        _set_flex(
            payload_with_flex,
            ev_id=ev_map[second_flex_charger],
            soc=0.45,
            target_soc=0.55,
            departure_minutes_from_now=180,
        )

        payload_without_flex = copy.deepcopy(payload_with_flex)
        payload_without_flex["observations"]["electric_vehicles"] = {}

        actions_with_flex = post_inference(boavista_with_flex_client, payload_with_flex)
        actions_without_flex = post_inference(boavista_with_flex_client, payload_without_flex)

        assert_board_and_phase_limits(actions_with_flex, payload_with_flex, board_limit_kw=55.0)
        assert_board_and_phase_limits(actions_without_flex, payload_without_flex, board_limit_kw=55.0)
        assert_connected_charger_action_bounds(actions_with_flex, payload_with_flex)
        assert_connected_charger_action_bounds(actions_without_flex, payload_without_flex)

        urgent_delta = actions_with_flex[urgent_charger] - actions_without_flex[urgent_charger]
        if urgent_delta > 0.4:
            changed_cases += 1
        assert actions_with_flex[urgent_charger] >= 4.5

    assert changed_cases >= int(total_cases * 0.8)


def test_infeasible_flex_is_curtailed_to_respect_phase_limit(boavista_with_flex_client):
    payload = _empty_hq_payload()
    selected_l1 = ["AC000002_1", "AC000005_1", "AC000008_1", "AC000011_1", "AC000014_1"]
    ev_map = _connect_chargers(payload, selected_l1)

    for charger_id in selected_l1:
        _set_flex(
            payload,
            ev_id=ev_map[charger_id],
            soc=0.10,
            target_soc=0.90,
            departure_minutes_from_now=30,
        )

    actions = post_inference(boavista_with_flex_client, payload)
    assert_board_and_phase_limits(actions, payload, board_limit_kw=55.0)
    assert_connected_charger_action_bounds(actions, payload)
    assert max(actions[cid] for cid in selected_l1) <= 4.0


def test_triphase_plus_l1_flex_contention_still_respects_phase_limit(boavista_with_flex_client):
    payload = _empty_hq_payload()
    connected = list(L1_CHARGERS) + ["BB000018_1"]
    ev_map = _connect_chargers(payload, connected)

    for charger_id in connected:
        _set_flex(
            payload,
            ev_id=ev_map[charger_id],
            soc=0.10,
            target_soc=0.95,
            departure_minutes_from_now=30,
        )

    actions = post_inference(boavista_with_flex_client, payload)
    assert_board_and_phase_limits(actions, payload, board_limit_kw=55.0)
    assert_connected_charger_action_bounds(actions, payload)
