from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store


BUNDLE_DIR = Path("examples/icharging_boavista_with_flex_community")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
ALIAS_PATH = BUNDLE_DIR / "aliases.json"
MESSAGE_PATH = BUNDLE_DIR / "exemplos_mensagem_i-charging_headquarters_2303.json"

DECISION_INTERVAL_HOURS = 15.0 / 3600.0


def _kwh_for_interval(power_kw: float) -> float:
    return power_kw * DECISION_INTERVAL_HOURS


@pytest.fixture
def boavista_with_flex_community_client():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0, ALIAS_PATH)
    client = TestClient(app)
    try:
        yield client
    finally:
        if store.is_configured():
            store.unload()


def _base_payload() -> dict:
    payload = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))[0]
    payload = copy.deepcopy(payload)
    payload["timestamp"] = "2026-03-01T10:00:00Z"
    observations = payload.setdefault("observations", {})
    observations["solar_generation"] = 0.0

    sessions = observations.setdefault("charging_sessions", {})
    for charger_id in list(sessions.keys()):
        sessions[charger_id] = {"power": 0.0, "electric_vehicle": ""}
    observations["electric_vehicles"] = {}
    payload["community"] = {
        "energy_in_total": 0.0,
        "energy_out_total": 0.0,
    }
    return payload


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


def _connect_many_chargers_for_contention(payload: dict) -> None:
    sessions = payload["observations"]["charging_sessions"]
    for idx, charger_id in enumerate(sorted(sessions.keys())):
        ev_id = f"EV_COMM_{idx}"
        sessions[charger_id] = {"power": 0.0, "electric_vehicle": ev_id}
        _set_flex(
            payload,
            ev_id=ev_id,
            soc=0.1,
            target_soc=0.9,
            departure_minutes_from_now=30,
        )


def _run(client: TestClient, payload: dict) -> dict[str, float]:
    response = client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    return response.json()["actions"]["0"]


def _connected_total_kw(payload: dict, actions: dict[str, float]) -> float:
    sessions = payload.get("observations", {}).get("charging_sessions", {})
    total = 0.0
    for charger_id, session in sessions.items():
        ev_id = str((session or {}).get("electric_vehicle") or "").strip()
        if not ev_id:
            continue
        total += float(actions.get(charger_id, 0.0))
    return total


def _connect_charger(payload: dict, charger_id: str, ev_id: str, *, flexible: bool) -> None:
    payload["observations"]["charging_sessions"][charger_id] = {
        "power": 0.0,
        "electric_vehicle": ev_id,
    }
    if flexible:
        _set_flex(
            payload,
            ev_id=ev_id,
            soc=0.15,
            target_soc=0.90,
            departure_minutes_from_now=45,
        )


def _line_total_kw(payload: dict, actions: dict[str, float], line: str) -> float:
    cfg = store.get_pipeline().agent._icharging_runtime.config  # noqa: SLF001
    sessions = payload["observations"]["charging_sessions"]
    total = 0.0
    for charger_id, meta in cfg.chargers.items():
        if str(meta.get("line", "")) != line:
            continue
        ev_id = str((sessions.get(charger_id) or {}).get("electric_vehicle") or "").strip()
        if not ev_id:
            continue
        total += float(actions.get(charger_id, 0.0))
    return total


def test_bundle_loads_boavista_with_flex_community(boavista_with_flex_community_client):
    pipeline = store.get_pipeline()
    assert pipeline.agent.strategy == "icharging_breaker"
    assert pipeline.agent._icharging_runtime is not None  # noqa: SLF001


def test_missing_required_community_fields_returns_400(boavista_with_flex_community_client):
    missing_in = _base_payload()
    del missing_in["community"]["energy_in_total"]
    response = boavista_with_flex_community_client.post("/inference", json={"features": missing_in})
    assert response.status_code == 400

    missing_out = _base_payload()
    del missing_out["community"]["energy_out_total"]
    response = boavista_with_flex_community_client.post("/inference", json={"features": missing_out})
    assert response.status_code == 400


def test_community_deficit_reduces_local_dispatch(boavista_with_flex_community_client):
    neutral = _base_payload()
    _connect_many_chargers_for_contention(neutral)

    deficit = copy.deepcopy(neutral)
    deficit["community"]["energy_in_total"] = _kwh_for_interval(60.0)
    deficit["community"]["energy_out_total"] = 0.0

    surplus = copy.deepcopy(neutral)
    surplus["community"]["energy_in_total"] = 0.0
    surplus["community"]["energy_out_total"] = _kwh_for_interval(60.0)

    neutral_actions = _run(boavista_with_flex_community_client, neutral)
    deficit_actions = _run(boavista_with_flex_community_client, deficit)
    surplus_actions = _run(boavista_with_flex_community_client, surplus)

    neutral_total = _connected_total_kw(neutral, neutral_actions)
    deficit_total = _connected_total_kw(deficit, deficit_actions)
    surplus_total = _connected_total_kw(surplus, surplus_actions)

    assert deficit_total < neutral_total
    assert neutral_total <= surplus_total


def test_small_community_kwh_value_changes_dispatch(boavista_with_flex_community_client):
    no_community_gap = _base_payload()
    _connect_many_chargers_for_contention(no_community_gap)

    with_small_gap = copy.deepcopy(no_community_gap)
    with_small_gap["community"]["energy_in_total"] = _kwh_for_interval(1.774152)
    with_small_gap["community"]["energy_out_total"] = 0.0

    no_gap_actions = _run(boavista_with_flex_community_client, no_community_gap)
    small_gap_actions = _run(boavista_with_flex_community_client, with_small_gap)

    no_gap_total = _connected_total_kw(no_community_gap, no_gap_actions)
    small_gap_total = _connected_total_kw(with_small_gap, small_gap_actions)
    assert small_gap_total < no_gap_total


def test_nonflex_is_prioritized_and_flex_respects_floor_under_community_deficit(
    boavista_with_flex_community_client,
):
    payload_neutral = _base_payload()
    payload_deficit = _base_payload()

    nonflex_ids = [
        "AC000002_1",
        "AC000003_1",
        "AC000004_1",
        "AC000005_1",
        "AC000006_1",
        "AC000007_1",
        "AC000008_1",
        "AC000009_1",
    ]
    flex_ids = ["AC000001_1", "AC000010_1", "AC000011_1", "AC000012_1", "AC000013_1"]

    for idx, charger_id in enumerate(nonflex_ids):
        ev_id = f"NF_{idx}"
        _connect_charger(payload_neutral, charger_id, ev_id, flexible=False)
        _connect_charger(payload_deficit, charger_id, ev_id, flexible=False)
    for idx, charger_id in enumerate(flex_ids):
        ev_id = f"F_{idx}"
        _connect_charger(payload_neutral, charger_id, ev_id, flexible=True)
        _connect_charger(payload_deficit, charger_id, ev_id, flexible=True)

    payload_deficit["community"]["energy_in_total"] = _kwh_for_interval(30.0)
    payload_deficit["community"]["energy_out_total"] = 0.0

    neutral_actions = _run(boavista_with_flex_community_client, payload_neutral)
    deficit_actions = _run(boavista_with_flex_community_client, payload_deficit)

    nonflex_neutral = [float(neutral_actions[cid]) for cid in nonflex_ids]
    nonflex_deficit = [float(deficit_actions[cid]) for cid in nonflex_ids]
    flex_neutral = [float(neutral_actions[cid]) for cid in flex_ids]
    flex_deficit = [float(deficit_actions[cid]) for cid in flex_ids]

    assert min(nonflex_deficit) >= 4.5
    assert min(flex_deficit) >= 1.6
    assert max(flex_deficit) <= 4.6 + 1e-6
    assert sum(nonflex_deficit) == pytest.approx(sum(nonflex_neutral), rel=1e-3)
    assert sum(flex_deficit) < sum(flex_neutral)

    assert _connected_total_kw(payload_deficit, deficit_actions) <= 55.0 + 1e-3
    assert _line_total_kw(payload_deficit, deficit_actions, "L1") <= 18.333 + 1e-3
    assert _line_total_kw(payload_deficit, deficit_actions, "L2") <= 18.333 + 1e-3
    assert _line_total_kw(payload_deficit, deficit_actions, "L3") <= 18.333 + 1e-3
