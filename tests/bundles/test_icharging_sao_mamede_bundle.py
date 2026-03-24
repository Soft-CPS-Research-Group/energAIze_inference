from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store


BUNDLE_DIR = Path("examples/icharging_sao_mamede_without_virtual_battery")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
ALIAS_PATH = BUNDLE_DIR / "aliases.json"
MESSAGE_PATH = BUNDLE_DIR / "exemplos_mensagem_SaoMamede_2303.json"

ACTION_CHARGER = "BB000SMI"
MIN_TECHNICAL_KW = 8.0
MAX_BB_KW = 50.0
PHYSICAL_PLUGS = {
    "BB000SMI_1",
    "BB000SMI_2",
    "M1123089-5_1",
    "M1123089-5_2",
    "M1123089-6_1",
    "M1123089-6_2",
}


def _run(client: TestClient, payload: dict) -> dict[str, float]:
    response = client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    return response.json()["actions"]["0"]


@pytest.fixture
def sao_mamede_client():
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
    payload["observations"]["charging_sessions"] = {
        "BB000SMI_1": {"power": 0.0, "electric_vehicle": ""},
        "BB000SMI_2": {"power": 0.0, "electric_vehicle": ""},
        "M1123089-5_1": {"power": 0.0, "electric_vehicle": ""},
        "M1123089-5_2": {"power": 0.0, "electric_vehicle": ""},
        "M1123089-6_1": {"power": 0.0, "electric_vehicle": ""},
        "M1123089-6_2": {"power": 0.0, "electric_vehicle": ""},
    }
    payload["observations"]["electric_vehicles"] = {}
    payload["forecasts"] = {}
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


def test_bundle_loads_sao_mamede(sao_mamede_client):
    pipeline = store.get_pipeline()
    assert pipeline.agent.strategy == "icharging_breaker"
    assert pipeline.agent._icharging_runtime is not None  # noqa: SLF001


def test_payload_assets_reflect_three_physical_chargers(sao_mamede_client):
    payload = _base_payload()
    sessions = set(payload["observations"]["charging_sessions"].keys())
    assert sessions == PHYSICAL_PLUGS


def test_actions_contract_has_single_controllable_charger(sao_mamede_client):
    payload = _base_payload()
    actions = _run(sao_mamede_client, payload)
    assert set(actions.keys()) == {ACTION_CHARGER}


def test_idle_outputs_minimum_even_without_ev(sao_mamede_client):
    payload = _base_payload()
    actions = _run(sao_mamede_client, payload)
    assert actions[ACTION_CHARGER] == pytest.approx(MIN_TECHNICAL_KW, rel=1e-6)


def test_connected_ev_on_second_plug_is_controlled(sao_mamede_client):
    payload = _base_payload()
    payload["observations"]["charging_sessions"]["BB000SMI_2"] = {
        "power": 0.0,
        "electric_vehicle": "SM_EV_02",
    }
    _set_flex(
        payload,
        ev_id="SM_EV_02",
        soc=0.20,
        target_soc=0.90,
        departure_minutes_from_now=60,
    )

    actions = _run(sao_mamede_client, payload)
    assert MIN_TECHNICAL_KW <= actions[ACTION_CHARGER] <= MAX_BB_KW
    assert actions[ACTION_CHARGER] > MIN_TECHNICAL_KW


def test_stale_power_on_other_bb_plug_does_not_mask_connected_ev(sao_mamede_client):
    payload = _base_payload()
    payload["observations"]["charging_sessions"]["BB000SMI_1"] = {
        "power": 38.0,
        "electric_vehicle": "0",
    }
    payload["observations"]["charging_sessions"]["BB000SMI_2"] = {
        "power": 0.0,
        "electric_vehicle": "SM_EV_03",
    }
    _set_flex(
        payload,
        ev_id="SM_EV_03",
        soc=0.10,
        target_soc=0.90,
        departure_minutes_from_now=30,
    )

    actions = _run(sao_mamede_client, payload)
    assert actions[ACTION_CHARGER] > MIN_TECHNICAL_KW


def test_grid_meter_headroom_reduces_dispatch_and_non_controllable_remain_unmanaged(
    sao_mamede_client,
):
    low_meter = _base_payload()
    low_meter["observations"]["charging_sessions"]["BB000SMI_2"] = {
        "power": 0.0,
        "electric_vehicle": "SM_EV_04",
    }
    _set_flex(
        low_meter,
        ev_id="SM_EV_04",
        soc=0.10,
        target_soc=0.90,
        departure_minutes_from_now=30,
    )
    low_meter["observations"]["grid_meters"]["GR01"]["energy_in"] = 0.0
    low_meter["observations"]["grid_meters"]["GR01"]["energy_out"] = 0.0
    low_meter["observations"]["solar_generation"] = 0.0
    low_meter["observations"]["charging_sessions"]["M1123089-5_1"] = {
        "power": 22.0,
        "electric_vehicle": "0",
    }
    low_meter["observations"]["charging_sessions"]["M1123089-6_2"] = {
        "power": 22.0,
        "electric_vehicle": "0",
    }

    high_meter = copy.deepcopy(low_meter)
    high_meter["observations"]["grid_meters"]["GR01"]["energy_in"] = 390.0
    high_meter["observations"]["grid_meters"]["GR01"]["energy_out"] = 0.0

    actions_low = _run(sao_mamede_client, low_meter)
    actions_high = _run(sao_mamede_client, high_meter)
    assert set(actions_low.keys()) == {ACTION_CHARGER}
    assert set(actions_high.keys()) == {ACTION_CHARGER}
    assert actions_high[ACTION_CHARGER] < actions_low[ACTION_CHARGER]


def test_price_forecast_do_not_change_dispatch_without_virtual_battery(sao_mamede_client):
    payload_a = _base_payload()
    payload_a["observations"]["charging_sessions"]["BB000SMI_2"] = {
        "power": 0.0,
        "electric_vehicle": "SM_EV_05",
    }
    _set_flex(
        payload_a,
        ev_id="SM_EV_05",
        soc=0.35,
        target_soc=0.85,
        departure_minutes_from_now=180,
    )

    payload_b = copy.deepcopy(payload_a)
    payload_b["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [9.0] * 96
    payload_b["forecasts"] = {"dummy": {"values": [123.0, 456.0]}}

    actions_a = _run(sao_mamede_client, payload_a)
    actions_b = _run(sao_mamede_client, payload_b)
    assert actions_a == actions_b
