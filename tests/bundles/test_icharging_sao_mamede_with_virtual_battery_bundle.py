from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store


BUNDLE_DIR = Path("examples/icharging_sao_mamede_with_virtual_battery")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
ALIAS_PATH = BUNDLE_DIR / "aliases.json"
MESSAGE_PATH = BUNDLE_DIR / "exemplos_mensagem_SaoMamede_2303.json"

ACTION_PLUG_1 = "BB000SMI_1"
ACTION_PLUG_2 = "BB000SMI_2"
ACTION_PLUGS = {ACTION_PLUG_1, ACTION_PLUG_2}
ACTION_BATTERY = "B01"
MIN_TECHNICAL_KW = 8.0


@pytest.fixture
def sao_mamede_with_battery_client():
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
    payload["observations"]["batteries"] = {
        "B01": {
            "energy_in": 0.0,
            "energy_out": 0.0,
            "SoC": 0.50,
        }
    }
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


def _run(client: TestClient, payload: dict) -> dict[str, float]:
    response = client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    return response.json()["actions"]["0"]


def _bb_total_kw(actions: dict[str, float]) -> float:
    return float(actions.get(ACTION_PLUG_1, 0.0) + actions.get(ACTION_PLUG_2, 0.0))


def _has_ev(raw: object) -> bool:
    text = "" if raw is None else str(raw).strip()
    if not text:
        return False
    return text.lower() not in {"0", "0.0", "none", "nan", "null"}


def _bb_counted_total_kw(payload: dict, actions: dict[str, float]) -> float:
    sessions = payload["observations"]["charging_sessions"]
    total = 0.0
    for plug in ACTION_PLUGS:
        ev_raw = (sessions.get(plug) or {}).get("electric_vehicle")
        if _has_ev(ev_raw):
            total += float(actions.get(plug, 0.0))
    return total


def _assert_idle_floor_on_unplugged(payload: dict, actions: dict[str, float]) -> None:
    sessions = payload["observations"]["charging_sessions"]
    for plug in ACTION_PLUGS:
        ev_raw = (sessions.get(plug) or {}).get("electric_vehicle")
        if not _has_ev(ev_raw):
            assert actions.get(plug, 0.0) == pytest.approx(MIN_TECHNICAL_KW, rel=1e-6)


def test_actions_contract_includes_virtual_battery(sao_mamede_with_battery_client):
    actions = _run(sao_mamede_with_battery_client, _base_payload())
    assert set(actions.keys()) == ACTION_PLUGS | {ACTION_BATTERY}


def test_idle_outputs_idle_floor_on_both_plugs(sao_mamede_with_battery_client):
    payload = _base_payload()
    actions = _run(sao_mamede_with_battery_client, payload)
    assert _bb_total_kw(actions) == pytest.approx(2 * MIN_TECHNICAL_KW, rel=1e-6)
    assert _bb_counted_total_kw(payload, actions) == pytest.approx(0.0, rel=1e-6)
    _assert_idle_floor_on_unplugged(payload, actions)


def test_manifest_models_single_pt_limit(sao_mamede_with_battery_client):
    runtime = store.get_pipeline().agent._icharging_runtime  # noqa: SLF001
    assert runtime is not None
    cfg = runtime.config
    assert cfg.max_board_kw == pytest.approx(400.0, rel=1e-6)
    assert set(cfg.line_limits.keys()) == {"PT"}
    assert cfg.chargers["BB000SMI"]["line"] == "PT"
    assert cfg.virtual_battery_soc_unit_mode == "fraction"


def test_single_controllable_bb_charger_uses_merged_plugs(sao_mamede_with_battery_client):
    payload = _base_payload()
    payload["observations"]["charging_sessions"]["BB000SMI_2"] = {
        "power": 0.0,
        "electric_vehicle": "SM_EV_1",
    }
    _set_flex(
        payload,
        ev_id="SM_EV_1",
        soc=0.30,
        target_soc=0.90,
        departure_minutes_from_now=120,
    )
    actions = _run(sao_mamede_with_battery_client, payload)
    assert actions[ACTION_PLUG_1] == pytest.approx(MIN_TECHNICAL_KW, rel=1e-6)
    assert actions[ACTION_PLUG_2] >= MIN_TECHNICAL_KW
    _assert_idle_floor_on_unplugged(payload, actions)


def test_virtual_battery_responds_to_energy_tariffs_price_vector(
    sao_mamede_with_battery_client,
):
    common = _base_payload()
    common["observations"]["charging_sessions"]["BB000SMI_2"] = {
        "power": 8.0,
        "electric_vehicle": "SM_EV_2",
    }
    _set_flex(
        common,
        ev_id="SM_EV_2",
        soc=0.50,
        target_soc=0.80,
        departure_minutes_from_now=120,
    )
    common["observations"]["solar_generation"] = 0.0
    common["observations"]["non_shiftable_load"] = 0.0

    expensive_now = copy.deepcopy(common)
    expensive_now["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.30] + [0.08] * 95
    expensive_actions = _run(sao_mamede_with_battery_client, expensive_now)

    cheap_now = copy.deepcopy(common)
    cheap_now["observations"]["solar_generation"] = 80.0
    cheap_now["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.05] + [0.20] * 95
    cheap_actions = _run(sao_mamede_with_battery_client, cheap_now)

    assert expensive_actions[ACTION_BATTERY] < -1e-6
    assert cheap_actions[ACTION_BATTERY] > 1e-6


def test_virtual_battery_soc_guards(sao_mamede_with_battery_client):
    high_soc = _base_payload()
    high_soc["observations"]["batteries"]["B01"]["SoC"] = 1.0
    high_soc["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.08] + [0.22] * 95
    high_actions = _run(sao_mamede_with_battery_client, high_soc)
    assert high_actions[ACTION_BATTERY] == pytest.approx(0.0, rel=1e-6)

    low_soc = _base_payload()
    low_soc["observations"]["batteries"]["B01"]["SoC"] = 0.05
    low_soc["observations"]["solar_generation"] = 0.0
    low_soc["observations"]["non_shiftable_load"] = 0.0
    low_soc["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.25] + [0.08] * 95
    low_actions = _run(sao_mamede_with_battery_client, low_soc)
    assert low_actions[ACTION_BATTERY] >= 0.0


def test_grid_meter_headroom_constrains_bb_dispatch(sao_mamede_with_battery_client):
    low_meter = _base_payload()
    low_meter["observations"]["charging_sessions"]["BB000SMI_2"] = {
        "power": 0.0,
        "electric_vehicle": "SM_EV_4",
    }
    _set_flex(
        low_meter,
        ev_id="SM_EV_4",
        soc=0.10,
        target_soc=0.90,
        departure_minutes_from_now=30,
    )
    low_meter["observations"]["grid_meters"]["GR01"]["energy_in_total"] = 0.0
    low_meter["observations"]["grid_meters"]["GR01"]["energy_out_total"] = 0.0
    low_meter["observations"]["solar_generation"] = 0.0

    high_meter = copy.deepcopy(low_meter)
    high_meter["observations"]["grid_meters"]["GR01"]["energy_in_total"] = 390.0
    high_meter["observations"]["grid_meters"]["GR01"]["energy_out_total"] = 0.0

    actions_low = _run(sao_mamede_with_battery_client, low_meter)
    actions_high = _run(sao_mamede_with_battery_client, high_meter)
    assert _bb_counted_total_kw(high_meter, actions_high) < _bb_counted_total_kw(
        low_meter, actions_low
    )
    _assert_idle_floor_on_unplugged(low_meter, actions_low)
    _assert_idle_floor_on_unplugged(high_meter, actions_high)


def test_virtual_battery_uses_pt_meter_net_import_for_local_dispatch(
    sao_mamede_with_battery_client,
):
    payload = _base_payload()
    payload["observations"]["batteries"]["B01"]["SoC"] = 0.70
    payload["observations"]["solar_generation"] = 0.0
    payload["observations"]["non_shiftable_load"] = 0.0
    payload["observations"]["grid_meters"]["GR01"]["energy_in_total"] = 120.0
    payload["observations"]["grid_meters"]["GR01"]["energy_out_total"] = 0.0
    payload["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.30] + [0.08] * 95

    actions = _run(sao_mamede_with_battery_client, payload)
    assert actions[ACTION_BATTERY] < -1e-6
