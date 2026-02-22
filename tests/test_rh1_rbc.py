from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store
from app.utils.flatten import flatten_payload


BUNDLE_DIR = Path("examples/rh1_bundle")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
ALIAS_PATH = BUNDLE_DIR / "aliases.json"
SEQUENCE_PATH = BUNDLE_DIR / "rh1_sequence.json"
MESSAGE_PATH = BUNDLE_DIR / "rh1_message_example.json"


def _safe_float(value, default: float = 0.0) -> float:  # noqa: ANN001
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _battery_bounds(soc: float, dt_hours: float, cfg) -> tuple[float, float]:  # noqa: ANN001
    cap = max(float(cfg.battery_capacity_kwh), 1e-6)
    eff = min(max(float(cfg.battery_efficiency), 1e-6), 1.0)
    soc_min = min(max(float(cfg.battery_soc_min), 0.0), 1.0)
    soc_max = min(max(float(cfg.battery_soc_max), 0.0), 1.0)
    nominal = max(float(cfg.battery_nominal_power_kw), 0.0)

    charge_room = max(soc_max - soc, 0.0) * cap
    discharge_room = max(soc - soc_min, 0.0) * cap

    max_charge_soc = charge_room / max(dt_hours * eff, 1e-6)
    max_discharge_soc = discharge_room * eff / max(dt_hours, 1e-6)

    return -min(nominal, max_discharge_soc), min(nominal, max_charge_soc)


def _net_grid_kw(flat_payload: dict[str, float], actions: dict[str, float]) -> float:
    non_shiftable = max(0.0, _safe_float(flat_payload.get("non_shiftable_load"), 0.0))
    solar = max(0.0, _safe_float(flat_payload.get("solar_generation"), 0.0))
    return (
        non_shiftable
        + _safe_float(actions.get("cooling_kw"), 0.0)
        + _safe_float(actions.get("dhw_heater_kw"), 0.0)
        + _safe_float(actions.get("ev_charge_kw"), 0.0)
        + _safe_float(actions.get("battery_kw"), 0.0)
        - solar
    )


def _step_cost_with_export(net_grid_kw: float, price: float, dt_hours: float, export_factor: float) -> float:
    imported = max(net_grid_kw, 0.0)
    exported = max(-net_grid_kw, 0.0)
    return imported * price * dt_hours - exported * price * export_factor * dt_hours


def _step_cost_baseline_no_export(net_grid_kw: float, price: float, dt_hours: float) -> float:
    imported = max(net_grid_kw, 0.0)
    return imported * price * dt_hours


def _baseline_actions(flat_payload: dict[str, float], cfg) -> dict[str, float]:  # noqa: ANN001
    threshold = float(cfg.baseline_price_threshold_eur_kwh)
    price = max(0.0, _safe_float(flat_payload.get("electricity_pricing.current"), 0.0))
    import_limit = _safe_float(flat_payload.get("grid.import_limit_kw"), float(cfg.grid_import_limit_kw))
    export_limit = _safe_float(flat_payload.get("grid.export_limit_kw"), import_limit)

    ev_connected = False
    ev_max = 0.0
    ev_min = 0.0
    for charger_id, meta in cfg.chargers.items():
        ev_id = str(flat_payload.get(f"charging_sessions.{charger_id}.electric_vehicle", "")).strip()
        if not ev_id:
            continue
        ev_connected = True
        max_kw = max(_safe_float(meta.get("max_kw"), 22.0), 0.0)
        min_kw = max(_safe_float(meta.get("min_kw"), 0.0), float(cfg.ev_min_connected_kw))
        ev_max += max_kw
        ev_min += min_kw

    ev_kw = ev_max if (ev_connected and price <= threshold) else ev_min

    cooling_temp = flat_payload.get("cooling.temperature.current_c")
    cooling_min = flat_payload.get("cooling.temperature.min_c")
    cooling_max = flat_payload.get("cooling.temperature.max_c")
    cooling_nominal = float(cfg.cooling_nominal_power_kw)
    safe_frac = float(cfg.fallback_safe_power_fraction)
    if cooling_temp is None or cooling_min is None or cooling_max is None:
        cooling_kw = cooling_nominal * safe_frac
    else:
        cooling_kw = cooling_nominal if _safe_float(cooling_temp) > _safe_float(cooling_max) else 0.0

    dhw_temp = flat_payload.get("dhw.temperature.current_c")
    dhw_min = flat_payload.get("dhw.temperature.min_c")
    dhw_max = flat_payload.get("dhw.temperature.max_c")
    dhw_nominal = float(cfg.dhw_nominal_power_kw)
    if dhw_temp is None or dhw_min is None or dhw_max is None:
        dhw_kw = dhw_nominal * safe_frac
    else:
        dhw_kw = dhw_nominal if _safe_float(dhw_temp) < _safe_float(dhw_min) else 0.0

    soc = min(max(_safe_float(flat_payload.get("electrical_storage.soc"), 0.5), 0.0), 1.0)
    dt_hours = max(float(cfg.control_interval_minutes) / 60.0, 1.0 / 60.0)
    batt_min, batt_max = _battery_bounds(soc, dt_hours, cfg)
    battery_kw = batt_max if price <= threshold else batt_min

    base = (
        max(0.0, _safe_float(flat_payload.get("non_shiftable_load"), 0.0))
        + cooling_kw
        + dhw_kw
        + ev_kw
        - max(0.0, _safe_float(flat_payload.get("solar_generation"), 0.0))
    )
    net = base + battery_kw

    if net > import_limit:
        delta = net - import_limit
        battery_kw = min(max(battery_kw - delta, batt_min), batt_max)
    net = base + battery_kw
    if net > import_limit:
        delta = net - import_limit
        ev_kw = max(ev_kw - delta, 0.0)

    base = (
        max(0.0, _safe_float(flat_payload.get("non_shiftable_load"), 0.0))
        + cooling_kw
        + dhw_kw
        + ev_kw
        - max(0.0, _safe_float(flat_payload.get("solar_generation"), 0.0))
    )
    net = base + battery_kw
    if net < -export_limit:
        delta = -export_limit - net
        battery_kw = min(max(battery_kw + delta, batt_min), batt_max)

    return {
        "ev_charge_kw": float(ev_kw),
        "battery_kw": float(battery_kw),
        "cooling_kw": float(cooling_kw),
        "dhw_heater_kw": float(dhw_kw),
    }


@pytest.fixture
def rh1_client():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0, ALIAS_PATH)
    client = TestClient(app)
    try:
        yield client
    finally:
        if store.is_configured():
            store.unload()


def test_rh1_bundle_loads_and_strategy_selected(rh1_client):
    pipeline = store.get_pipeline()
    assert pipeline.agent.strategy == "rh1_house_rbc_v1"

    message = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))
    response = rh1_client.post("/inference", json=message)
    assert response.status_code == 200

    actions = response.json()["actions"]["0"]
    assert set(actions.keys()) == {"ev_charge_kw", "battery_kw", "cooling_kw", "dhw_heater_kw"}


def test_rh1_actions_respect_limits_and_grid_across_sequence(rh1_client):
    sequence = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))
    cfg = store.get_pipeline().agent._rh1_runtime.config  # noqa: SLF001

    for step in sequence:
        response = rh1_client.post("/inference", json={"features": step})
        assert response.status_code == 200, step["description"]

        actions = response.json()["actions"]["0"]
        flat = flatten_payload(step)
        connected = bool(str(flat.get("charging_sessions.EVC01.electric_vehicle", "")).strip())

        ev_max = 22.0 if connected else 0.0
        assert 0.0 <= actions["ev_charge_kw"] <= ev_max + 1e-6
        assert -cfg.battery_nominal_power_kw - 1e-6 <= actions["battery_kw"] <= cfg.battery_nominal_power_kw + 1e-6
        assert 0.0 <= actions["cooling_kw"] <= cfg.cooling_nominal_power_kw + 1e-6
        assert 0.0 <= actions["dhw_heater_kw"] <= cfg.dhw_nominal_power_kw + 1e-6

        net = _net_grid_kw(flat, actions)
        import_limit = _safe_float(flat.get("grid.import_limit_kw"), float(cfg.grid_import_limit_kw))
        export_limit = _safe_float(flat.get("grid.export_limit_kw"), import_limit)
        assert net <= import_limit + 1e-6
        assert net >= -export_limit - 1e-6


def test_rh1_ev_hard_deadline_behavior(rh1_client):
    payload = {
        "timestamp": "2026-02-22T10:00:00Z",
        "non_shiftable_load": 0.5,
        "solar_generation": 0.0,
        "electricity_pricing": {
            "current": 0.25,
            "h1": 0.20,
            "h2": 0.18,
            "h6": 0.15,
            "h12": 0.14,
            "h24": 0.13,
        },
        "grid": {"import_limit_kw": 40.0, "export_limit_kw": 20.0},
        "electrical_storage": {"soc": 0.5},
        "cooling": {"temperature": {"current_c": 23.0, "min_c": 21.0, "max_c": 25.0}},
        "dhw": {"temperature": {"current_c": 50.0, "min_c": 47.0, "max_c": 56.0}},
        "charging_sessions": {
            "EVC01": {"power": 0.0, "electric_vehicle": "EV1"}
        },
        "electric_vehicles": {
            "EV1": {
                "SoC": 0.30,
                "flexibility": {
                    "estimated_soc_at_departure": 0.90,
                    "estimated_time_at_departure": "2026-02-22T11:00:00Z",
                },
            }
        },
    }

    response = rh1_client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    ev_action = response.json()["actions"]["0"]["ev_charge_kw"]

    now = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
    departure = datetime.fromisoformat(
        payload["electric_vehicles"]["EV1"]["flexibility"]["estimated_time_at_departure"].replace("Z", "+00:00")
    )
    gap_kwh = (0.90 - 0.30) * 60.0
    minutes_remaining = max((departure - now).total_seconds() / 60.0, 15.0)
    required_kw = gap_kwh / (minutes_remaining / 60.0)
    expected_floor = min(max(required_kw, 0.0), 22.0)

    assert ev_action + 0.11 >= expected_floor


def test_rh1_ev_without_flex_is_treated_as_non_flex(rh1_client):
    payload = {
        "timestamp": "2026-02-22T20:00:00Z",
        "non_shiftable_load": 1.0,
        "solar_generation": 0.0,
        "electricity_pricing": {
            "current": 0.35,
            "h1": 0.30,
            "h2": 0.26,
            "h6": 0.20,
            "h12": 0.18,
            "h24": 0.17,
        },
        "grid": {"import_limit_kw": 8.0, "export_limit_kw": 5.0},
        "electrical_storage": {"soc": 0.55},
        "cooling": {"temperature": {"current_c": 23.0, "min_c": 21.0, "max_c": 25.0}},
        "dhw": {"temperature": {"current_c": 50.0, "min_c": 47.0, "max_c": 56.0}},
        "charging_sessions": {
            "EVC01": {"power": 0.0, "electric_vehicle": "EV1"}
        },
        "electric_vehicles": {
            "EV1": {
                "SoC": 0.50,
                "flexibility": {
                    "estimated_soc_at_departure": -1,
                    "estimated_time_at_departure": "",
                },
            }
        },
    }

    response = rh1_client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    actions = response.json()["actions"]["0"]
    assert actions["ev_charge_kw"] == pytest.approx(0.0, rel=1e-6)


def test_rh1_missing_temperature_data_uses_safe_mode(rh1_client):
    payload = {
        "timestamp": "2026-02-23T02:00:00Z",
        "non_shiftable_load": 1.0,
        "solar_generation": 0.0,
        "electricity_pricing": {
            "current": 0.18,
            "h1": 0.19,
            "h2": 0.20,
            "h6": 0.22,
            "h12": 0.21,
            "h24": 0.18,
        },
        "grid": {"import_limit_kw": 20.0, "export_limit_kw": 5.0},
        "electrical_storage": {"soc": 0.5},
        "charging_sessions": {
            "EVC01": {"power": 0.0, "electric_vehicle": ""}
        },
        "electric_vehicles": {},
    }

    response = rh1_client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    actions = response.json()["actions"]["0"]
    assert actions["cooling_kw"] == pytest.approx(1.2, rel=1e-6)
    assert actions["dhw_heater_kw"] == pytest.approx(1.4, rel=1e-6)


def test_rh1_price_interpolation_influences_battery_dispatch(rh1_client):
    common = {
        "timestamp": "2026-02-23T08:00:00Z",
        "non_shiftable_load": 1.0,
        "solar_generation": 0.0,
        "grid": {"import_limit_kw": 12.0, "export_limit_kw": 6.0},
        "electrical_storage": {"soc": 0.55},
        "cooling": {"temperature": {"current_c": 23.0, "min_c": 21.0, "max_c": 25.0}},
        "dhw": {"temperature": {"current_c": 50.0, "min_c": 47.0, "max_c": 56.0}},
        "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
        "electric_vehicles": {},
    }

    cheap_now = dict(common)
    cheap_now["electricity_pricing"] = {
        "current": 0.10,
        "h1": 0.15,
        "h2": 0.20,
        "h6": 0.26,
        "h12": 0.24,
        "h24": 0.18,
    }

    expensive_now = dict(common)
    expensive_now["electricity_pricing"] = {
        "current": 0.32,
        "h1": 0.27,
        "h2": 0.22,
        "h6": 0.16,
        "h12": 0.14,
        "h24": 0.12,
    }

    cheap_resp = rh1_client.post("/inference", json={"features": cheap_now})
    expensive_resp = rh1_client.post("/inference", json={"features": expensive_now})

    assert cheap_resp.status_code == 200
    assert expensive_resp.status_code == 200

    cheap_battery = cheap_resp.json()["actions"]["0"]["battery_kw"]
    expensive_battery = expensive_resp.json()["actions"]["0"]["battery_kw"]

    assert cheap_battery > 0.0
    assert expensive_battery < 0.0


def test_rh1_cost_is_better_than_baseline_on_sequence(rh1_client):
    cfg = store.get_pipeline().agent._rh1_runtime.config  # noqa: SLF001
    sequence = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))

    rbc_cost = 0.0
    baseline_cost = 0.0
    dt_hours = max(float(cfg.control_interval_minutes) / 60.0, 1.0 / 60.0)

    for step in sequence:
        response = rh1_client.post("/inference", json={"features": step})
        assert response.status_code == 200, step["description"]

        rbc_actions = response.json()["actions"]["0"]
        flat = flatten_payload(step)
        price_now = max(0.0, _safe_float(flat.get("electricity_pricing.current"), 0.0))

        rbc_net = _net_grid_kw(flat, rbc_actions)
        rbc_cost += _step_cost_with_export(
            rbc_net,
            price_now,
            dt_hours,
            float(cfg.export_price_factor),
        )

        baseline_actions = _baseline_actions(flat, cfg)
        baseline_net = _net_grid_kw(flat, baseline_actions)
        baseline_cost += _step_cost_baseline_no_export(baseline_net, price_now, dt_hours)

    assert rbc_cost < baseline_cost


def test_rh1_smoke_with_example_message(rh1_client):
    message = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))
    response = rh1_client.post("/inference", json=message)
    assert response.status_code == 200
    body = response.json()
    assert "actions" in body
    assert "0" in body["actions"]
