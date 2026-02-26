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
REAL_SEQUENCE_PATH = BUNDLE_DIR / "exemplos_reais_mensagem.json"
MESSAGE_PATH = BUNDLE_DIR / "rh1_message_example.json"


def _safe_float(value, default: float = 0.0) -> float:  # noqa: ANN001
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_price(value: float, cfg) -> float:  # noqa: ANN001
    mode = str(getattr(cfg, "price_unit_mode", "auto"))
    threshold = float(getattr(cfg, "price_auto_mwh_threshold", 3.0))
    if mode == "eur_mwh":
        value /= 1000.0
    elif mode == "auto" and value > threshold:
        value /= 1000.0
    return value


def _extract_price_now(flat_payload: dict[str, float], cfg) -> float:  # noqa: ANN001
    for key in ("electricity_pricing.current", "electricity_pricing", "energy_price"):
        if key in flat_payload:
            return max(_normalize_price(_safe_float(flat_payload.get(key), 0.0), cfg), 0.0)
    return 0.0


def _normalize_soc(value: float, cfg) -> float:  # noqa: ANN001
    mode = str(getattr(cfg, "soc_unit_mode", "auto"))
    if mode == "percent":
        value /= 100.0
    elif mode == "auto" and 1.0 < value <= 100.0:
        value /= 100.0
    return min(max(value, 0.0), 1.0)


def _extract_battery_soc(flat_payload: dict[str, float], cfg) -> float:  # noqa: ANN001
    keys = list(getattr(cfg, "battery_soc_keys", ["electrical_storage.soc"]))
    for key in keys:
        if key in flat_payload:
            return _normalize_soc(_safe_float(flat_payload.get(key), 0.5), cfg)

    fallback_keys = sorted(
        key
        for key in flat_payload
        if key.lower().startswith("batteries.") and key.lower().endswith(".soc")
    )
    if fallback_keys:
        return _normalize_soc(_safe_float(flat_payload.get(fallback_keys[0]), 0.5), cfg)

    return 0.5


def _extract_solar(flat_payload: dict[str, float]) -> float:
    direct = flat_payload.get("solar_generation")
    if direct is not None:
        return max(_safe_float(direct, 0.0), 0.0)

    total = 0.0
    found = False
    for key, value in flat_payload.items():
        if key.startswith("pv_panels.") and key.endswith(".energy"):
            total += _safe_float(value, 0.0)
            found = True
    return max(total, 0.0) if found else 0.0


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
    solar = _extract_solar(flat_payload)
    return (
        non_shiftable
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
    price = _extract_price_now(flat_payload, cfg)
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

    soc = _extract_battery_soc(flat_payload, cfg)
    dt_hours = max(float(cfg.control_interval_minutes) / 60.0, 1.0 / 60.0)
    batt_min, batt_max = _battery_bounds(soc, dt_hours, cfg)
    battery_kw = batt_max if price <= threshold else batt_min

    base = max(0.0, _safe_float(flat_payload.get("non_shiftable_load"), 0.0)) + ev_kw - _extract_solar(flat_payload)
    net = base + battery_kw

    if net > import_limit:
        delta = net - import_limit
        battery_kw = min(max(battery_kw - delta, batt_min), batt_max)
    net = base + battery_kw
    if net > import_limit:
        delta = net - import_limit
        ev_kw = max(ev_kw - delta, 0.0)

    base = max(0.0, _safe_float(flat_payload.get("non_shiftable_load"), 0.0)) + ev_kw - _extract_solar(flat_payload)
    net = base + battery_kw
    if net < -export_limit:
        delta = -export_limit - net
        battery_kw = min(max(battery_kw + delta, batt_min), batt_max)

    return {
        "ev_charge_kw": float(ev_kw),
        "battery_kw": float(battery_kw),
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
    assert set(actions.keys()) == {"ev_charge_kw", "battery_kw"}


def test_rh1_actions_contract_ev_battery_only(rh1_client):
    message = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))
    response = rh1_client.post("/inference", json=message)
    assert response.status_code == 200

    actions = response.json()["actions"]["0"]
    assert set(actions.keys()) == {"ev_charge_kw", "battery_kw"}
    assert "cooling_kw" not in actions
    assert "dhw_heater_kw" not in actions


def test_rh1_replay_real_sequence_parses_and_runs(rh1_client):
    sequence = json.loads(REAL_SEQUENCE_PATH.read_text(encoding="utf-8"))
    cfg = store.get_pipeline().agent._rh1_runtime.config  # noqa: SLF001
    charger_max = _safe_float(cfg.chargers.get("EVC01", {}).get("max_kw"), 4.6)

    for step in sequence:
        response = rh1_client.post("/inference", json={"features": step})
        assert response.status_code == 200

        actions = response.json()["actions"]["0"]
        flat = flatten_payload(step)
        connected = bool(str(flat.get("charging_sessions.EVC01.electric_vehicle", "")).strip())

        ev_max = charger_max if connected else 0.0
        assert 0.0 <= actions["ev_charge_kw"] <= ev_max + 1e-6
        assert -cfg.battery_nominal_power_kw - 1e-6 <= actions["battery_kw"] <= cfg.battery_nominal_power_kw + 1e-6

        net = _net_grid_kw(flat, actions)
        import_limit = _safe_float(flat.get("grid.import_limit_kw"), float(cfg.grid_import_limit_kw))
        export_limit = _safe_float(flat.get("grid.export_limit_kw"), import_limit)
        assert net <= import_limit + 1e-6
        assert net >= -export_limit - 1e-6


def test_rh1_soc_auto_percent_normalization(rh1_client):
    cfg = store.get_pipeline().agent._rh1_runtime.config  # noqa: SLF001
    payload = {
        "timestamp": "2026-02-22T10:00:00Z",
        "non_shiftable_load": 0.4,
        "solar_generation": 0.0,
        "energy_price": 120.0,
        "electricity_pricing": {
            "h1": 220.0,
            "h2": 210.0,
            "h6": 200.0,
            "h12": 190.0,
            "h24": 180.0,
        },
        "grid": {"import_limit_kw": 12.0, "export_limit_kw": 12.0},
        "batteries": {"B01": {"SoC": 89.0, "energy_in": 0.0, "energy_out": 0.0}},
        "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
        "electric_vehicles": {},
    }

    response = rh1_client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    actions = response.json()["actions"]["0"]

    assert -cfg.battery_nominal_power_kw <= actions["battery_kw"] <= cfg.battery_nominal_power_kw
    assert actions["battery_kw"] > 0.0


def test_rh1_price_auto_mwh_normalization_affects_dispatch(rh1_client):
    common = {
        "timestamp": "2026-02-22T12:00:00Z",
        "non_shiftable_load": 1.0,
        "solar_generation": 0.0,
        "grid": {"import_limit_kw": 20.0, "export_limit_kw": 20.0},
        "electrical_storage": {"soc": 0.55},
        "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
        "electric_vehicles": {},
    }

    cheap_now = dict(common)
    cheap_now["energy_price"] = 100.0
    cheap_now["electricity_pricing"] = {
        "h1": 180.0,
        "h2": 200.0,
        "h6": 230.0,
        "h12": 240.0,
        "h24": 220.0,
    }

    expensive_now = dict(common)
    expensive_now["energy_price"] = 320.0
    expensive_now["electricity_pricing"] = {
        "h1": 260.0,
        "h2": 210.0,
        "h6": 160.0,
        "h12": 140.0,
        "h24": 120.0,
    }

    cheap_resp = rh1_client.post("/inference", json={"features": cheap_now})
    expensive_resp = rh1_client.post("/inference", json={"features": expensive_now})

    assert cheap_resp.status_code == 200
    assert expensive_resp.status_code == 200

    cheap_battery = cheap_resp.json()["actions"]["0"]["battery_kw"]
    expensive_battery = expensive_resp.json()["actions"]["0"]["battery_kw"]

    assert cheap_battery > 0.0
    assert expensive_battery < 0.0


def test_rh1_ev_hard_deadline_behavior(rh1_client):
    cfg = store.get_pipeline().agent._rh1_runtime.config  # noqa: SLF001
    payload = {
        "timestamp": "2026-02-22T10:00:00Z",
        "non_shiftable_load": 0.5,
        "solar_generation": 0.0,
        "energy_price": 250.0,
        "electricity_pricing": {
            "h1": 200.0,
            "h2": 180.0,
            "h6": 150.0,
            "h12": 140.0,
            "h24": 130.0,
        },
        "grid": {"import_limit_kw": 40.0, "export_limit_kw": 20.0},
        "batteries": {"B01": {"SoC": 50.0, "energy_in": 0.0, "energy_out": 0.0}},
        "charging_sessions": {
            "EVC01": {"power": 0.0, "electric_vehicle": "EV01"}
        },
        "electric_vehicles": {
            "EV01": {
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
        payload["electric_vehicles"]["EV01"]["flexibility"]["estimated_time_at_departure"].replace("Z", "+00:00")
    )
    gap_kwh = (0.90 - 0.30) * 60.0
    minutes_remaining = max((departure - now).total_seconds() / 60.0, 1.0)
    required_kw = gap_kwh / (minutes_remaining / 60.0)
    charger_max = _safe_float(cfg.chargers.get("EVC01", {}).get("max_kw"), 4.6)
    expected_floor = min(max(required_kw, 0.0), charger_max)

    assert ev_action + 0.11 >= expected_floor


def test_rh1_ev_without_flex_is_treated_as_non_flex(rh1_client):
    cfg = store.get_pipeline().agent._rh1_runtime.config  # noqa: SLF001
    payload = {
        "timestamp": "2026-02-22T20:00:00Z",
        "non_shiftable_load": 1.0,
        "solar_generation": 0.0,
        "energy_price": 350.0,
        "electricity_pricing": {
            "h1": 300.0,
            "h2": 260.0,
            "h6": 200.0,
            "h12": 180.0,
            "h24": 170.0,
        },
        "grid": {"import_limit_kw": 8.0, "export_limit_kw": 5.0},
        "batteries": {"B01": {"SoC": 55.0, "energy_in": 0.0, "energy_out": 0.0}},
        "charging_sessions": {
            "EVC01": {"power": 0.0, "electric_vehicle": "EV01"}
        },
        "electric_vehicles": {
            "EV01": {
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
    assert actions["ev_charge_kw"] == pytest.approx(float(cfg.ev_min_connected_kw), rel=1e-6)


def test_rh1_cost_is_better_than_baseline_on_sequence(rh1_client):
    cfg = store.get_pipeline().agent._rh1_runtime.config  # noqa: SLF001
    sequence = [
        {
            "timestamp": "2026-02-22T09:00:00Z",
            "non_shiftable_load": 2.5,
            "solar_generation": 0.0,
            "energy_price": 190.0,
            "electricity_pricing": {"h1": 120.0, "h2": 100.0, "h6": 90.0, "h12": 95.0, "h24": 110.0},
            "grid": {"import_limit_kw": 8.0, "export_limit_kw": 5.0},
            "electrical_storage": {"soc": 0.70},
            "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
            "electric_vehicles": {},
        },
        {
            "timestamp": "2026-02-22T12:00:00Z",
            "non_shiftable_load": 2.6,
            "solar_generation": 0.0,
            "energy_price": 195.0,
            "electricity_pricing": {"h1": 130.0, "h2": 110.0, "h6": 100.0, "h12": 95.0, "h24": 105.0},
            "grid": {"import_limit_kw": 8.0, "export_limit_kw": 5.0},
            "electrical_storage": {"soc": 0.65},
            "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
            "electric_vehicles": {},
        },
        {
            "timestamp": "2026-02-22T15:00:00Z",
            "non_shiftable_load": 0.7,
            "solar_generation": 3.0,
            "energy_price": 185.0,
            "electricity_pricing": {"h1": 120.0, "h2": 95.0, "h6": 85.0, "h12": 90.0, "h24": 100.0},
            "grid": {"import_limit_kw": 8.0, "export_limit_kw": 2.0},
            "electrical_storage": {"soc": 0.60},
            "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
            "electric_vehicles": {},
        },
        {
            "timestamp": "2026-02-22T18:00:00Z",
            "non_shiftable_load": 2.4,
            "solar_generation": 0.0,
            "energy_price": 188.0,
            "electricity_pricing": {"h1": 125.0, "h2": 105.0, "h6": 95.0, "h12": 90.0, "h24": 98.0},
            "grid": {"import_limit_kw": 8.0, "export_limit_kw": 5.0},
            "electrical_storage": {"soc": 0.58},
            "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
            "electric_vehicles": {},
        },
    ]

    rbc_cost = 0.0
    baseline_cost = 0.0
    dt_hours = max(float(cfg.control_interval_minutes) / 60.0, 1.0 / 60.0)

    for step in sequence:
        response = rh1_client.post("/inference", json={"features": step})
        assert response.status_code == 200

        rbc_actions = response.json()["actions"]["0"]
        flat = flatten_payload(step)
        price_now = _extract_price_now(flat, cfg)

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
