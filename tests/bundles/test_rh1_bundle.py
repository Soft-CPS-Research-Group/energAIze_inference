from __future__ import annotations

import copy
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
MESSAGE_PATH = BUNDLE_DIR / "exemplos_mensagem_R-H-01_2303.json"


def _safe_float(value, default: float = 0.0) -> float:  # noqa: ANN001
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _price_curve(now: float, h1: float, h2: float, h6: float, h12: float, h24: float) -> dict:
    values = [now] * 96
    values[4] = h1
    values[8] = h2
    values[24] = h6
    values[48] = h12
    values[95] = h24
    return {
        "values": values,
        "measurement_unit": "€/kWh",
        "frequency_seconds": 900,
        "horizon_seconds": 86400,
    }


def _run(client: TestClient, payload: dict) -> dict[str, float]:
    response = client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    return response.json()["actions"]["0"]


def _net_grid_kw(record: dict, actions: dict[str, float]) -> float:
    obs = record.get("observations", {})
    non_shiftable = max(0.0, _safe_float(obs.get("non_shiftable_load"), 0.0))
    solar = max(0.0, _safe_float(obs.get("solar_generation"), 0.0))
    return non_shiftable + _safe_float(actions.get("ev_charge_kw"), 0.0) + _safe_float(actions.get("battery_kw"), 0.0) - solar


def _step_cost_with_export(net_grid_kw: float, price: float, dt_hours: float, export_factor: float) -> float:
    imported = max(net_grid_kw, 0.0)
    exported = max(-net_grid_kw, 0.0)
    return imported * price * dt_hours - exported * price * export_factor * dt_hours


def _step_cost_baseline_no_export(net_grid_kw: float, price: float, dt_hours: float) -> float:
    imported = max(net_grid_kw, 0.0)
    return imported * price * dt_hours


def _baseline_actions(record: dict, cfg) -> dict[str, float]:  # noqa: ANN001
    obs = record.get("observations", {})
    threshold = float(cfg.baseline_price_threshold_eur_kwh)
    values = obs.get("energy_price", {}).get("values", [])
    price = _safe_float(values[0] if values else 0.0, 0.0)

    ev_id = str(obs.get("charging_sessions", {}).get("EVC01", {}).get("electric_vehicle") or "").strip()
    charger_meta = cfg.chargers.get("EVC01", {})
    ev_min = max(_safe_float(charger_meta.get("min_kw"), cfg.ev_min_connected_kw), cfg.ev_min_connected_kw)
    ev_max = max(_safe_float(charger_meta.get("max_kw"), 4.6), ev_min)
    ev_kw = 0.0
    if ev_id:
        ev_kw = ev_max if price <= threshold else ev_min

    soc = _safe_float(obs.get("batteries", {}).get("B01", {}).get("SoC"), 50.0)
    if soc > 1.0:
        soc /= 100.0
    soc = min(max(soc, 0.0), 1.0)

    dt_hours = max(float(cfg.control_interval_minutes) / 60.0, 1.0 / 60.0)
    cap = max(float(cfg.battery_capacity_kwh), 1e-6)
    eff = min(max(float(cfg.battery_efficiency), 1e-6), 1.0)
    soc_min = min(max(float(cfg.battery_soc_min), 0.0), 1.0)
    soc_max = min(max(float(cfg.battery_soc_max), 0.0), 1.0)
    nominal = max(float(cfg.battery_nominal_power_kw), 0.0)

    charge_room = max(soc_max - soc, 0.0) * cap
    discharge_room = max(soc - soc_min, 0.0) * cap
    batt_max = min(nominal, charge_room / max(dt_hours * eff, 1e-6))
    batt_min = -min(nominal, discharge_room * eff / max(dt_hours, 1e-6))
    battery_kw = batt_max if price <= threshold else batt_min

    return {"ev_charge_kw": float(ev_kw), "battery_kw": float(battery_kw)}


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


def test_rh1_actions_contract_ev_battery_only(rh1_client):
    message = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))[0]
    actions = _run(rh1_client, message)
    assert set(actions.keys()) == {"ev_charge_kw", "battery_kw"}


def test_rh1_replay_real_sequence_parses_and_runs(rh1_client):
    sequence = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))
    cfg = store.get_pipeline().agent._rh1_runtime.config  # noqa: SLF001

    for step in sequence:
        actions = _run(rh1_client, step)
        assert 0.0 <= actions["ev_charge_kw"] <= 4.6 + 1e-6
        assert -cfg.battery_nominal_power_kw - 1e-6 <= actions["battery_kw"] <= cfg.battery_nominal_power_kw + 1e-6


def test_rh1_soc_auto_percent_normalization(rh1_client):
    payload = {
        "timestamp": "2026-03-01T10:00:00Z",
        "observations": {
            "non_shiftable_load": 0.4,
            "solar_generation": 0.0,
            "energy_price": _price_curve(0.12, 0.22, 0.21, 0.20, 0.19, 0.18),
            "batteries": {"B01": {"SoC": 89.0}},
            "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
            "electric_vehicles": {},
        },
        "forecasts": {},
    }
    actions = _run(rh1_client, payload)
    assert -4.8 <= actions["battery_kw"] <= 4.8


def test_rh1_price_vector_normalization_affects_dispatch(rh1_client):
    common_obs = {
        "non_shiftable_load": 1.0,
        "solar_generation": 0.0,
        "batteries": {"B01": {"SoC": 55.0}},
        "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
        "electric_vehicles": {},
    }

    cheap = {
        "timestamp": "2026-03-01T12:00:00Z",
        "observations": dict(common_obs),
        "forecasts": {},
    }
    cheap["observations"]["energy_price"] = _price_curve(0.05, 0.25, 0.24, 0.23, 0.22, 0.21)

    expensive = {
        "timestamp": "2026-03-01T13:00:00Z",
        "observations": dict(common_obs),
        "forecasts": {},
    }
    expensive["observations"]["energy_price"] = _price_curve(0.30, 0.10, 0.09, 0.08, 0.07, 0.06)

    cheap_actions = _run(rh1_client, cheap)
    expensive_actions = _run(rh1_client, expensive)

    assert cheap_actions["battery_kw"] > 0.0
    assert expensive_actions["battery_kw"] < 0.0


def test_rh1_ev_hard_deadline_behavior(rh1_client):
    payload = {
        "timestamp": "2026-03-01T10:00:00Z",
        "observations": {
            "non_shiftable_load": 0.5,
            "solar_generation": 0.0,
            "energy_price": _price_curve(0.25, 0.20, 0.18, 0.15, 0.14, 0.13),
            "batteries": {"B01": {"SoC": 50.0}},
            "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": "EV01"}},
            "electric_vehicles": {
                "EV01": {
                    "SoC": 0.30,
                    "flexibility": {
                        "estimated_soc_at_departure": 0.90,
                        "estimated_time_at_departure": "2026-03-01T11:00:00Z",
                    },
                }
            },
        },
        "forecasts": {},
    }

    actions = _run(rh1_client, payload)
    now = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
    departure = datetime.fromisoformat(
        payload["observations"]["electric_vehicles"]["EV01"]["flexibility"]["estimated_time_at_departure"].replace("Z", "+00:00")
    )
    gap_kwh = (0.90 - 0.30) * 60.0
    minutes_remaining = max((departure - now).total_seconds() / 60.0, 1.0)
    required_kw = gap_kwh / (minutes_remaining / 60.0)
    expected_floor = min(max(required_kw, 0.0), 4.6)

    assert actions["ev_charge_kw"] + 0.11 >= expected_floor


def test_rh1_ev_without_flex_is_treated_as_non_flex(rh1_client):
    payload = {
        "timestamp": "2026-03-01T20:00:00Z",
        "observations": {
            "non_shiftable_load": 1.0,
            "solar_generation": 0.0,
            "energy_price": _price_curve(0.35, 0.32, 0.30, 0.28, 0.26, 0.24),
            "batteries": {"B01": {"SoC": 55.0}},
            "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": "EV01"}},
            "electric_vehicles": {
                "EV01": {
                    "SoC": 0.50,
                    "flexibility": {
                        "estimated_soc_at_departure": -1,
                        "estimated_time_at_departure": "",
                    },
                }
            },
        },
        "forecasts": {},
    }

    actions = _run(rh1_client, payload)
    assert actions["ev_charge_kw"] == pytest.approx(1.6, rel=1e-6)


def test_rh1_cost_is_better_than_baseline_on_synthetic_sequence(rh1_client):
    cfg = store.get_pipeline().agent._rh1_runtime.config  # noqa: SLF001
    sequence = [
        {
            "timestamp": "2026-03-01T09:00:00Z",
            "observations": {
                "non_shiftable_load": 2.8,
                "solar_generation": 0.0,
                "energy_price": _price_curve(0.06, 0.20, 0.22, 0.24, 0.23, 0.21),
                "batteries": {"B01": {"SoC": 55.0}},
                "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
                "electric_vehicles": {},
            },
            "forecasts": {},
        },
        {
            "timestamp": "2026-03-01T12:00:00Z",
            "observations": {
                "non_shiftable_load": 0.5,
                "solar_generation": 2.5,
                "energy_price": _price_curve(0.07, 0.18, 0.20, 0.22, 0.21, 0.19),
                "batteries": {"B01": {"SoC": 65.0}},
                "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
                "electric_vehicles": {},
            },
            "forecasts": {},
        },
        {
            "timestamp": "2026-03-01T18:00:00Z",
            "observations": {
                "non_shiftable_load": 3.0,
                "solar_generation": 0.0,
                "energy_price": _price_curve(0.30, 0.10, 0.09, 0.08, 0.08, 0.07),
                "batteries": {"B01": {"SoC": 70.0}},
                "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
                "electric_vehicles": {},
            },
            "forecasts": {},
        },
        {
            "timestamp": "2026-03-01T21:00:00Z",
            "observations": {
                "non_shiftable_load": 2.9,
                "solar_generation": 0.0,
                "energy_price": _price_curve(0.28, 0.12, 0.10, 0.09, 0.08, 0.07),
                "batteries": {"B01": {"SoC": 60.0}},
                "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
                "electric_vehicles": {},
            },
            "forecasts": {},
        },
    ]

    dt_hours = max(float(cfg.control_interval_minutes) / 60.0, 1.0 / 60.0)
    rbc_cost = 0.0
    baseline_cost = 0.0

    for step in sequence:
        rbc_actions = _run(rh1_client, step)
        obs_flat = flatten_payload(step["observations"])
        runtime = store.get_pipeline().agent._rh1_runtime  # noqa: SLF001
        prices = runtime._extract_price_points(obs_flat, [])  # noqa: SLF001
        price_now = prices[0.0]

        rbc_net = _net_grid_kw(step, rbc_actions)
        rbc_cost += _step_cost_with_export(rbc_net, price_now, dt_hours, float(cfg.export_price_factor))

        base_actions = _baseline_actions(step, cfg)
        base_net = _net_grid_kw(step, base_actions)
        baseline_cost += _step_cost_baseline_no_export(base_net, price_now, dt_hours)

    assert rbc_cost < baseline_cost


def test_rh1_smoke_with_example_message(rh1_client):
    message = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))[0]
    response = rh1_client.post("/inference", json={"features": message})
    assert response.status_code == 200
    body = response.json()
    assert "actions" in body
    assert "0" in body["actions"]
