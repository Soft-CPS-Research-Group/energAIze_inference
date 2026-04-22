from __future__ import annotations

import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store


BUNDLE_DIR = Path("examples/rh1_bundle_community")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
ALIAS_PATH = BUNDLE_DIR / "aliases.json"
ACTION_EV = "EVC01"
ACTION_BATTERY = "B01"
DECISION_INTERVAL_HOURS = 15.0 / 3600.0


def _kwh_for_interval(power_kw: float) -> float:
    return power_kw * DECISION_INTERVAL_HOURS


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


def _set_site_forecasts(
    payload: dict,
    *,
    consumption_values: list[float] | None = None,
    production_values: list[float] | None = None,
    frequency_seconds: int = 900,
) -> None:
    forecasts = payload.setdefault("forecasts", {})
    if consumption_values is not None:
        forecasts["ConsumptionForecastService"] = {
            "consumption_total": {
                "values": consumption_values,
                "measurement_unit": "kWh",
                "frequency_seconds": frequency_seconds,
                "horizon_seconds": len(consumption_values) * frequency_seconds,
            }
        }
    if production_values is not None:
        forecasts["ProductionForecastService"] = {
            "production_total": {
                "values": production_values,
                "measurement_unit": "kWh",
                "frequency_seconds": frequency_seconds,
                "horizon_seconds": len(production_values) * frequency_seconds,
            }
        }


@pytest.fixture
def rh1_community_client():
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
    return {
        "timestamp": "2026-03-01T10:00:00Z",
        "community": {
            "energy_in_total": 0.0,
            "energy_out_total": 0.0,
        },
        "observations": {
            "non_shiftable_load": 0.2,
            "solar_generation": 0.0,
            "energy_price": _price_curve(0.05, 0.25, 0.25, 0.25, 0.25, 0.25),
            "batteries": {"B01": {"SoC": 0.6}},
            "charging_sessions": {"EVC01": {"power": 0.0, "electric_vehicle": ""}},
            "electric_vehicles": {},
        },
        "forecasts": {},
    }


def _run(client: TestClient, payload: dict) -> dict[str, float]:
    response = client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    return response.json()["actions"]["0"]


def _net_grid_kw(payload: dict, actions: dict[str, float]) -> float:
    obs = payload["observations"]
    return (
        float(obs.get("non_shiftable_load", 0.0))
        + float(actions.get(ACTION_EV, 0.0))
        + float(actions.get(ACTION_BATTERY, 0.0))
        - float(obs.get("solar_generation", 0.0))
    )


def test_bundle_loads_rh1_community(rh1_community_client):
    pipeline = store.get_pipeline()
    assert pipeline.agent.strategy == "rh1_house_rbc_v1"
    runtime = pipeline.agent._rh1_runtime  # noqa: SLF001
    assert runtime is not None
    cfg = runtime.config
    assert cfg.price_quantile_cheap == pytest.approx(0.35, rel=1e-6)
    assert cfg.price_quantile_expensive == pytest.approx(0.7, rel=1e-6)
    assert cfg.reserve_soc_cheap == pytest.approx(0.35, rel=1e-6)
    assert cfg.reserve_soc_neutral == pytest.approx(0.3, rel=1e-6)
    assert cfg.reserve_soc_expensive == pytest.approx(0.25, rel=1e-6)
    assert cfg.target_soc_cheap == pytest.approx(0.85, rel=1e-6)
    assert cfg.target_soc_neutral == pytest.approx(0.7, rel=1e-6)
    assert cfg.target_soc_expensive == pytest.approx(0.6, rel=1e-6)
    assert cfg.forecast_support_enabled is True


def test_missing_required_community_fields_returns_400(rh1_community_client):
    missing_in = _base_payload()
    del missing_in["community"]["energy_in_total"]
    response = rh1_community_client.post("/inference", json={"features": missing_in})
    assert response.status_code == 400

    missing_out = _base_payload()
    del missing_out["community"]["energy_out_total"]
    response = rh1_community_client.post("/inference", json={"features": missing_out})
    assert response.status_code == 400


def test_community_deficit_bias_reduces_import_vs_surplus(rh1_community_client):
    deficit = _base_payload()
    deficit["community"]["energy_in_total"] = _kwh_for_interval(30.0)
    deficit["community"]["energy_out_total"] = 0.0

    surplus = copy.deepcopy(deficit)
    surplus["community"]["energy_in_total"] = 0.0
    surplus["community"]["energy_out_total"] = _kwh_for_interval(30.0)

    deficit_actions = _run(rh1_community_client, deficit)
    surplus_actions = _run(rh1_community_client, surplus)

    deficit_net = _net_grid_kw(deficit, deficit_actions)
    surplus_net = _net_grid_kw(surplus, surplus_actions)

    assert deficit_actions[ACTION_BATTERY] <= surplus_actions[ACTION_BATTERY]
    assert deficit_net <= surplus_net


def test_small_community_energy_interval_converts_to_kw(rh1_community_client):
    neutral = _base_payload()
    neutral["observations"]["energy_price"] = {
        "values": [0.30] + [0.10] * 95,
        "measurement_unit": "€/kWh",
        "frequency_seconds": 900,
        "horizon_seconds": 86400,
    }
    neutral_actions = _run(rh1_community_client, neutral)

    deficit_small = _base_payload()
    deficit_small["observations"]["energy_price"] = {
        "values": [0.30] + [0.10] * 95,
        "measurement_unit": "€/kWh",
        "frequency_seconds": 900,
        "horizon_seconds": 86400,
    }
    deficit_small["community"]["energy_in_total"] = _kwh_for_interval(1.774152)
    deficit_small["community"]["energy_out_total"] = 0.0
    deficit_small_actions = _run(rh1_community_client, deficit_small)

    assert deficit_small_actions[ACTION_BATTERY] < neutral_actions[ACTION_BATTERY]


def test_high_soc_and_community_deficit_can_still_discharge_when_price_is_favorable(
    rh1_community_client,
):
    payload = _base_payload()
    payload["observations"]["batteries"]["B01"]["SoC"] = 0.9
    payload["observations"]["energy_price"] = _price_curve(0.35, 0.08, 0.07, 0.06, 0.05, 0.05)
    payload["community"]["energy_in_total"] = _kwh_for_interval(30.0)
    payload["community"]["energy_out_total"] = 0.0

    actions = _run(rh1_community_client, payload)
    assert actions[ACTION_BATTERY] < -0.1


def test_community_deficit_does_not_force_discharge_when_price_is_zero(rh1_community_client):
    payload = _base_payload()
    payload["observations"]["batteries"]["B01"]["SoC"] = 0.9
    payload["observations"]["energy_price"] = _price_curve(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    payload["community"]["energy_in_total"] = _kwh_for_interval(30.0)
    payload["community"]["energy_out_total"] = 0.0

    actions = _run(rh1_community_client, payload)
    assert actions[ACTION_BATTERY] >= -0.1


def test_price_signal_still_trades_off_with_community_target(rh1_community_client):
    base = _base_payload()
    base["observations"]["batteries"]["B01"]["SoC"] = 0.9
    base["community"]["energy_in_total"] = _kwh_for_interval(30.0)
    base["community"]["energy_out_total"] = 0.0

    cheap_now = copy.deepcopy(base)
    cheap_now["observations"]["energy_price"] = _price_curve(0.05, 0.25, 0.24, 0.23, 0.22, 0.21)

    expensive_now = copy.deepcopy(base)
    expensive_now["observations"]["energy_price"] = _price_curve(0.30, 0.10, 0.09, 0.08, 0.07, 0.06)

    cheap_actions = _run(rh1_community_client, cheap_now)
    expensive_actions = _run(rh1_community_client, expensive_now)

    assert expensive_actions[ACTION_BATTERY] < cheap_actions[ACTION_BATTERY]


def test_forecast_deficit_tempers_battery_discharge_under_community_deficit(rh1_community_client):
    base = _base_payload()
    base["observations"]["batteries"]["B01"]["SoC"] = 0.55
    base["observations"]["energy_price"] = {
        "values": [0.25] + ([0.05] * 30) + ([0.20] * 30) + ([0.35] * 35),
        "measurement_unit": "€/kWh",
        "frequency_seconds": 900,
        "horizon_seconds": 86400,
    }
    base["community"]["energy_in_total"] = _kwh_for_interval(30.0)
    base["community"]["energy_out_total"] = 0.0

    forecasted = copy.deepcopy(base)
    _set_site_forecasts(
        forecasted,
        consumption_values=[1.0] * 8,
        production_values=[0.0] * 8,
    )

    base_actions = _run(rh1_community_client, base)
    forecast_actions = _run(rh1_community_client, forecasted)

    assert forecast_actions[ACTION_BATTERY] > base_actions[ACTION_BATTERY] + 0.1


def test_community_bias_still_applies_when_local_forecast_is_present(rh1_community_client):
    neutral = _base_payload()
    neutral["observations"]["batteries"]["B01"]["SoC"] = 0.55
    neutral["observations"]["energy_price"] = _price_curve(0.20, 0.20, 0.20, 0.20, 0.20, 0.20)
    _set_site_forecasts(
        neutral,
        consumption_values=[1.0] * 8,
        production_values=[0.0] * 8,
    )

    deficit = copy.deepcopy(neutral)
    deficit["community"]["energy_in_total"] = _kwh_for_interval(30.0)
    deficit["community"]["energy_out_total"] = 0.0

    neutral_actions = _run(rh1_community_client, neutral)
    deficit_actions = _run(rh1_community_client, deficit)

    assert deficit_actions[ACTION_BATTERY] <= neutral_actions[ACTION_BATTERY]


def test_cheap_price_and_low_soc_prefers_charging_before_community_discharge(rh1_community_client):
    payload = _base_payload()
    payload["observations"]["batteries"]["B01"]["SoC"] = 0.30
    payload["community"]["energy_in_total"] = _kwh_for_interval(30.0)
    payload["community"]["energy_out_total"] = 0.0
    payload["observations"]["energy_price"] = _price_curve(0.02, 0.20, 0.20, 0.20, 0.20, 0.20)

    actions = _run(rh1_community_client, payload)
    assert actions[ACTION_BATTERY] >= 1.0


def test_reserve_floor_blocks_discharge_near_target_soc_band(rh1_community_client):
    payload = _base_payload()
    payload["observations"]["batteries"]["B01"]["SoC"] = 0.30
    payload["community"]["energy_in_total"] = _kwh_for_interval(30.0)
    payload["community"]["energy_out_total"] = 0.0
    payload["observations"]["energy_price"] = _price_curve(0.10, 0.05, 0.15, 0.15, 0.05, 0.15)

    actions = _run(rh1_community_client, payload)
    assert actions[ACTION_BATTERY] >= -0.1


def test_connected_ev_keeps_minimum_and_is_not_modulated_by_community(rh1_community_client):
    base = _base_payload()
    base["observations"]["charging_sessions"]["EVC01"] = {
        "power": 0.0,
        "electric_vehicle": "EV01",
    }
    base["observations"]["electric_vehicles"] = {
        "EV01": {
            "SoC": 0.4,
            "flexibility": {
                "estimated_soc_at_departure": 0.9,
                "estimated_time_at_departure": "2026-03-01T12:00:00Z",
            },
        }
    }
    base["observations"]["energy_price"] = _price_curve(0.35, 0.25, 0.23, 0.22, 0.20, 0.18)

    deficit = copy.deepcopy(base)
    deficit["community"]["energy_in_total"] = _kwh_for_interval(30.0)
    deficit["community"]["energy_out_total"] = 0.0

    surplus = copy.deepcopy(base)
    surplus["community"]["energy_in_total"] = 0.0
    surplus["community"]["energy_out_total"] = _kwh_for_interval(30.0)

    deficit_actions = _run(rh1_community_client, deficit)
    surplus_actions = _run(rh1_community_client, surplus)

    assert deficit_actions[ACTION_EV] >= 1.6 - 1e-6
    assert surplus_actions[ACTION_EV] >= 1.6 - 1e-6
    assert deficit_actions[ACTION_EV] == pytest.approx(surplus_actions[ACTION_EV], rel=1e-6)
