from __future__ import annotations

import copy
from pathlib import Path
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store


BUNDLE_DIR = Path("examples/icharging_sao_mamede_with_virtual_battery_community")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
ALIAS_PATH = BUNDLE_DIR / "aliases.json"
MESSAGE_PATH = BUNDLE_DIR / "exemplos_mensagem_SaoMamede_2303.json"

ACTION_BATTERY = "B01"
DECISION_INTERVAL_HOURS = 15.0 / 3600.0


def _kwh_for_interval(power_kw: float) -> float:
    return power_kw * DECISION_INTERVAL_HOURS


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
def sao_mamede_with_battery_community_client():
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
    payload["observations"]["solar_generation"] = 0.0
    payload["observations"]["non_shiftable_load"] = 0.0
    payload["observations"]["batteries"] = {
        "B01": {
            "energy_in": 0.0,
            "energy_out": 0.0,
            "SoC": 0.70,
        }
    }
    payload["community"] = {
        "energy_in_total": 0.0,
        "energy_out_total": 0.0,
    }
    payload["forecasts"] = {}
    return payload


def _run(client: TestClient, payload: dict) -> dict[str, float]:
    response = client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    return response.json()["actions"]["0"]


def test_bundle_loads_sao_mamede_with_virtual_battery_community(
    sao_mamede_with_battery_community_client,
):
    pipeline = store.get_pipeline()
    assert pipeline.agent.strategy == "icharging_breaker"
    assert pipeline.agent._icharging_runtime is not None  # noqa: SLF001
    assert pipeline.agent._icharging_runtime.config.forecast_support_enabled is True  # noqa: SLF001


def test_missing_required_community_fields_returns_400(
    sao_mamede_with_battery_community_client,
):
    missing_in = _base_payload()
    del missing_in["community"]["energy_in_total"]
    response = sao_mamede_with_battery_community_client.post(
        "/inference", json={"features": missing_in}
    )
    assert response.status_code == 400

    missing_out = _base_payload()
    del missing_out["community"]["energy_out_total"]
    response = sao_mamede_with_battery_community_client.post(
        "/inference", json={"features": missing_out}
    )
    assert response.status_code == 400


def test_community_deficit_forces_virtual_battery_discharge(
    sao_mamede_with_battery_community_client,
):
    payload = _base_payload()
    payload["community"]["energy_in_total"] = _kwh_for_interval(40.0)
    payload["community"]["energy_out_total"] = 0.0

    actions = _run(sao_mamede_with_battery_community_client, payload)
    assert actions[ACTION_BATTERY] < -0.1


def test_community_surplus_forces_virtual_battery_charge(
    sao_mamede_with_battery_community_client,
):
    payload = _base_payload()
    payload["community"]["energy_in_total"] = 0.0
    payload["community"]["energy_out_total"] = _kwh_for_interval(40.0)

    actions = _run(sao_mamede_with_battery_community_client, payload)
    assert actions[ACTION_BATTERY] > 0.1


def test_small_community_energy_interval_converts_to_kw(
    sao_mamede_with_battery_community_client,
):
    neutral = _base_payload()
    neutral_actions = _run(sao_mamede_with_battery_community_client, neutral)

    deficit_small = _base_payload()
    deficit_small["community"]["energy_in_total"] = _kwh_for_interval(1.774152)
    deficit_small["community"]["energy_out_total"] = 0.0
    deficit_small_actions = _run(sao_mamede_with_battery_community_client, deficit_small)

    assert deficit_small_actions[ACTION_BATTERY] == pytest.approx(
        neutral_actions[ACTION_BATTERY], abs=0.1
    )


def test_low_soc_does_not_force_aggressive_discharge_under_community_deficit(
    sao_mamede_with_battery_community_client,
):
    payload = _base_payload()
    payload["observations"]["batteries"]["B01"]["SoC"] = 0.11
    payload["community"]["energy_in_total"] = _kwh_for_interval(40.0)
    payload["community"]["energy_out_total"] = 0.0

    actions = _run(sao_mamede_with_battery_community_client, payload)
    assert actions[ACTION_BATTERY] >= -0.1


def test_low_soc_can_recharge_with_persistent_community_deficit_when_price_favors_charge(
    sao_mamede_with_battery_community_client,
):
    payload = _base_payload()
    payload["observations"]["batteries"]["B01"]["SoC"] = 0.15
    payload["community"]["energy_in_total"] = _kwh_for_interval(35.0)
    payload["community"]["energy_out_total"] = 0.0
    payload["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.05] + [0.30] * 95

    actions = _run(sao_mamede_with_battery_community_client, payload)
    assert actions[ACTION_BATTERY] > 0.1


def test_forecast_deficit_tempers_virtual_battery_discharge_under_community_deficit(
    sao_mamede_with_battery_community_client,
):
    base = _base_payload()
    base["observations"]["batteries"]["B01"]["SoC"] = 0.55
    base["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.20] * 96
    base["community"]["energy_in_total"] = _kwh_for_interval(40.0)
    base["community"]["energy_out_total"] = 0.0

    forecasted = copy.deepcopy(base)
    _set_site_forecasts(
        forecasted,
        consumption_values=[8.0] * 8,
        production_values=[0.0] * 8,
    )

    base_actions = _run(sao_mamede_with_battery_community_client, base)
    forecast_actions = _run(sao_mamede_with_battery_community_client, forecasted)

    assert forecast_actions[ACTION_BATTERY] > base_actions[ACTION_BATTERY] + 0.1


def test_community_signal_still_biases_virtual_battery_when_forecast_is_present(
    sao_mamede_with_battery_community_client,
):
    neutral = _base_payload()
    neutral["observations"]["batteries"]["B01"]["SoC"] = 0.55
    neutral["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.20] * 96
    _set_site_forecasts(
        neutral,
        consumption_values=[8.0] * 8,
        production_values=[0.0] * 8,
    )

    deficit = copy.deepcopy(neutral)
    deficit["community"]["energy_in_total"] = _kwh_for_interval(40.0)
    deficit["community"]["energy_out_total"] = 0.0

    neutral_actions = _run(sao_mamede_with_battery_community_client, neutral)
    deficit_actions = _run(sao_mamede_with_battery_community_client, deficit)

    assert deficit_actions[ACTION_BATTERY] <= neutral_actions[ACTION_BATTERY]


def test_deadband_reduces_near_zero_dispatch_chattering(
    sao_mamede_with_battery_community_client,
):
    base = _base_payload()
    base["observations"]["batteries"]["B01"]["SoC"] = 0.85
    base["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.10] * 96

    deficit_small = copy.deepcopy(base)
    deficit_small["community"]["energy_in_total"] = _kwh_for_interval(0.4)
    deficit_small["community"]["energy_out_total"] = 0.0

    surplus_small = copy.deepcopy(base)
    surplus_small["community"]["energy_in_total"] = 0.0
    surplus_small["community"]["energy_out_total"] = _kwh_for_interval(0.4)

    neutral = copy.deepcopy(base)

    a_deficit = _run(sao_mamede_with_battery_community_client, deficit_small)[ACTION_BATTERY]
    a_surplus = _run(sao_mamede_with_battery_community_client, surplus_small)[ACTION_BATTERY]
    a_neutral = _run(sao_mamede_with_battery_community_client, neutral)[ACTION_BATTERY]

    assert abs(a_deficit) <= 0.1
    assert abs(a_surplus) <= 0.1
    assert abs(a_neutral) <= 0.1


def test_sign_flip_requires_extra_magnitude_threshold(
    sao_mamede_with_battery_community_client,
):
    first_payload = _base_payload()
    first_payload["observations"]["batteries"]["B01"]["SoC"] = 0.84
    first_payload["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.10] * 96
    first_payload["observations"]["solar_generation"] = 8.0
    first_payload["community"]["energy_in_total"] = 0.0
    first_payload["community"]["energy_out_total"] = 0.0

    second_payload = _base_payload()
    second_payload["observations"]["batteries"]["B01"]["SoC"] = 0.84
    second_payload["observations"]["energy_tariffs"]["OMIE"]["energy_price"]["values"] = [0.10] * 96
    second_payload["observations"]["solar_generation"] = 0.0
    second_payload["community"]["energy_in_total"] = _kwh_for_interval(10.4)
    second_payload["community"]["energy_out_total"] = 0.0

    first = _run(sao_mamede_with_battery_community_client, first_payload)[ACTION_BATTERY]
    second = _run(sao_mamede_with_battery_community_client, second_payload)[ACTION_BATTERY]

    assert first >= 5.0
    assert abs(second) <= 0.1
