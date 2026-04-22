from __future__ import annotations

import pytest

from app.services.rbc.forecasts import read_site_forecasts


CONSUMPTION_PREFIX = "forecasts.ConsumptionForecastService.consumption_total"
PRODUCTION_PREFIX = "forecasts.ProductionForecastService.production_total"


def test_read_site_forecasts_parses_consumption_and_production_sample():
    payload = {
        f"{CONSUMPTION_PREFIX}.values[0]": 0.00000935,
        f"{CONSUMPTION_PREFIX}.values[1]": 0.00000913,
        f"{CONSUMPTION_PREFIX}.measurement_unit": "kWh",
        f"{CONSUMPTION_PREFIX}.frequency_seconds": 900,
        f"{CONSUMPTION_PREFIX}.horizon_seconds": 1800,
        f"{PRODUCTION_PREFIX}.values[0]": 0.00008903,
        f"{PRODUCTION_PREFIX}.values[1]": 0.00008870,
        f"{PRODUCTION_PREFIX}.measurement_unit": "kWh",
        f"{PRODUCTION_PREFIX}.frequency_seconds": 900,
        f"{PRODUCTION_PREFIX}.horizon_seconds": 1800,
    }

    result = read_site_forecasts(
        payload,
        consumption_prefix=CONSUMPTION_PREFIX,
        production_prefix=PRODUCTION_PREFIX,
    )

    assert result.snapshot is not None
    assert result.snapshot.consumption is not None
    assert result.snapshot.production is not None
    assert result.snapshot.consumption.raw_values == pytest.approx([0.00000935, 0.00000913], rel=1e-9)
    assert result.snapshot.production.raw_values == pytest.approx([0.00008903, 0.00008870], rel=1e-9)
    assert result.snapshot.net_step_energy_kwh == pytest.approx(
        [-0.00007968, -0.00007957],
        rel=1e-9,
    )
    assert result.snapshot.window_net_energy_kwh(1800.0) == pytest.approx(-0.00015925, rel=1e-9)


def test_read_site_forecasts_supports_consumption_only():
    payload = {
        f"{CONSUMPTION_PREFIX}.values[0]": 0.2,
        f"{CONSUMPTION_PREFIX}.values[1]": 0.1,
        f"{CONSUMPTION_PREFIX}.measurement_unit": "kWh",
        f"{CONSUMPTION_PREFIX}.frequency_seconds": 900,
        f"{CONSUMPTION_PREFIX}.horizon_seconds": 1800,
    }

    result = read_site_forecasts(
        payload,
        consumption_prefix=CONSUMPTION_PREFIX,
        production_prefix=PRODUCTION_PREFIX,
    )

    assert result.snapshot is not None
    assert result.snapshot.consumption is not None
    assert result.snapshot.production is None
    assert result.snapshot.window_net_energy_kwh(1800.0) == pytest.approx(0.3, rel=1e-9)


def test_read_site_forecasts_supports_production_only():
    payload = {
        f"{PRODUCTION_PREFIX}.values[0]": 0.15,
        f"{PRODUCTION_PREFIX}.values[1]": 0.05,
        f"{PRODUCTION_PREFIX}.measurement_unit": "kWh",
        f"{PRODUCTION_PREFIX}.frequency_seconds": 900,
        f"{PRODUCTION_PREFIX}.horizon_seconds": 1800,
    }

    result = read_site_forecasts(
        payload,
        consumption_prefix=CONSUMPTION_PREFIX,
        production_prefix=PRODUCTION_PREFIX,
    )

    assert result.snapshot is not None
    assert result.snapshot.consumption is None
    assert result.snapshot.production is not None
    assert result.snapshot.window_net_energy_kwh(1800.0) == pytest.approx(-0.2, rel=1e-9)


def test_read_site_forecasts_returns_none_for_missing_or_invalid_metadata():
    payload = {
        f"{CONSUMPTION_PREFIX}.values[0]": 0.2,
        f"{CONSUMPTION_PREFIX}.measurement_unit": "kWh",
        f"{CONSUMPTION_PREFIX}.frequency_seconds": -1,
        f"{CONSUMPTION_PREFIX}.horizon_seconds": 900,
        f"{PRODUCTION_PREFIX}.values[0]": 0.1,
        f"{PRODUCTION_PREFIX}.frequency_seconds": 900,
        f"{PRODUCTION_PREFIX}.horizon_seconds": 900,
    }

    result = read_site_forecasts(
        payload,
        consumption_prefix=CONSUMPTION_PREFIX,
        production_prefix=PRODUCTION_PREFIX,
    )

    assert result.snapshot is None
    assert any("invalid_frequency_seconds" in issue for issue in result.issues)
    assert any("missing_measurement_unit" in issue for issue in result.issues)
