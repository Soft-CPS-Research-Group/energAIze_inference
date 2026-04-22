from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric != numeric:  # NaN check
            return None
        return numeric
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none"}:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
        if numeric != numeric:
            return None
        return numeric
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _normalize_unit(unit_hint: Any) -> str:
    return str(unit_hint or "").strip().lower().replace(" ", "")


def _normalize_energy_sample_kwh(
    value: float,
    *,
    unit_hint: Any,
    frequency_seconds: float,
) -> float | None:
    unit = _normalize_unit(unit_hint)
    interval_hours = frequency_seconds / 3600.0
    if interval_hours <= 0.0:
        return None

    if "kwh" in unit:
        return value
    if "mwh" in unit:
        return value * 1000.0
    if "wh" in unit:
        return value / 1000.0
    if unit.endswith("kw") or unit == "kw":
        return value * interval_hours
    if unit.endswith("mw") or unit == "mw":
        return value * 1000.0 * interval_hours
    if unit.endswith("w") or unit == "w":
        return (value / 1000.0) * interval_hours
    return None


@dataclass(frozen=True)
class ForecastSeries:
    prefix: str
    measurement_unit: str
    frequency_seconds: float
    horizon_seconds: float
    raw_values: List[float]
    step_energy_kwh: List[float]
    step_power_kw: List[float]

    def window_energy_kwh(self, window_seconds: float) -> float:
        if window_seconds <= 0.0 or not self.step_energy_kwh:
            return 0.0

        remaining = window_seconds
        total = 0.0
        for value in self.step_energy_kwh:
            if remaining <= 1e-9:
                break
            covered_seconds = min(self.frequency_seconds, remaining)
            total += value * (covered_seconds / self.frequency_seconds)
            remaining -= covered_seconds
        return total


@dataclass(frozen=True)
class SiteForecastSnapshot:
    consumption: ForecastSeries | None
    production: ForecastSeries | None
    net_step_energy_kwh: List[float] | None
    net_step_power_kw: List[float] | None
    net_frequency_seconds: float | None

    def window_net_energy_kwh(self, window_seconds: float) -> float | None:
        consumption_kwh = (
            self.consumption.window_energy_kwh(window_seconds) if self.consumption else None
        )
        production_kwh = (
            self.production.window_energy_kwh(window_seconds) if self.production else None
        )
        if consumption_kwh is None and production_kwh is None:
            return None
        return float((consumption_kwh or 0.0) - (production_kwh or 0.0))


@dataclass(frozen=True)
class ForecastReadResult:
    snapshot: SiteForecastSnapshot | None
    issues: Sequence[str]


@dataclass(frozen=True)
class ForecastSocGuidance:
    window_net_energy_kwh: float
    window_avg_net_kw: float
    imbalance_ratio: float
    reserve_floor_soc: float
    target_soc: float
    target_kw: float


def _prefix_present(payload: Dict[str, Any], prefix: str) -> bool:
    if not prefix:
        return False
    value_prefix = f"{prefix}."
    for key in payload:
        if not isinstance(key, str):
            continue
        if key == prefix or key.startswith(value_prefix):
            return True
    return False


def _extract_forecast_series(payload: Dict[str, Any], prefix: str) -> tuple[ForecastSeries | None, List[str]]:
    issues: List[str] = []
    if not prefix or not _prefix_present(payload, prefix):
        return None, issues

    values: List[tuple[int, float]] = []
    values_prefix = f"{prefix}.values["
    for key, raw in payload.items():
        if not isinstance(key, str):
            continue
        if not key.startswith(values_prefix) or not key.endswith("]"):
            continue
        idx_text = key[len(values_prefix) : -1]
        if not idx_text.isdigit():
            continue
        numeric = _maybe_float(raw)
        if numeric is None:
            issues.append(f"{prefix}:invalid_value_at_index_{idx_text}")
            return None, issues
        values.append((int(idx_text), numeric))

    if not values:
        issues.append(f"{prefix}:missing_values")
        return None, issues

    frequency_seconds = _maybe_float(payload.get(f"{prefix}.frequency_seconds"))
    if frequency_seconds is None or frequency_seconds <= 0.0:
        issues.append(f"{prefix}:invalid_frequency_seconds")
        return None, issues

    measurement_unit = payload.get(f"{prefix}.measurement_unit")
    if not _normalize_unit(measurement_unit):
        issues.append(f"{prefix}:missing_measurement_unit")
        return None, issues

    horizon_seconds = _maybe_float(payload.get(f"{prefix}.horizon_seconds"))
    if horizon_seconds is None or horizon_seconds <= 0.0:
        horizon_seconds = len(values) * frequency_seconds
        issues.append(f"{prefix}:missing_or_invalid_horizon_seconds_inferred")

    ordered_values = [value for _, value in sorted(values, key=lambda pair: pair[0])]
    step_energy_kwh: List[float] = []
    step_power_kw: List[float] = []
    for value in ordered_values:
        energy_kwh = _normalize_energy_sample_kwh(
            value,
            unit_hint=measurement_unit,
            frequency_seconds=frequency_seconds,
        )
        if energy_kwh is None:
            issues.append(f"{prefix}:unsupported_measurement_unit")
            return None, issues
        step_energy_kwh.append(float(energy_kwh))
        step_power_kw.append(float(energy_kwh / (frequency_seconds / 3600.0)))

    return (
        ForecastSeries(
            prefix=prefix,
            measurement_unit=str(measurement_unit),
            frequency_seconds=float(frequency_seconds),
            horizon_seconds=float(horizon_seconds),
            raw_values=[float(value) for value in ordered_values],
            step_energy_kwh=step_energy_kwh,
            step_power_kw=step_power_kw,
        ),
        issues,
    )


def read_site_forecasts(
    payload: Dict[str, Any],
    *,
    consumption_prefix: str,
    production_prefix: str,
) -> ForecastReadResult:
    issues: List[str] = []

    consumption, consumption_issues = _extract_forecast_series(payload, consumption_prefix)
    production, production_issues = _extract_forecast_series(payload, production_prefix)
    issues.extend(consumption_issues)
    issues.extend(production_issues)

    if consumption is None and production is None:
        return ForecastReadResult(snapshot=None, issues=tuple(issues))

    net_step_energy_kwh: List[float] | None = None
    net_step_power_kw: List[float] | None = None
    net_frequency_seconds: float | None = None

    if consumption and production and abs(consumption.frequency_seconds - production.frequency_seconds) <= 1e-9:
        net_frequency_seconds = float(consumption.frequency_seconds)
        count = max(len(consumption.step_energy_kwh), len(production.step_energy_kwh))
        net_step_energy_kwh = []
        net_step_power_kw = []
        interval_hours = net_frequency_seconds / 3600.0
        for idx in range(count):
            consumption_kwh = consumption.step_energy_kwh[idx] if idx < len(consumption.step_energy_kwh) else 0.0
            production_kwh = production.step_energy_kwh[idx] if idx < len(production.step_energy_kwh) else 0.0
            net_kwh = consumption_kwh - production_kwh
            net_step_energy_kwh.append(float(net_kwh))
            net_step_power_kw.append(float(net_kwh / interval_hours))

    return ForecastReadResult(
        snapshot=SiteForecastSnapshot(
            consumption=consumption,
            production=production,
            net_step_energy_kwh=net_step_energy_kwh,
            net_step_power_kw=net_step_power_kw,
            net_frequency_seconds=net_frequency_seconds,
        ),
        issues=tuple(issues),
    )


def build_forecast_soc_guidance(
    snapshot: SiteForecastSnapshot | None,
    *,
    window_hours: float,
    capacity_kwh: float,
    current_soc: float,
    base_reserve_soc: float,
    base_target_soc: float,
    soc_min: float,
    soc_max: float,
    charge_limit_kw: float,
    discharge_limit_kw: float,
    soc_bias_gain: float,
    soc_bias_max: float,
    dispatch_weight: float,
) -> ForecastSocGuidance | None:
    if snapshot is None or window_hours <= 0.0 or capacity_kwh <= 1e-9:
        return None

    window_seconds = window_hours * 3600.0
    window_net_energy_kwh = snapshot.window_net_energy_kwh(window_seconds)
    if window_net_energy_kwh is None:
        return None

    imbalance_ratio = _clamp(window_net_energy_kwh / capacity_kwh, -1.0, 1.0)
    bias_magnitude = min(abs(imbalance_ratio) * max(soc_bias_gain, 0.0), max(soc_bias_max, 0.0))

    reserve_floor_soc = _clamp(base_reserve_soc, soc_min, soc_max)
    target_soc = _clamp(base_target_soc, reserve_floor_soc, soc_max)
    if imbalance_ratio > 1e-9:
        reserve_floor_soc = _clamp(reserve_floor_soc + (0.75 * bias_magnitude), soc_min, soc_max)
        target_soc = _clamp(target_soc + bias_magnitude, reserve_floor_soc, soc_max)
    elif imbalance_ratio < -1e-9:
        target_soc = _clamp(target_soc - bias_magnitude, reserve_floor_soc, soc_max)

    window_avg_net_kw = window_net_energy_kwh / window_hours
    target_kw = 0.0
    abs_weighted_signal_kw = abs(window_avg_net_kw) * max(dispatch_weight, 0.0)
    if imbalance_ratio > 1e-9 and charge_limit_kw > 1e-9 and target_soc > current_soc + 1e-9:
        charge_room_kwh = max(target_soc - current_soc, 0.0) * capacity_kwh
        charge_room_kw = charge_room_kwh / window_hours
        target_kw = min(charge_limit_kw, charge_room_kw, abs_weighted_signal_kw)
    elif imbalance_ratio < -1e-9 and discharge_limit_kw > 1e-9 and current_soc > target_soc + 1e-9:
        discharge_room_kwh = max(current_soc - target_soc, 0.0) * capacity_kwh
        discharge_room_kw = discharge_room_kwh / window_hours
        target_kw = -min(discharge_limit_kw, discharge_room_kw, abs_weighted_signal_kw)

    return ForecastSocGuidance(
        window_net_energy_kwh=float(window_net_energy_kwh),
        window_avg_net_kw=float(window_avg_net_kw),
        imbalance_ratio=float(imbalance_ratio),
        reserve_floor_soc=float(reserve_floor_soc),
        target_soc=float(target_soc),
        target_kw=float(target_kw),
    )
