from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List

from app.logging import get_logger


PRICE_POINTS_HOURS = (0.0, 1.0, 2.0, 6.0, 12.0, 24.0)
DEFAULT_EV_FLEX_FIELDS = {
    "soc": "electric_vehicles.{ev_id}.SoC",
    "target_soc": "electric_vehicles.{ev_id}.flexibility.estimated_soc_at_departure",
    "departure_time": "electric_vehicles.{ev_id}.flexibility.estimated_time_at_departure",
}


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric):
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
        if math.isnan(numeric):
            return None
        return numeric
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    numeric = _maybe_float(value)
    return default if numeric is None else numeric


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _parse_datetime(raw: Any) -> datetime | None:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or text.lower() in {"none", "nan"}:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _now_from_payload(payload: Dict[str, Any]) -> datetime:
    for key in ("timestamp", "timestamp.$date"):
        parsed = _parse_datetime(payload.get(key))
        if parsed:
            return parsed
    return datetime.now(timezone.utc)


def _round_one_decimal_towards_zero(value: float) -> float:
    scaled = value * 10.0
    if scaled >= 0:
        return math.floor(scaled) / 10.0
    return math.ceil(scaled) / 10.0


@dataclass
class Rh1HouseConfig:
    grid_import_limit_kw: float
    control_interval_minutes: float = 15.0
    export_price_factor: float = 0.8
    ev_min_connected_kw: float = 0.0
    fallback_safe_power_fraction: float = 0.30
    baseline_price_threshold_eur_kwh: float = 0.20
    battery_capacity_kwh: float = 4.0
    battery_nominal_power_kw: float = 3.32
    battery_efficiency: float = 0.95
    battery_soc_min: float = 0.20
    battery_soc_max: float = 1.0
    battery_degradation_penalty_eur_per_kwh: float = 0.01
    ev_default_capacity_kwh: float = 60.0
    cooling_nominal_power_kw: float = 4.109619617462158
    dhw_nominal_power_kw: float = 4.861171245574951
    chargers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    vehicle_capacities: Dict[str, float] = field(default_factory=dict)
    ev_flexibility_fields: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EV_FLEX_FIELDS))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Rh1HouseConfig":
        data = dict(payload)
        grid_import_limit_kw = _maybe_float(data.pop("grid_import_limit_kw", None))
        if grid_import_limit_kw is None or grid_import_limit_kw <= 0.0:
            raise ValueError("rh1_house_rbc_v1 requires a positive 'grid_import_limit_kw' in config")

        vehicle_capacities = {
            str(k): _safe_float(v, 0.0)
            for k, v in (data.pop("vehicle_capacities", {}) or {}).items()
            if _safe_float(v, 0.0) > 0.0
        }
        chargers_raw = dict(data.pop("chargers", {}) or {})
        chargers: Dict[str, Dict[str, Any]] = {}
        for cid, meta in chargers_raw.items():
            if not isinstance(meta, dict):
                continue
            chargers[str(cid)] = dict(meta)
        if not chargers:
            chargers = {"EVC01": {"min_kw": 0.0, "max_kw": 22.0}}

        flex_fields = dict(data.pop("ev_flexibility_fields", {}) or DEFAULT_EV_FLEX_FIELDS)

        return cls(
            grid_import_limit_kw=grid_import_limit_kw,
            control_interval_minutes=max(_safe_float(data.pop("control_interval_minutes", 15.0), 15.0), 1.0),
            export_price_factor=_clamp(_safe_float(data.pop("export_price_factor", 0.8), 0.8), 0.0, 1.0),
            ev_min_connected_kw=max(_safe_float(data.pop("ev_min_connected_kw", 0.0), 0.0), 0.0),
            fallback_safe_power_fraction=_clamp(
                _safe_float(data.pop("fallback_safe_power_fraction", 0.30), 0.30),
                0.0,
                1.0,
            ),
            baseline_price_threshold_eur_kwh=max(
                _safe_float(data.pop("baseline_price_threshold_eur_kwh", 0.20), 0.20),
                0.0,
            ),
            battery_capacity_kwh=max(_safe_float(data.pop("battery_capacity_kwh", 4.0), 4.0), 0.1),
            battery_nominal_power_kw=max(
                _safe_float(data.pop("battery_nominal_power_kw", 3.32), 3.32),
                0.0,
            ),
            battery_efficiency=_clamp(_safe_float(data.pop("battery_efficiency", 0.95), 0.95), 0.1, 1.0),
            battery_soc_min=_clamp(_safe_float(data.pop("battery_soc_min", 0.20), 0.20), 0.0, 1.0),
            battery_soc_max=_clamp(_safe_float(data.pop("battery_soc_max", 1.0), 1.0), 0.0, 1.0),
            battery_degradation_penalty_eur_per_kwh=max(
                _safe_float(data.pop("battery_degradation_penalty_eur_per_kwh", 0.01), 0.01),
                0.0,
            ),
            ev_default_capacity_kwh=max(_safe_float(data.pop("ev_default_capacity_kwh", 60.0), 60.0), 5.0),
            cooling_nominal_power_kw=max(
                _safe_float(data.pop("cooling_nominal_power_kw", 4.109619617462158), 4.109619617462158),
                0.0,
            ),
            dhw_nominal_power_kw=max(
                _safe_float(data.pop("dhw_nominal_power_kw", 4.861171245574951), 4.861171245574951),
                0.0,
            ),
            chargers=chargers,
            vehicle_capacities=vehicle_capacities,
            ev_flexibility_fields=flex_fields,
        )

    def resolve_ev_field(self, key: str, ev_id: str) -> str:
        template = self.ev_flexibility_fields.get(key)
        if not template:
            return ""
        return template.replace("{ev_id}", ev_id)


@dataclass
class BatteryPowerBounds:
    min_kw: float
    max_kw: float


@dataclass
class EvAggregate:
    connected: bool
    hard_min_kw: float
    max_kw: float


class Rh1HouseRuntime:
    def __init__(self, config: Rh1HouseConfig):
        self.config = config

    def allocate(self, payload: Dict[str, Any]) -> Dict[str, float]:
        cfg = self.config
        log = get_logger()
        warnings: List[str] = []
        local_constraint_flags: Dict[str, bool] = {}

        dt_hours = max(cfg.control_interval_minutes / 60.0, 1.0 / 60.0)
        now = _now_from_payload(payload)

        prices = self._extract_price_points(payload, warnings)
        price_now = prices[0]
        future_avg_price = self._average_interpolated_price(prices)

        non_shiftable_load = max(0.0, _safe_float(payload.get("non_shiftable_load"), 0.0))
        solar_generation = max(0.0, _safe_float(payload.get("solar_generation"), 0.0))

        grid_import_limit_kw = _safe_float(
            payload.get("grid.import_limit_kw"),
            cfg.grid_import_limit_kw,
        )
        if grid_import_limit_kw <= 0.0:
            warnings.append("invalid_or_missing_grid_import_limit_kw")
            grid_import_limit_kw = cfg.grid_import_limit_kw

        grid_export_limit_kw = _safe_float(
            payload.get("grid.export_limit_kw"),
            grid_import_limit_kw,
        )
        grid_export_limit_kw = max(grid_export_limit_kw, 0.0)

        battery_soc = _maybe_float(payload.get("electrical_storage.soc"))
        if battery_soc is None:
            warnings.append("missing_battery_soc_defaulted")
            battery_soc = 0.5
        battery_soc = _clamp(battery_soc, 0.0, 1.0)

        cooling_kw, cooling_forced = self._cooling_action(payload, warnings)
        dhw_kw, dhw_forced = self._dhw_action(payload, warnings)

        ev = self._ev_action_bounds(payload, now, dt_hours)
        ev_kw = ev.hard_min_kw
        if ev.connected and price_now <= min(cfg.baseline_price_threshold_eur_kwh, future_avg_price):
            ev_kw = ev.max_kw
        ev_kw = _clamp(ev_kw, 0.0, ev.max_kw)

        base_without_battery = non_shiftable_load + cooling_kw + dhw_kw + ev_kw - solar_generation

        battery_bounds = self._battery_power_bounds(battery_soc, dt_hours)
        battery_kw = self._battery_dispatch(
            base_without_battery=base_without_battery,
            price_now=price_now,
            future_avg_price=future_avg_price,
            bounds=battery_bounds,
        )

        battery_kw = _clamp(battery_kw, battery_bounds.min_kw, battery_bounds.max_kw)

        battery_kw = self._fit_battery_to_grid(
            base_without_battery,
            battery_kw,
            battery_bounds,
            grid_import_limit_kw,
            grid_export_limit_kw,
        )

        net_grid_kw = base_without_battery + battery_kw

        if net_grid_kw > grid_import_limit_kw + 1e-6:
            shortfall = net_grid_kw - grid_import_limit_kw
            reduced = min(shortfall, ev_kw)
            ev_kw -= reduced
            shortfall -= reduced
            if ev_kw + 1e-6 < ev.hard_min_kw:
                local_constraint_flags["ev_hard_min_unmet"] = True

            if shortfall > 1e-6:
                reduced = min(shortfall, cooling_kw)
                cooling_kw -= reduced
                shortfall -= reduced
                if cooling_forced and cooling_kw < cfg.cooling_nominal_power_kw - 1e-6:
                    local_constraint_flags["cooling_band_unmet"] = True

            if shortfall > 1e-6:
                reduced = min(shortfall, dhw_kw)
                dhw_kw -= reduced
                shortfall -= reduced
                if dhw_forced and dhw_kw < cfg.dhw_nominal_power_kw - 1e-6:
                    local_constraint_flags["dhw_band_unmet"] = True

            base_without_battery = non_shiftable_load + cooling_kw + dhw_kw + ev_kw - solar_generation
            battery_kw = self._fit_battery_to_grid(
                base_without_battery,
                battery_kw,
                battery_bounds,
                grid_import_limit_kw,
                grid_export_limit_kw,
            )
            net_grid_kw = base_without_battery + battery_kw
            if net_grid_kw > grid_import_limit_kw + 1e-6:
                local_constraint_flags["grid_import_limit_unmet"] = True

        if net_grid_kw < -grid_export_limit_kw - 1e-6:
            excess_export = -grid_export_limit_kw - net_grid_kw

            ev_room = max(ev.max_kw - ev_kw, 0.0)
            increased = min(excess_export, ev_room)
            ev_kw += increased
            excess_export -= increased

            if excess_export > 1e-6:
                dhw_room = max(cfg.dhw_nominal_power_kw - dhw_kw, 0.0)
                increased = min(excess_export, dhw_room)
                dhw_kw += increased
                excess_export -= increased

            if excess_export > 1e-6:
                cooling_room = max(cfg.cooling_nominal_power_kw - cooling_kw, 0.0)
                increased = min(excess_export, cooling_room)
                cooling_kw += increased
                excess_export -= increased

            base_without_battery = non_shiftable_load + cooling_kw + dhw_kw + ev_kw - solar_generation
            battery_kw = self._fit_battery_to_grid(
                base_without_battery,
                battery_kw,
                battery_bounds,
                grid_import_limit_kw,
                grid_export_limit_kw,
            )
            net_grid_kw = base_without_battery + battery_kw
            if net_grid_kw < -grid_export_limit_kw - 1e-6:
                local_constraint_flags["grid_export_limit_unmet"] = True

        ev_kw = _clamp(ev_kw, 0.0, ev.max_kw)
        cooling_kw = _clamp(cooling_kw, 0.0, cfg.cooling_nominal_power_kw)
        dhw_kw = _clamp(dhw_kw, 0.0, cfg.dhw_nominal_power_kw)
        battery_kw = _clamp(battery_kw, battery_bounds.min_kw, battery_bounds.max_kw)

        actions = {
            "ev_charge_kw": ev_kw,
            "battery_kw": battery_kw,
            "cooling_kw": cooling_kw,
            "dhw_heater_kw": dhw_kw,
        }

        quantized = {
            key: float(_round_one_decimal_towards_zero(value))
            for key, value in actions.items()
        }

        quantized["ev_charge_kw"] = float(_clamp(quantized["ev_charge_kw"], 0.0, ev.max_kw))
        quantized["cooling_kw"] = float(
            _clamp(quantized["cooling_kw"], 0.0, cfg.cooling_nominal_power_kw)
        )
        quantized["dhw_heater_kw"] = float(
            _clamp(quantized["dhw_heater_kw"], 0.0, cfg.dhw_nominal_power_kw)
        )
        quantized["battery_kw"] = float(
            _clamp(quantized["battery_kw"], battery_bounds.min_kw, battery_bounds.max_kw)
        )

        net_grid_kw = (
            non_shiftable_load
            + quantized["cooling_kw"]
            + quantized["dhw_heater_kw"]
            + quantized["ev_charge_kw"]
            + quantized["battery_kw"]
            - solar_generation
        )

        log.info(
            "rbc.actions",
            strategy="rh1_house_rbc_v1",
            actions=quantized,
            connected={"ev": ev.connected},
            phase_totals={"grid": net_grid_kw},
            board_total=max(net_grid_kw, 0.0),
        )
        if warnings:
            log.warning("rh1.payload_fallbacks", issues=warnings)
        if local_constraint_flags:
            log.warning("rh1.constraint_flags", flags=local_constraint_flags)

        return quantized

    def _extract_price_points(self, payload: Dict[str, Any], warnings: List[str]) -> Dict[float, float]:
        current_candidates = [
            payload.get("electricity_pricing.current"),
            payload.get("electricity_pricing"),
            payload.get("energy_price"),
        ]
        current = None
        for candidate in current_candidates:
            current = _maybe_float(candidate)
            if current is not None:
                break
        if current is None:
            warnings.append("missing_price_current_defaulted")
            current = 0.0

        points = {0.0: max(current, 0.0)}
        key_by_hour = {
            1.0: "electricity_pricing.h1",
            2.0: "electricity_pricing.h2",
            6.0: "electricity_pricing.h6",
            12.0: "electricity_pricing.h12",
            24.0: "electricity_pricing.h24",
        }
        for hour, key in key_by_hour.items():
            value = _maybe_float(payload.get(key))
            if value is None:
                warnings.append(f"missing_price_{int(hour)}h_defaulted")
                value = current
            points[hour] = max(value, 0.0)
        return points

    def _interpolate_price(self, hour: float, points: Dict[float, float]) -> float:
        if hour <= 0.0:
            return points[0.0]
        if hour >= 24.0:
            return points[24.0]

        ordered = sorted(points.items())
        for idx in range(1, len(ordered)):
            left_h, left_p = ordered[idx - 1]
            right_h, right_p = ordered[idx]
            if left_h <= hour <= right_h:
                if right_h - left_h <= 1e-9:
                    return right_p
                alpha = (hour - left_h) / (right_h - left_h)
                return left_p + alpha * (right_p - left_p)
        return points[24.0]

    def _average_interpolated_price(self, points: Dict[float, float]) -> float:
        samples = [self._interpolate_price(float(h), points) for h in range(1, 25)]
        if not samples:
            return points[0.0]
        return float(sum(samples) / len(samples))

    def _cooling_action(self, payload: Dict[str, Any], warnings: List[str]) -> tuple[float, bool]:
        cfg = self.config
        temp = _maybe_float(payload.get("cooling.temperature.current_c"))
        min_c = _maybe_float(payload.get("cooling.temperature.min_c"))
        max_c = _maybe_float(payload.get("cooling.temperature.max_c"))
        safe_kw = cfg.cooling_nominal_power_kw * cfg.fallback_safe_power_fraction

        if temp is None or min_c is None or max_c is None:
            warnings.append("missing_cooling_temperature_data_safe_mode")
            return safe_kw, False

        if temp > max_c + 1e-6:
            return cfg.cooling_nominal_power_kw, True
        return 0.0, False

    def _dhw_action(self, payload: Dict[str, Any], warnings: List[str]) -> tuple[float, bool]:
        cfg = self.config
        temp = _maybe_float(payload.get("dhw.temperature.current_c"))
        min_c = _maybe_float(payload.get("dhw.temperature.min_c"))
        max_c = _maybe_float(payload.get("dhw.temperature.max_c"))
        safe_kw = cfg.dhw_nominal_power_kw * cfg.fallback_safe_power_fraction

        if temp is None or min_c is None or max_c is None:
            warnings.append("missing_dhw_temperature_data_safe_mode")
            return safe_kw, False

        if temp < min_c - 1e-6:
            return cfg.dhw_nominal_power_kw, True
        return 0.0, False

    def _ev_action_bounds(self, payload: Dict[str, Any], now: datetime, dt_hours: float) -> EvAggregate:
        cfg = self.config
        control_minutes = max(cfg.control_interval_minutes, 1.0)
        hard_min_total = 0.0
        max_total = 0.0
        connected_any = False

        for charger_id, meta in cfg.chargers.items():
            ev_raw = payload.get(f"charging_sessions.{charger_id}.electric_vehicle")
            ev_id = str(ev_raw).strip() if ev_raw is not None else ""
            if not ev_id:
                continue
            connected_any = True

            charger_min = max(_safe_float(meta.get("min_kw"), cfg.ev_min_connected_kw), cfg.ev_min_connected_kw)
            charger_max = max(_safe_float(meta.get("max_kw"), 22.0), charger_min)
            max_total += charger_max

            required_kw = self._required_ev_power(
                payload=payload,
                ev_id=ev_id,
                charger_min_kw=charger_min,
                charger_max_kw=charger_max,
                now=now,
                control_minutes=control_minutes,
                dt_hours=dt_hours,
            )
            hard_min_total += max(required_kw, charger_min)

        if not connected_any:
            return EvAggregate(connected=False, hard_min_kw=0.0, max_kw=0.0)

        hard_min_total = _clamp(hard_min_total, 0.0, max_total)
        return EvAggregate(connected=True, hard_min_kw=hard_min_total, max_kw=max_total)

    def _required_ev_power(
        self,
        payload: Dict[str, Any],
        ev_id: str,
        charger_min_kw: float,
        charger_max_kw: float,
        now: datetime,
        control_minutes: float,
        dt_hours: float,
    ) -> float:
        cfg = self.config
        soc_key = cfg.resolve_ev_field("soc", ev_id)
        target_key = cfg.resolve_ev_field("target_soc", ev_id)
        departure_key = cfg.resolve_ev_field("departure_time", ev_id)

        soc = _maybe_float(payload.get(soc_key))
        target_soc = _maybe_float(payload.get(target_key))
        departure = _parse_datetime(payload.get(departure_key))

        if soc is None or target_soc is None or departure is None or target_soc <= 0.0:
            return cfg.ev_min_connected_kw

        soc = _clamp(soc, 0.0, 1.0)
        target_soc = _clamp(target_soc, soc, 1.0)
        if target_soc <= soc + 1e-6:
            return cfg.ev_min_connected_kw

        minutes_remaining = max((departure - now).total_seconds() / 60.0, control_minutes)
        capacity_kwh = cfg.vehicle_capacities.get(ev_id, cfg.ev_default_capacity_kwh)
        capacity_kwh = max(capacity_kwh, 1.0)
        gap_kwh = max(target_soc - soc, 0.0) * capacity_kwh

        if gap_kwh <= 1e-9:
            return cfg.ev_min_connected_kw

        if minutes_remaining <= control_minutes + 1e-9:
            required_kw = charger_max_kw
        else:
            required_kw = gap_kwh / max(minutes_remaining / 60.0, dt_hours)
        required_kw = _clamp(required_kw, charger_min_kw, charger_max_kw)
        return required_kw

    def _battery_power_bounds(self, soc: float, dt_hours: float) -> BatteryPowerBounds:
        cfg = self.config
        cap_kwh = max(cfg.battery_capacity_kwh, 1e-6)
        eff = _clamp(cfg.battery_efficiency, 1e-6, 1.0)

        charge_room_kwh = max(cfg.battery_soc_max - soc, 0.0) * cap_kwh
        discharge_room_kwh = max(soc - cfg.battery_soc_min, 0.0) * cap_kwh

        max_charge_kw_soc = charge_room_kwh / max(dt_hours * eff, 1e-6)
        max_discharge_kw_soc = discharge_room_kwh * eff / max(dt_hours, 1e-6)

        max_charge_kw = min(cfg.battery_nominal_power_kw, max_charge_kw_soc)
        max_discharge_kw = min(cfg.battery_nominal_power_kw, max_discharge_kw_soc)
        return BatteryPowerBounds(min_kw=-max_discharge_kw, max_kw=max_charge_kw)

    def _battery_dispatch(
        self,
        base_without_battery: float,
        price_now: float,
        future_avg_price: float,
        bounds: BatteryPowerBounds,
    ) -> float:
        cfg = self.config
        spread = future_avg_price - price_now
        penalty = cfg.battery_degradation_penalty_eur_per_kwh

        if spread > penalty + 1e-9:
            return bounds.max_kw

        discharge_value_now = price_now
        if base_without_battery <= 0.0:
            discharge_value_now = price_now * cfg.export_price_factor
        if discharge_value_now - future_avg_price > penalty + 1e-9:
            return bounds.min_kw

        return 0.0

    def _fit_battery_to_grid(
        self,
        base_without_battery: float,
        battery_kw: float,
        bounds: BatteryPowerBounds,
        grid_import_limit_kw: float,
        grid_export_limit_kw: float,
    ) -> float:
        adjusted = _clamp(battery_kw, bounds.min_kw, bounds.max_kw)
        net_grid = base_without_battery + adjusted

        if net_grid > grid_import_limit_kw + 1e-9:
            delta = net_grid - grid_import_limit_kw
            adjusted -= delta
            adjusted = _clamp(adjusted, bounds.min_kw, bounds.max_kw)

        net_grid = base_without_battery + adjusted
        if net_grid < -grid_export_limit_kw - 1e-9:
            delta = -grid_export_limit_kw - net_grid
            adjusted += delta
            adjusted = _clamp(adjusted, bounds.min_kw, bounds.max_kw)

        return adjusted
