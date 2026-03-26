from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, List

from app.logging import get_logger


PRICE_POINTS_HOURS = (0.0, 1.0, 2.0, 6.0, 12.0, 24.0)
VALID_PRICE_UNIT_MODES = {"auto", "eur_kwh", "eur_mwh"}
VALID_SOC_UNIT_MODES = {"auto", "fraction", "percent"}
DEFAULT_EV_FLEX_FIELDS = {
    "soc": "electric_vehicles.{ev_id}.SoC",
    "target_soc": "electric_vehicles.{ev_id}.flexibility.estimated_soc_at_departure",
    "departure_time": "electric_vehicles.{ev_id}.flexibility.estimated_time_at_departure",
}


def _extract_battery_id_from_soc_key(key: str) -> str | None:
    parts = [segment for segment in key.split(".") if segment]
    if len(parts) < 3:
        return None
    if parts[0].lower() != "batteries" or parts[-1].lower() != "soc":
        return None
    candidate = parts[1].strip()
    return candidate or None


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
    control_interval_minutes: float = 1.0
    export_price_factor: float = 0.8
    ev_min_connected_kw: float = 0.0
    baseline_price_threshold_eur_kwh: float = 0.20
    battery_capacity_kwh: float = 4.0
    battery_nominal_power_kw: float = 3.32
    battery_efficiency: float = 0.95
    battery_soc_min: float = 0.20
    battery_soc_max: float = 1.0
    battery_degradation_penalty_eur_per_kwh: float = 0.01
    ev_default_capacity_kwh: float = 60.0
    chargers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    vehicle_capacities: Dict[str, float] = field(default_factory=dict)
    ev_flexibility_fields: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EV_FLEX_FIELDS))
    price_unit_mode: str = "auto"
    price_auto_mwh_threshold: float = 3.0
    soc_unit_mode: str = "auto"
    battery_soc_keys: List[str] = field(default_factory=lambda: ["electrical_storage.soc"])
    ev_action_name: str = ""
    battery_action_name: str = ""

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
            if isinstance(meta, dict):
                chargers[str(cid)] = dict(meta)
        if not chargers:
            chargers = {"EVC01": {"min_kw": 0.0, "max_kw": 22.0}}

        flex_fields = dict(data.pop("ev_flexibility_fields", {}) or DEFAULT_EV_FLEX_FIELDS)

        price_unit_mode = str(data.pop("price_unit_mode", "auto")).strip().lower() or "auto"
        if price_unit_mode not in VALID_PRICE_UNIT_MODES:
            price_unit_mode = "auto"

        soc_unit_mode = str(data.pop("soc_unit_mode", "auto")).strip().lower() or "auto"
        if soc_unit_mode not in VALID_SOC_UNIT_MODES:
            soc_unit_mode = "auto"

        battery_soc_keys_raw = data.pop("battery_soc_keys", ["electrical_storage.soc"])
        battery_soc_keys: List[str] = []
        if isinstance(battery_soc_keys_raw, list):
            battery_soc_keys = [str(item) for item in battery_soc_keys_raw if str(item).strip()]
        elif battery_soc_keys_raw is not None:
            value = str(battery_soc_keys_raw).strip()
            if value:
                battery_soc_keys = [value]
        if not battery_soc_keys:
            battery_soc_keys = ["electrical_storage.soc"]

        ev_action_name = str(data.pop("ev_action_name", "")).strip()
        battery_action_name = str(data.pop("battery_action_name", "")).strip()

        return cls(
            grid_import_limit_kw=grid_import_limit_kw,
            control_interval_minutes=max(
                _safe_float(data.pop("control_interval_minutes", 1.0), 1.0),
                1.0 / 60.0,
            ),
            export_price_factor=_clamp(_safe_float(data.pop("export_price_factor", 0.8), 0.8), 0.0, 1.0),
            ev_min_connected_kw=max(_safe_float(data.pop("ev_min_connected_kw", 0.0), 0.0), 0.0),
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
            chargers=chargers,
            vehicle_capacities=vehicle_capacities,
            ev_flexibility_fields=flex_fields,
            price_unit_mode=price_unit_mode,
            price_auto_mwh_threshold=max(
                _safe_float(data.pop("price_auto_mwh_threshold", 3.0), 3.0),
                0.0,
            ),
            soc_unit_mode=soc_unit_mode,
            battery_soc_keys=battery_soc_keys,
            ev_action_name=ev_action_name,
            battery_action_name=battery_action_name,
        )

    def resolve_ev_field(self, key: str, ev_id: str) -> str:
        template = self.ev_flexibility_fields.get(key)
        if not template:
            return ""
        return template.replace("{ev_id}", ev_id)

    def resolve_ev_action_name(self) -> str:
        if self.ev_action_name:
            return self.ev_action_name
        for charger_id in self.chargers:
            candidate = str(charger_id).strip()
            if candidate:
                return candidate
        return "ev_charge_kw"

    def resolve_battery_action_name(self) -> str:
        if self.battery_action_name:
            return self.battery_action_name
        for key in self.battery_soc_keys:
            candidate = _extract_battery_id_from_soc_key(str(key))
            if candidate:
                return candidate
        return "battery_kw"


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

    def _meter_component_kw(self, payload: Dict[str, Any], component: str) -> float | None:
        """Read RH01 grid meter import/export with support for total, legacy and phase schemas."""
        base = "grid_meters.GR01"
        if component == "energy_in":
            keys = [
                f"{base}.energy_in_total",
                f"{base}.energy_in",
            ]
            phase_keys = (
                f"{base}.energy_in_l1",
                f"{base}.energy_in_l2",
                f"{base}.energy_in_l3",
            )
        else:
            keys = [
                f"{base}.energy_out_total",
                f"{base}.energy_out",
            ]
            phase_keys = (
                f"{base}.energy_out_l1",
                f"{base}.energy_out_l2",
                f"{base}.energy_out_l3",
            )

        for key in keys:
            value = _maybe_float(payload.get(key))
            if value is not None:
                return max(value, 0.0)

        phase_values: List[float] = []
        for key in phase_keys:
            value = _maybe_float(payload.get(key))
            if value is None:
                continue
            phase_values.append(max(value, 0.0))
        if phase_values:
            return sum(phase_values)

        return None

    def allocate(self, payload: Dict[str, Any]) -> Dict[str, float]:
        cfg = self.config
        log = get_logger()
        warnings: List[str] = []
        local_constraint_flags: Dict[str, bool] = {}

        dt_hours = max(cfg.control_interval_minutes / 60.0, 1.0 / 60.0)
        now = _now_from_payload(payload)

        prices = self._extract_price_points(payload, warnings)
        price_now = prices[0.0]
        future_avg_price = self._average_interpolated_price(prices)

        non_shiftable_load = _safe_float(payload.get("non_shiftable_load"), 0.0)
        solar_generation = self._extract_solar_generation(payload, warnings)

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

        battery_soc = self._extract_battery_soc(payload, warnings)

        ev = self._ev_action_bounds(payload, now, dt_hours)
        ev_kw = ev.hard_min_kw
        if ev.connected and price_now <= min(cfg.baseline_price_threshold_eur_kwh, future_avg_price):
            ev_kw = ev.max_kw
        ev_kw = _clamp(ev_kw, 0.0, ev.max_kw)

        base_without_battery = self._estimate_base_without_battery(
            payload=payload,
            ev_kw=ev_kw,
            non_shiftable_load=non_shiftable_load,
            solar_generation=solar_generation,
        )

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

            base_without_battery = self._estimate_base_without_battery(
                payload=payload,
                ev_kw=ev_kw,
                non_shiftable_load=non_shiftable_load,
                solar_generation=solar_generation,
            )
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

            base_without_battery = self._estimate_base_without_battery(
                payload=payload,
                ev_kw=ev_kw,
                non_shiftable_load=non_shiftable_load,
                solar_generation=solar_generation,
            )
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
        battery_kw = _clamp(battery_kw, battery_bounds.min_kw, battery_bounds.max_kw)

        actions = {
            "ev_charge_kw": ev_kw,
            "battery_kw": battery_kw,
        }

        quantized = {key: float(_round_one_decimal_towards_zero(value)) for key, value in actions.items()}
        quantized["ev_charge_kw"] = float(_clamp(quantized["ev_charge_kw"], 0.0, ev.max_kw))
        quantized["battery_kw"] = float(
            _clamp(quantized["battery_kw"], battery_bounds.min_kw, battery_bounds.max_kw)
        )

        base_without_battery = self._estimate_base_without_battery(
            payload=payload,
            ev_kw=quantized["ev_charge_kw"],
            non_shiftable_load=non_shiftable_load,
            solar_generation=solar_generation,
        )
        quantized["battery_kw"] = self._fit_quantized_battery_to_grid(
            base_without_battery,
            quantized["battery_kw"],
            battery_bounds,
            grid_import_limit_kw,
            grid_export_limit_kw,
        )
        net_grid_kw = base_without_battery + quantized["battery_kw"]
        if net_grid_kw > grid_import_limit_kw + 1e-6:
            local_constraint_flags["grid_import_limit_unmet"] = True
        if net_grid_kw < -grid_export_limit_kw - 1e-6:
            local_constraint_flags["grid_export_limit_unmet"] = True

        meter_import_kw = self._meter_component_kw(payload, "energy_in")
        meter_export_kw = self._meter_component_kw(payload, "energy_out")
        meter_net_kw = None
        if meter_import_kw is not None or meter_export_kw is not None:
            meter_net_kw = max(meter_import_kw or 0.0, 0.0) - max(meter_export_kw or 0.0, 0.0)

        ev_action_name = cfg.resolve_ev_action_name()
        battery_action_name = cfg.resolve_battery_action_name()
        if ev_action_name == battery_action_name:
            warnings.append("duplicate_action_names_fallback_to_defaults")
            ev_action_name = "ev_charge_kw"
            battery_action_name = "battery_kw"

        output_actions = {
            ev_action_name: quantized["ev_charge_kw"],
            battery_action_name: quantized["battery_kw"],
        }

        log.info(
            "rbc.actions",
            strategy="rh1_house_rbc_v1",
            actions=output_actions,
            connected={"ev": ev.connected},
            phase_totals={"grid": net_grid_kw},
            board_total=max(net_grid_kw, 0.0),
            grid_import_limit_kw=grid_import_limit_kw,
            grid_export_limit_kw=grid_export_limit_kw,
            price_now_eur_kwh=price_now,
            avg_future_price_eur_kwh=future_avg_price,
            meter_net_kw=meter_net_kw,
        )

        def fmt_kw(value: float) -> str:
            return f"{_round_one_decimal_towards_zero(value):.1f}"

        def fmt_optional_kw(value: float | None) -> str:
            if value is None:
                return "-"
            return fmt_kw(value)

        def fmt_price(value: float) -> str:
            return f"{value:.4f}"

        summary_lines = [
            "Strategy: rh1_house_rbc_v1",
            (
                f"Inputs: non_shiftable={fmt_kw(non_shiftable_load)} kW"
                f" solar={fmt_kw(solar_generation)} kW"
                f" meter_in={fmt_optional_kw(meter_import_kw)} kW"
                f" meter_out={fmt_optional_kw(meter_export_kw)} kW"
                f" meter_net={fmt_optional_kw(meter_net_kw)} kW"
            ),
            (
                f"Prices (EUR/kWh): now={fmt_price(price_now)}"
                f" h1={fmt_price(prices.get(1.0, price_now))}"
                f" h2={fmt_price(prices.get(2.0, price_now))}"
                f" h6={fmt_price(prices.get(6.0, price_now))}"
                f" h12={fmt_price(prices.get(12.0, price_now))}"
                f" h24={fmt_price(prices.get(24.0, price_now))}"
                f" avg_24h={fmt_price(future_avg_price)}"
            ),
            (
                f"EV: connected={'yes' if ev.connected else 'no'}"
                f" hard_min={fmt_kw(ev.hard_min_kw)} kW"
                f" max={fmt_kw(ev.max_kw)} kW"
                f" dispatch={fmt_kw(quantized['ev_charge_kw'])} kW"
            ),
            (
                f"Battery: soc={battery_soc * 100.0:.1f}%"
                f" bounds=[{fmt_kw(battery_bounds.min_kw)}, {fmt_kw(battery_bounds.max_kw)}] kW"
                f" dispatch={fmt_kw(quantized['battery_kw'])} kW"
            ),
            (
                f"Grid: base_wo_battery={fmt_kw(base_without_battery)} kW"
                f" import_limit={fmt_kw(grid_import_limit_kw)} kW"
                f" export_limit={fmt_kw(grid_export_limit_kw)} kW"
                f" net={fmt_kw(net_grid_kw)} kW"
            ),
            "Objective: minimize_import_cost_with_free_local_pv_self_consumption",
        ]
        if warnings:
            summary_lines.append(f"Fallbacks: {', '.join(warnings)}")
        if local_constraint_flags:
            summary_lines.append(
                "Constraint flags: " + ", ".join(sorted(local_constraint_flags.keys()))
            )
        log.info("rbc.summary\n{}\n", "\n".join(summary_lines))

        if warnings:
            log.warning("rh1.payload_fallbacks", issues=warnings)
        if local_constraint_flags:
            log.warning("rh1.constraint_flags", flags=local_constraint_flags)

        return output_actions

    def _extract_price_points(self, payload: Dict[str, Any], warnings: List[str]) -> Dict[float, float]:
        vector_points = self._extract_price_points_from_vector(payload, warnings)
        if vector_points is not None:
            return vector_points

        current = None
        current_candidates = [
            payload.get("electricity_pricing.current"),
            payload.get("electricity_pricing"),
            payload.get("energy_price"),
            payload.get("energy_tariffs.OMIE.energy_price.current"),
            payload.get("energy_tariffs.OMIE.energy_price"),
        ]
        for candidate in current_candidates:
            numeric = _maybe_float(candidate)
            if numeric is not None:
                current = numeric
                break

        if current is None:
            warnings.append("missing_price_current_defaulted")
            current = 0.0

        current = self._normalize_price(current)
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
                points[hour] = max(current, 0.0)
                continue
            points[hour] = max(self._normalize_price(value), 0.0)
        return points

    def _extract_price_points_from_vector(
        self, payload: Dict[str, Any], warnings: List[str]
    ) -> Dict[float, float] | None:
        prefixes = ("energy_tariffs.OMIE.energy_price", "energy_price")
        indexed_values: Dict[int, float] = {}
        selected_prefix: str | None = None
        for prefix in prefixes:
            candidate_values: Dict[int, float] = {}
            value_prefix = f"{prefix}.values["
            for key, raw in payload.items():
                if not isinstance(key, str):
                    continue
                if not key.startswith(value_prefix) or not key.endswith("]"):
                    continue
                idx_raw = key[len(value_prefix) : -1]
                if not idx_raw.isdigit():
                    continue
                numeric = _maybe_float(raw)
                if numeric is None:
                    continue
                candidate_values[int(idx_raw)] = numeric
            if candidate_values:
                indexed_values = candidate_values
                selected_prefix = prefix
                break

        if not indexed_values or selected_prefix is None:
            return None

        freq_seconds = _maybe_float(payload.get(f"{selected_prefix}.frequency_seconds"))
        if freq_seconds is None or freq_seconds <= 0.0:
            warnings.append("missing_energy_price_frequency_defaulted")
            freq_seconds = 900.0

        unit_hint = payload.get(f"{selected_prefix}.measurement_unit")
        current_value = indexed_values.get(0)
        if current_value is None:
            first_idx = min(indexed_values)
            current_value = indexed_values[first_idx]
            warnings.append("missing_energy_price_index_0_defaulted")

        points: Dict[float, float] = {
            0.0: max(self._normalize_price(current_value, unit_hint=unit_hint), 0.0)
        }
        for hour in (1.0, 2.0, 6.0, 12.0, 24.0):
            target_idx = int(round((hour * 3600.0) / freq_seconds))
            sampled = self._sample_indexed_price(indexed_values, target_idx)
            if sampled is None:
                warnings.append(f"missing_energy_price_index_{target_idx}_defaulted")
                sampled = current_value
            points[hour] = max(self._normalize_price(sampled, unit_hint=unit_hint), 0.0)

        return points

    def _sample_indexed_price(self, values: Dict[int, float], target_idx: int) -> float | None:
        if not values:
            return None
        if target_idx in values:
            return values[target_idx]
        ordered = sorted(values.items())
        first_idx, first_value = ordered[0]
        last_idx, last_value = ordered[-1]
        if target_idx <= first_idx:
            return first_value
        if target_idx >= last_idx:
            return last_value
        for i in range(1, len(ordered)):
            left_idx, left_val = ordered[i - 1]
            right_idx, right_val = ordered[i]
            if left_idx <= target_idx <= right_idx:
                if right_idx == left_idx:
                    return right_val
                alpha = (target_idx - left_idx) / float(right_idx - left_idx)
                return left_val + alpha * (right_val - left_val)
        return None

    def _normalize_price(self, value: float, unit_hint: Any = None) -> float:
        cfg = self.config
        mode = cfg.price_unit_mode
        normalized = value
        normalized_unit = str(unit_hint).strip().lower()
        if mode == "eur_mwh":
            normalized = value / 1000.0
        elif mode == "auto" and normalized_unit:
            if "mwh" in normalized_unit:
                normalized = value / 1000.0
        elif mode == "auto" and value > cfg.price_auto_mwh_threshold:
            normalized = value / 1000.0
        return normalized

    def _normalize_soc(self, value: float) -> float:
        mode = self.config.soc_unit_mode
        normalized = value
        if mode == "percent":
            normalized = value / 100.0
        elif mode == "auto" and 1.0 < value <= 100.0:
            normalized = value / 100.0
        return _clamp(normalized, 0.0, 1.0)

    def _extract_solar_generation(self, payload: Dict[str, Any], warnings: List[str]) -> float:
        direct = _maybe_float(payload.get("solar_generation"))
        if direct is not None:
            return max(direct, 0.0)

        total = 0.0
        found = False
        for key, raw in payload.items():
            if not isinstance(key, str):
                continue
            if not key.startswith("pv_panels.") or not key.endswith(".energy"):
                continue
            numeric = _maybe_float(raw)
            if numeric is None:
                continue
            found = True
            total += numeric

        if found:
            return max(total, 0.0)

        warnings.append("missing_solar_generation_defaulted")
        return 0.0

    def _extract_battery_soc(self, payload: Dict[str, Any], warnings: List[str]) -> float:
        for key in self.config.battery_soc_keys:
            numeric = _maybe_float(payload.get(key))
            if numeric is not None:
                return self._normalize_soc(numeric)

        fallback_key = self._find_battery_soc_fallback_key(payload)
        if fallback_key:
            numeric = _maybe_float(payload.get(fallback_key))
            if numeric is not None:
                warnings.append("battery_soc_from_fallback_key")
                return self._normalize_soc(numeric)

        warnings.append("missing_battery_soc_defaulted")
        return 0.5

    def _estimate_base_without_battery(
        self,
        payload: Dict[str, Any],
        ev_kw: float,
        non_shiftable_load: float,
        solar_generation: float,
    ) -> float:
        projected_meter_base = self._project_meter_net_without_battery(payload, ev_kw)
        if projected_meter_base is not None:
            return projected_meter_base
        # Local production is modeled as free self-consumption by subtracting PV.
        return non_shiftable_load + ev_kw - solar_generation

    def _project_meter_net_without_battery(self, payload: Dict[str, Any], ev_kw: float) -> float | None:
        meter_import = self._meter_component_kw(payload, "energy_in")
        meter_export = self._meter_component_kw(payload, "energy_out")
        if meter_import is None and meter_export is None:
            return None

        measured_net_kw = max(meter_import or 0.0, 0.0) - max(meter_export or 0.0, 0.0)
        measured_ev_kw = 0.0
        for charger_id in self.config.chargers:
            ev_raw = payload.get(f"charging_sessions.{charger_id}.electric_vehicle")
            ev_id = str(ev_raw).strip() if ev_raw is not None else ""
            if not ev_id:
                continue
            measured_ev_kw += max(
                _safe_float(payload.get(f"charging_sessions.{charger_id}.power"), 0.0),
                0.0,
            )
        return measured_net_kw + (ev_kw - measured_ev_kw)

    def _find_battery_soc_fallback_key(self, payload: Dict[str, Any]) -> str | None:
        for key in sorted(payload.keys()):
            if not isinstance(key, str):
                continue
            lowered = key.lower()
            if lowered.startswith("batteries.") and lowered.endswith(".soc"):
                return key
        return None

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

        soc = self._normalize_soc(soc)
        target_soc = self._normalize_soc(target_soc)
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

    def _fit_quantized_battery_to_grid(
        self,
        base_without_battery: float,
        battery_kw: float,
        bounds: BatteryPowerBounds,
        grid_import_limit_kw: float,
        grid_export_limit_kw: float,
    ) -> float:
        """Grid-fit helper for already-quantized battery actions.

        Uses 0.1 kW steps first, then falls back to an exact in-bounds correction when
        step granularity alone cannot satisfy the grid limits.
        """
        adjusted = _clamp(_round_one_decimal_towards_zero(battery_kw), bounds.min_kw, bounds.max_kw)
        net_grid = base_without_battery + adjusted
        step_kw = 0.1

        if net_grid > grid_import_limit_kw + 1e-9:
            while net_grid > grid_import_limit_kw + 1e-9 and adjusted - step_kw >= bounds.min_kw - 1e-9:
                adjusted = round(adjusted - step_kw, 10)
                net_grid = base_without_battery + adjusted

        if net_grid < -grid_export_limit_kw - 1e-9:
            while net_grid < -grid_export_limit_kw - 1e-9 and adjusted + step_kw <= bounds.max_kw + 1e-9:
                adjusted = round(adjusted + step_kw, 10)
                net_grid = base_without_battery + adjusted

        # If quantized stepping cannot satisfy limits but an exact in-bounds setpoint can,
        # use that exact value to keep the returned action grid-feasible.
        if net_grid > grid_import_limit_kw + 1e-9:
            exact = _clamp(grid_import_limit_kw - base_without_battery, bounds.min_kw, bounds.max_kw)
            if exact < adjusted - 1e-9:
                adjusted = exact
                net_grid = base_without_battery + adjusted

        if net_grid < -grid_export_limit_kw - 1e-9:
            exact = _clamp(-grid_export_limit_kw - base_without_battery, bounds.min_kw, bounds.max_kw)
            if exact > adjusted + 1e-9:
                adjusted = exact

        return float(_clamp(adjusted, bounds.min_kw, bounds.max_kw))
