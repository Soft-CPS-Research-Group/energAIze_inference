from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, List

from .forecasts import ForecastSocGuidance, build_forecast_soc_guidance, read_site_forecasts
from app.logging import get_logger


PRICE_POINTS_HOURS = (0.0, 1.0, 2.0, 6.0, 12.0, 24.0)
VALID_PRICE_UNIT_MODES = {"auto", "eur_kwh", "eur_mwh"}
VALID_SOC_UNIT_MODES = {"auto", "fraction", "percent"}
DEFAULT_EV_FLEX_FIELDS = {
    "soc": "electric_vehicles.{ev_id}.SoC",
    "target_soc": "electric_vehicles.{ev_id}.flexibility.estimated_soc_at_departure",
    "departure_time": "electric_vehicles.{ev_id}.flexibility.estimated_time_at_departure",
}
MIN_CONTROL_INTERVAL_MINUTES = 1.0 / 60.0


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


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _extract_prefixed_fields(payload: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix:
            fields[suffix] = value
    return dict(sorted(fields.items()))


def _round_one_decimal_towards_zero(value: float) -> float:
    scaled = value * 10.0
    if scaled >= 0:
        return math.floor(scaled) / 10.0
    return math.ceil(scaled) / 10.0


@dataclass
class Rh1HouseConfig:
    grid_import_limit_kw: float
    control_interval_minutes: float = 0.25
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
    ev_use_flexibility: bool = True
    ev_always_on_when_connected: bool = False
    price_unit_mode: str = "auto"
    price_auto_mwh_threshold: float = 3.0
    price_quantile_cheap: float = 0.35
    price_quantile_expensive: float = 0.70
    soc_unit_mode: str = "auto"
    battery_soc_keys: List[str] = field(default_factory=lambda: ["electrical_storage.soc"])
    community_participation_enabled: bool = False
    community_energy_in_key: str = "community.energy_in_total"
    community_energy_out_key: str = "community.energy_out_total"
    community_energy_deadband_kw: float = 0.2
    community_deficit_weight: float = 1.0
    community_surplus_weight: float = 1.0
    community_dispatch_weight: float = 0.2
    community_dispatch_cap_kw: float = 0.0
    community_price_signal_key: str = "community.price_signal"
    forecast_support_enabled: bool = False
    forecast_consumption_prefix: str = "forecasts.ConsumptionForecastService.consumption_total"
    forecast_production_prefix: str = "forecasts.ProductionForecastService.production_total"
    forecast_window_hours: float = 2.0
    forecast_soc_bias_gain: float = 0.5
    forecast_soc_bias_max: float = 0.15
    forecast_dispatch_weight: float = 0.25
    reserve_soc_cheap: float = 0.35
    reserve_soc_neutral: float = 0.30
    reserve_soc_expensive: float = 0.25
    target_soc_cheap: float = 0.85
    target_soc_neutral: float = 0.70
    target_soc_expensive: float = 0.60
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
        community_energy_in_key = str(
            data.pop("community_energy_in_key", "community.energy_in_total")
        ).strip() or "community.energy_in_total"
        community_energy_out_key = str(
            data.pop("community_energy_out_key", "community.energy_out_total")
        ).strip() or "community.energy_out_total"
        community_price_signal_key = str(
            data.pop("community_price_signal_key", "community.price_signal")
        ).strip() or "community.price_signal"

        cfg = cls(
            grid_import_limit_kw=grid_import_limit_kw,
            control_interval_minutes=max(
                _safe_float(data.pop("control_interval_minutes", 0.25), 0.25),
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
            ev_use_flexibility=bool(data.pop("ev_use_flexibility", True)),
            ev_always_on_when_connected=bool(data.pop("ev_always_on_when_connected", False)),
            price_unit_mode=price_unit_mode,
            price_auto_mwh_threshold=max(
                _safe_float(data.pop("price_auto_mwh_threshold", 3.0), 3.0),
                0.0,
            ),
            price_quantile_cheap=_clamp(
                _safe_float(data.pop("price_quantile_cheap", 0.35), 0.35),
                0.0,
                1.0,
            ),
            price_quantile_expensive=_clamp(
                _safe_float(data.pop("price_quantile_expensive", 0.70), 0.70),
                0.0,
                1.0,
            ),
            soc_unit_mode=soc_unit_mode,
            battery_soc_keys=battery_soc_keys,
            community_participation_enabled=bool(
                data.pop("community_participation_enabled", False)
            ),
            community_energy_in_key=community_energy_in_key,
            community_energy_out_key=community_energy_out_key,
            community_energy_deadband_kw=max(
                _safe_float(data.pop("community_energy_deadband_kw", 0.2), 0.2),
                0.0,
            ),
            community_deficit_weight=max(
                _safe_float(data.pop("community_deficit_weight", 1.0), 1.0),
                0.0,
            ),
            community_surplus_weight=max(
                _safe_float(data.pop("community_surplus_weight", 1.0), 1.0),
                0.0,
            ),
            community_dispatch_weight=max(
                _safe_float(data.pop("community_dispatch_weight", 0.2), 0.2),
                0.0,
            ),
            community_dispatch_cap_kw=max(
                _safe_float(data.pop("community_dispatch_cap_kw", 0.0), 0.0),
                0.0,
            ),
            community_price_signal_key=community_price_signal_key,
            forecast_support_enabled=bool(data.pop("forecast_support_enabled", False)),
            forecast_consumption_prefix=str(
                data.pop(
                    "forecast_consumption_prefix",
                    "forecasts.ConsumptionForecastService.consumption_total",
                )
            ).strip()
            or "forecasts.ConsumptionForecastService.consumption_total",
            forecast_production_prefix=str(
                data.pop(
                    "forecast_production_prefix",
                    "forecasts.ProductionForecastService.production_total",
                )
            ).strip()
            or "forecasts.ProductionForecastService.production_total",
            forecast_window_hours=max(
                _safe_float(data.pop("forecast_window_hours", 2.0), 2.0),
                0.0,
            ),
            forecast_soc_bias_gain=max(
                _safe_float(data.pop("forecast_soc_bias_gain", 0.5), 0.5),
                0.0,
            ),
            forecast_soc_bias_max=_clamp(
                _safe_float(data.pop("forecast_soc_bias_max", 0.15), 0.15),
                0.0,
                1.0,
            ),
            forecast_dispatch_weight=max(
                _safe_float(data.pop("forecast_dispatch_weight", 0.25), 0.25),
                0.0,
            ),
            reserve_soc_cheap=_clamp(
                _safe_float(data.pop("reserve_soc_cheap", 0.35), 0.35),
                0.0,
                1.0,
            ),
            reserve_soc_neutral=_clamp(
                _safe_float(data.pop("reserve_soc_neutral", 0.30), 0.30),
                0.0,
                1.0,
            ),
            reserve_soc_expensive=_clamp(
                _safe_float(data.pop("reserve_soc_expensive", 0.25), 0.25),
                0.0,
                1.0,
            ),
            target_soc_cheap=_clamp(
                _safe_float(data.pop("target_soc_cheap", 0.85), 0.85),
                0.0,
                1.0,
            ),
            target_soc_neutral=_clamp(
                _safe_float(data.pop("target_soc_neutral", 0.70), 0.70),
                0.0,
                1.0,
            ),
            target_soc_expensive=_clamp(
                _safe_float(data.pop("target_soc_expensive", 0.60), 0.60),
                0.0,
                1.0,
            ),
            ev_action_name=ev_action_name,
            battery_action_name=battery_action_name,
        )
        if cfg.battery_soc_max < cfg.battery_soc_min:
            cfg.battery_soc_max = cfg.battery_soc_min

        cfg.price_quantile_cheap = _clamp(cfg.price_quantile_cheap, 0.0, 1.0)
        cfg.price_quantile_expensive = _clamp(cfg.price_quantile_expensive, 0.0, 1.0)
        if cfg.price_quantile_expensive < cfg.price_quantile_cheap:
            cfg.price_quantile_expensive = cfg.price_quantile_cheap

        cfg.reserve_soc_cheap = _clamp(
            cfg.reserve_soc_cheap,
            cfg.battery_soc_min,
            cfg.battery_soc_max,
        )
        cfg.reserve_soc_neutral = _clamp(
            cfg.reserve_soc_neutral,
            cfg.battery_soc_min,
            cfg.battery_soc_max,
        )
        cfg.reserve_soc_expensive = _clamp(
            cfg.reserve_soc_expensive,
            cfg.battery_soc_min,
            cfg.battery_soc_max,
        )

        cfg.target_soc_cheap = _clamp(
            cfg.target_soc_cheap,
            cfg.reserve_soc_cheap,
            cfg.battery_soc_max,
        )
        cfg.target_soc_neutral = _clamp(
            cfg.target_soc_neutral,
            cfg.reserve_soc_neutral,
            cfg.battery_soc_max,
        )
        cfg.target_soc_expensive = _clamp(
            cfg.target_soc_expensive,
            cfg.reserve_soc_expensive,
            cfg.battery_soc_max,
        )
        return cfg

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
    details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CommunityTargetDecision:
    raw_target_kw: float
    effective_target_kw: float
    reserve_limited: bool
    soc_recovery_target_kw: float


class Rh1HouseRuntime:
    def __init__(self, config: Rh1HouseConfig):
        self.config = config

    def _decision_interval_hours(self) -> float:
        return max(self.config.control_interval_minutes, MIN_CONTROL_INTERVAL_MINUTES) / 60.0

    def _meter_component_kwh(self, payload: Dict[str, Any], component: str) -> float | None:
        """Read RH01 grid meter import/export energy with support for total, legacy and phase schemas."""
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

    def _meter_component_kw(self, payload: Dict[str, Any], component: str) -> float | None:
        energy_kwh = self._meter_component_kwh(payload, component)
        if energy_kwh is None:
            return None
        return energy_kwh / max(self._decision_interval_hours(), 1e-9)

    def _community_component_kw(self, payload: Dict[str, Any], key: str) -> float | None:
        if not key:
            return None
        energy_kwh = _maybe_float(payload.get(key))
        if energy_kwh is None:
            return None
        return max(energy_kwh, 0.0) / max(self._decision_interval_hours(), 1e-9)

    def _community_net_kw(self, payload: Dict[str, Any]) -> float | None:
        cfg = self.config
        if not cfg.community_participation_enabled:
            return None
        import_kw = self._community_component_kw(payload, cfg.community_energy_in_key)
        export_kw = self._community_component_kw(payload, cfg.community_energy_out_key)
        if import_kw is None or export_kw is None:
            return None
        return import_kw - export_kw

    def allocate(self, payload: Dict[str, Any]) -> Dict[str, float]:
        cfg = self.config
        log = get_logger()
        warnings: List[str] = []
        local_constraint_flags: Dict[str, bool] = {}

        dt_hours = max(cfg.control_interval_minutes, MIN_CONTROL_INTERVAL_MINUTES) / 60.0
        now = _now_from_payload(payload)

        prices = self._extract_price_points(payload, warnings)
        price_now = prices[0.0]
        future_avg_price = self._average_interpolated_price(prices)
        price_regime = self._price_regime(payload, prices, price_now)

        non_shiftable_load = _safe_float(payload.get("non_shiftable_load"), 0.0)
        solar_generation = self._extract_solar_generation(payload, warnings)
        community_import_kw = self._community_component_kw(payload, cfg.community_energy_in_key)
        community_export_kw = self._community_component_kw(payload, cfg.community_energy_out_key)
        community_net_kw = self._community_net_kw(payload)

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
        reserve_floor_soc, charge_target_soc = self._soc_profile_for_regime(price_regime)

        ev = self._ev_action_bounds(payload, now, dt_hours)
        battery_bounds = self._battery_power_bounds(battery_soc, dt_hours)
        forecast_guidance, forecast_issues = self._forecast_guidance(
            payload=payload,
            battery_soc=battery_soc,
            battery_bounds=battery_bounds,
            reserve_floor_soc=reserve_floor_soc,
            charge_target_soc=charge_target_soc,
        )
        warnings.extend(f"forecast:{issue}" for issue in forecast_issues)
        forecast_target_kw = None
        if forecast_guidance is not None:
            reserve_floor_soc = forecast_guidance.reserve_floor_soc
            charge_target_soc = forecast_guidance.target_soc
            forecast_target_kw = forecast_guidance.target_kw
        community_target = self._community_battery_target_kw(
            battery_bounds=battery_bounds,
            soc=battery_soc,
            dt_hours=dt_hours,
            community_net_kw=community_net_kw,
            reserve_floor_soc=reserve_floor_soc,
            charge_target_soc=charge_target_soc,
            price_regime=price_regime,
        )
        community_target_kw = community_target.effective_target_kw
        if cfg.community_participation_enabled:
            ev_kw, _, _ = self._optimize_joint_dispatch(
                payload=payload,
                ev=ev,
                battery_soc=battery_soc,
                battery_bounds=battery_bounds,
                non_shiftable_load=non_shiftable_load,
                solar_generation=solar_generation,
                price_now=price_now,
                future_avg_price=future_avg_price,
                grid_import_limit_kw=grid_import_limit_kw,
                grid_export_limit_kw=grid_export_limit_kw,
                dt_hours=dt_hours,
                community_target_kw=0.0,
                forecast_target_kw=forecast_target_kw,
            )
            battery_kw, base_without_battery = self._optimize_battery_dispatch_for_fixed_ev(
                payload=payload,
                ev_kw=ev_kw,
                ev_hard_min_kw=ev.hard_min_kw,
                battery_bounds=battery_bounds,
                non_shiftable_load=non_shiftable_load,
                solar_generation=solar_generation,
                price_now=price_now,
                future_avg_price=future_avg_price,
                grid_import_limit_kw=grid_import_limit_kw,
                grid_export_limit_kw=grid_export_limit_kw,
                dt_hours=dt_hours,
                community_target_kw=community_target_kw,
                forecast_target_kw=forecast_target_kw,
            )
        else:
            ev_kw, battery_kw, base_without_battery = self._optimize_joint_dispatch(
                payload=payload,
                ev=ev,
                battery_soc=battery_soc,
                battery_bounds=battery_bounds,
                non_shiftable_load=non_shiftable_load,
                solar_generation=solar_generation,
                price_now=price_now,
                future_avg_price=future_avg_price,
                grid_import_limit_kw=grid_import_limit_kw,
                grid_export_limit_kw=grid_export_limit_kw,
                dt_hours=dt_hours,
                community_target_kw=community_target_kw,
                forecast_target_kw=forecast_target_kw,
            )

        ev_floor_kw = ev.hard_min_kw if (ev.connected and cfg.ev_always_on_when_connected) else 0.0
        ev_kw = _clamp(ev_kw, ev_floor_kw, ev.max_kw)
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
            ev_reducible_kw = max(ev_kw - ev_floor_kw, 0.0)
            reduced = min(shortfall, ev_reducible_kw)
            ev_kw -= reduced
            if ev_floor_kw <= 1e-6 and ev_kw + 1e-6 < ev.hard_min_kw:
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
                if ev_floor_kw > 1e-6:
                    local_constraint_flags["ev_floor_forced_connected"] = True

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

        ev_kw = _clamp(ev_kw, ev_floor_kw, ev.max_kw)
        battery_kw = _clamp(battery_kw, battery_bounds.min_kw, battery_bounds.max_kw)

        actions = {
            "ev_charge_kw": ev_kw,
            "battery_kw": battery_kw,
        }

        quantized = {key: float(_round_one_decimal_towards_zero(value)) for key, value in actions.items()}
        quantized["ev_charge_kw"] = float(_clamp(quantized["ev_charge_kw"], ev_floor_kw, ev.max_kw))
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

        meter_import_kwh = self._meter_component_kwh(payload, "energy_in")
        meter_export_kwh = self._meter_component_kwh(payload, "energy_out")
        meter_import_kw = self._meter_component_kw(payload, "energy_in")
        meter_export_kw = self._meter_component_kw(payload, "energy_out")
        meter_net_kwh = None
        if meter_import_kwh is not None or meter_export_kwh is not None:
            meter_net_kwh = max(meter_import_kwh or 0.0, 0.0) - max(meter_export_kwh or 0.0, 0.0)
        meter_net_kw = None
        if meter_import_kw is not None or meter_export_kw is not None:
            meter_net_kw = max(meter_import_kw or 0.0, 0.0) - max(meter_export_kw or 0.0, 0.0)
        community_import_kwh = (
            _maybe_float(payload.get(cfg.community_energy_in_key))
            if cfg.community_participation_enabled
            else None
        )
        community_export_kwh = (
            _maybe_float(payload.get(cfg.community_energy_out_key))
            if cfg.community_participation_enabled
            else None
        )
        community_net_kwh = None
        if community_import_kwh is not None and community_export_kwh is not None:
            community_net_kwh = max(community_import_kwh, 0.0) - max(community_export_kwh, 0.0)

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
        community_alignment_penalty = self._community_alignment_penalty(
            battery_kw=quantized["battery_kw"],
            community_target_kw=community_target_kw,
            dt_hours=dt_hours,
        )

        log.info(
            "rbc.actions",
            strategy="rh1_house_rbc_v1",
            actions=output_actions,
            connected={"ev": ev.connected},
            ev_flex=ev.details,
            phase_totals={"grid": net_grid_kw},
            board_total=max(net_grid_kw, 0.0),
            grid_import_limit_kw=grid_import_limit_kw,
            grid_export_limit_kw=grid_export_limit_kw,
            price_now_eur_kwh=price_now,
            avg_future_price_eur_kwh=future_avg_price,
            meter_import_kwh=meter_import_kwh,
            meter_export_kwh=meter_export_kwh,
            meter_net_kwh=meter_net_kwh,
            meter_net_kw=meter_net_kw,
            community_import_kwh=community_import_kwh,
            community_export_kwh=community_export_kwh,
            community_net_kwh=community_net_kwh,
            community_net_kw=community_net_kw,
            community_target_kw=community_target_kw,
            community_target_raw_kw=community_target.raw_target_kw,
            effective_community_target_kw=community_target.effective_target_kw,
            community_reserve_limited=community_target.reserve_limited,
            soc_recovery_target_kw=community_target.soc_recovery_target_kw,
            community_alignment_penalty=community_alignment_penalty,
            price_regime=price_regime,
            reserve_floor_soc=reserve_floor_soc,
            charge_target_soc=charge_target_soc,
            forecast_window_net_energy_kwh=(
                forecast_guidance.window_net_energy_kwh if forecast_guidance else None
            ),
            forecast_window_avg_net_kw=(
                forecast_guidance.window_avg_net_kw if forecast_guidance else None
            ),
            forecast_imbalance_ratio=(
                forecast_guidance.imbalance_ratio if forecast_guidance else None
            ),
            forecast_target_kw=forecast_target_kw,
        )

        def fmt_kw(value: float) -> str:
            return f"{_round_one_decimal_towards_zero(value):.1f}"

        def fmt_optional_kw(value: float | None) -> str:
            if value is None:
                return "-"
            return fmt_kw(value)

        def fmt_meter(energy_kwh: float | None, power_kw: float | None) -> str:
            if energy_kwh is None or power_kw is None:
                return "-"
            return f"{energy_kwh:.4f} kWh ({power_kw:.2f} kW)"

        def fmt_price(value: float) -> str:
            return f"{value:.4f}"

        def fmt_optional_pct(value: float | None) -> str:
            if value is None:
                return "-"
            return f"{value * 100.0:.1f}%"

        def fmt_optional_num(value: float | None) -> str:
            if value is None:
                return "-"
            return f"{value:.4f}"

        def fmt_optional_minutes(value: float | None) -> str:
            if value is None:
                return "-"
            return f"{value:.1f}"

        def fmt_optional_text(value: Any) -> str:
            text = _clean_text(value)
            return text or "-"

        summary_lines = [
            "Strategy: rh1_house_rbc_v1",
            (
                f"Inputs: non_shiftable={fmt_kw(non_shiftable_load)} kW"
                f" solar={fmt_kw(solar_generation)} kW"
                f" meter_in={fmt_meter(meter_import_kwh, meter_import_kw)}"
                f" meter_out={fmt_meter(meter_export_kwh, meter_export_kw)}"
                f" meter_net={fmt_meter(meter_net_kwh, meter_net_kw)}"
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
                f"Policy: price_regime={price_regime}"
                f" reserve_floor_soc={reserve_floor_soc * 100.0:.1f}%"
                f" charge_target_soc={charge_target_soc * 100.0:.1f}%"
                f" forecast_target={fmt_optional_kw(forecast_target_kw)} kW"
            ),
            (
                f"EV: connected={'yes' if ev.connected else 'no'}"
                f" hard_min={fmt_kw(ev.hard_min_kw)} kW"
                f" floor={fmt_kw(ev_floor_kw)} kW"
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
            (
                "Objective: minimize_import_cost_with_free_local_pv_self_consumption"
                " + community_battery_alignment"
            ),
        ]
        for detail in ev.details:
            summary_lines.append(
                (
                    f"  EV detail: charger={detail.get('charger_id', '-')}"
                    f" ev={detail.get('ev_id', '-')}"
                    f" flex_enabled={'yes' if detail.get('flexibility_enabled') else 'no'}"
                    f" reason={fmt_optional_text(detail.get('reason'))}"
                    f" req={fmt_optional_kw(detail.get('required_kw'))} kW"
                    f" prio={fmt_optional_num(detail.get('priority'))}"
                    f" soc={fmt_optional_pct(detail.get('soc'))}"
                    f" target={fmt_optional_pct(detail.get('target_soc'))}"
                    f" arrival_soc={fmt_optional_pct(detail.get('arrival_soc'))}"
                    f" departure_soc={fmt_optional_pct(detail.get('departure_soc'))}"
                    f" gap_kwh={fmt_optional_num(detail.get('energy_gap_kwh'))}"
                    f" capacity_kwh={fmt_optional_num(detail.get('capacity_kwh'))}"
                    f" arrival={fmt_optional_text(detail.get('arrival_time'))}"
                    f" departure={fmt_optional_text(detail.get('departure_time'))}"
                    f" mins_to_departure={fmt_optional_minutes(detail.get('minutes_remaining'))}"
                    f" mode={fmt_optional_text(detail.get('mode'))}"
                    f" flex_charger={fmt_optional_text(detail.get('flex_charger'))}"
                    f" missing={','.join(detail.get('missing_fields') or []) or '-'}"
                )
            )
        if community_import_kw is not None and community_export_kw is not None:
            summary_lines.append(
                (
                    f"Community: in={fmt_meter(community_import_kwh, community_import_kw)}"
                    f" out={fmt_meter(community_export_kwh, community_export_kw)}"
                    f" net={fmt_meter(community_net_kwh, community_net_kw)}"
                    f" deadband={fmt_kw(cfg.community_energy_deadband_kw)} kW"
                    f" target_battery_raw={fmt_kw(community_target.raw_target_kw)} kW"
                    f" soc_recovery_target_kw={fmt_kw(community_target.soc_recovery_target_kw)} kW"
                    f" effective_community_target_kw={fmt_kw(community_target.effective_target_kw)} kW"
                    f" reserve_limited={'yes' if community_target.reserve_limited else 'no'}"
                    f" alignment_penalty={community_alignment_penalty:.6f}"
                )
            )
        else:
            summary_lines.append(
                (
                    f"Community: target_battery_raw={fmt_kw(community_target.raw_target_kw)} kW"
                    f" soc_recovery_target_kw={fmt_kw(community_target.soc_recovery_target_kw)} kW"
                    f" effective_community_target_kw={fmt_kw(community_target.effective_target_kw)} kW"
                    f" reserve_limited={'yes' if community_target.reserve_limited else 'no'}"
                    f" alignment_penalty={community_alignment_penalty:.6f}"
                )
            )
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
        # RH1 telemetry provides PV production as interval energy (kWh),
        # so convert to power using the configured control interval.
        interval_hours = max(self._decision_interval_hours(), 1e-9)
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
            return max(total, 0.0) / interval_hours

        direct = _maybe_float(payload.get("solar_generation"))
        if direct is not None:
            return max(direct, 0.0)

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

    def _extract_price_series_for_quantiles(
        self,
        payload: Dict[str, Any],
        fallback_points: Dict[float, float],
    ) -> List[float]:
        prefixes = ("energy_tariffs.OMIE.energy_price", "energy_price")
        for prefix in prefixes:
            entries: List[tuple[int, float]] = []
            value_prefix = f"{prefix}.values["
            for key, raw in payload.items():
                if not isinstance(key, str):
                    continue
                if not key.startswith(value_prefix) or not key.endswith("]"):
                    continue
                idx_raw = key[len(value_prefix) : -1]
                if not idx_raw.isdigit():
                    continue
                value = _maybe_float(raw)
                if value is None:
                    continue
                entries.append((int(idx_raw), value))
            if entries:
                unit_hint = payload.get(f"{prefix}.measurement_unit")
                return [
                    max(self._normalize_price(value, unit_hint=unit_hint), 0.0)
                    for _, value in sorted(entries, key=lambda item: item[0])
                ]
        return [max(value, 0.0) for value in fallback_points.values()]

    def _quantile(self, values: List[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        quantile = _clamp(q, 0.0, 1.0)
        position = quantile * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        alpha = position - lower
        return ordered[lower] + alpha * (ordered[upper] - ordered[lower])

    def _price_regime(
        self,
        payload: Dict[str, Any],
        price_points: Dict[float, float],
        price_now: float,
    ) -> str:
        cfg = self.config
        series = self._extract_price_series_for_quantiles(payload, price_points)
        cheap_threshold = self._quantile(series, cfg.price_quantile_cheap)
        expensive_threshold = self._quantile(series, cfg.price_quantile_expensive)
        if price_now <= cheap_threshold + 1e-9:
            return "cheap"
        if price_now >= expensive_threshold - 1e-9:
            return "expensive"
        return "neutral"

    def _soc_profile_for_regime(self, price_regime: str) -> tuple[float, float]:
        cfg = self.config
        if price_regime == "cheap":
            reserve_floor_soc = cfg.reserve_soc_cheap
            charge_target_soc = cfg.target_soc_cheap
        elif price_regime == "expensive":
            reserve_floor_soc = cfg.reserve_soc_expensive
            charge_target_soc = cfg.target_soc_expensive
        else:
            reserve_floor_soc = cfg.reserve_soc_neutral
            charge_target_soc = cfg.target_soc_neutral
        reserve_floor_soc = _clamp(reserve_floor_soc, cfg.battery_soc_min, cfg.battery_soc_max)
        charge_target_soc = _clamp(charge_target_soc, reserve_floor_soc, cfg.battery_soc_max)
        return reserve_floor_soc, charge_target_soc

    def _forecast_guidance(
        self,
        payload: Dict[str, Any],
        battery_soc: float,
        battery_bounds: BatteryPowerBounds,
        reserve_floor_soc: float,
        charge_target_soc: float,
    ) -> tuple[ForecastSocGuidance | None, List[str]]:
        cfg = self.config
        if not cfg.forecast_support_enabled:
            return None, []

        read_result = read_site_forecasts(
            payload,
            consumption_prefix=cfg.forecast_consumption_prefix,
            production_prefix=cfg.forecast_production_prefix,
        )
        if read_result.snapshot is None:
            return None, list(read_result.issues)

        guidance = build_forecast_soc_guidance(
            read_result.snapshot,
            window_hours=cfg.forecast_window_hours,
            capacity_kwh=cfg.battery_capacity_kwh,
            current_soc=battery_soc,
            base_reserve_soc=reserve_floor_soc,
            base_target_soc=charge_target_soc,
            soc_min=cfg.battery_soc_min,
            soc_max=cfg.battery_soc_max,
            charge_limit_kw=max(battery_bounds.max_kw, 0.0),
            discharge_limit_kw=max(-battery_bounds.min_kw, 0.0),
            soc_bias_gain=cfg.forecast_soc_bias_gain,
            soc_bias_max=cfg.forecast_soc_bias_max,
            dispatch_weight=cfg.forecast_dispatch_weight,
        )
        return guidance, list(read_result.issues)

    def _build_ev_flex_detail(
        self,
        payload: Dict[str, Any],
        ev_id: str,
        charger_id: str,
        charger_min_kw: float,
        charger_max_kw: float,
        now: datetime,
        control_minutes: float,
        flexibility_enabled: bool,
    ) -> Dict[str, Any]:
        cfg = self.config
        soc_key = cfg.resolve_ev_field("soc", ev_id)
        target_key = cfg.resolve_ev_field("target_soc", ev_id)
        departure_key = cfg.resolve_ev_field("departure_time", ev_id)
        raw_prefix = f"electric_vehicles.{ev_id}.flexibility."
        raw_flex_fields = _extract_prefixed_fields(payload, raw_prefix)

        soc_raw = _maybe_float(payload.get(soc_key))
        soc = self._normalize_soc(soc_raw) if soc_raw is not None else None

        target_soc_raw = _maybe_float(payload.get(target_key))
        if target_soc_raw is not None and target_soc_raw > 0.0:
            target_soc = self._normalize_soc(target_soc_raw)
            if soc is not None:
                target_soc = _clamp(target_soc, soc, 1.0)
        else:
            target_soc = None

        arrival_soc_raw = _maybe_float(raw_flex_fields.get("estimated_soc_at_arrival"))
        arrival_soc = self._normalize_soc(arrival_soc_raw) if arrival_soc_raw is not None else None
        departure_soc_raw = _maybe_float(raw_flex_fields.get("estimated_soc_at_departure"))
        departure_soc = (
            self._normalize_soc(departure_soc_raw) if departure_soc_raw is not None else None
        )

        arrival_time = _clean_text(raw_flex_fields.get("estimated_time_at_arrival"))
        departure_time = _clean_text(payload.get(departure_key)) or _clean_text(
            raw_flex_fields.get("estimated_time_at_departure")
        )
        departure_dt = _parse_datetime(payload.get(departure_key)) or _parse_datetime(departure_time)
        if departure_dt is not None:
            departure_dt = departure_dt.astimezone(timezone.utc)

        minutes_remaining = None
        if departure_dt is not None:
            minutes_remaining = max(
                (departure_dt - now).total_seconds() / 60.0,
                control_minutes,
            )

        capacity_kwh = max(cfg.vehicle_capacities.get(ev_id, cfg.ev_default_capacity_kwh), 1.0)
        energy_gap_kwh = None
        if soc is not None and target_soc is not None:
            energy_gap_kwh = max(target_soc - soc, 0.0) * capacity_kwh

        missing_fields: List[str] = []
        if soc_raw is None:
            missing_fields.append("soc")
        if target_soc_raw is None or target_soc_raw <= 0.0:
            missing_fields.append("target_soc")
        if departure_dt is None and not departure_time:
            missing_fields.append("departure_time")

        return {
            "charger_id": charger_id,
            "ev_id": ev_id,
            "flexibility_enabled": flexibility_enabled,
            "charger_min_kw": charger_min_kw,
            "charger_max_kw": charger_max_kw,
            "soc_raw": soc_raw,
            "soc": soc,
            "target_soc_raw": target_soc_raw,
            "target_soc": target_soc,
            "arrival_soc_raw": arrival_soc_raw,
            "arrival_soc": arrival_soc,
            "departure_soc_raw": departure_soc_raw,
            "departure_soc": departure_soc,
            "arrival_time": arrival_time,
            "departure_time": departure_time,
            "minutes_remaining": minutes_remaining,
            "capacity_kwh": capacity_kwh,
            "energy_gap_kwh": energy_gap_kwh,
            "mode": _clean_text(raw_flex_fields.get("mode")),
            "flex_charger": _clean_text(raw_flex_fields.get("charger")),
            "missing_fields": missing_fields,
            "raw_flex_fields": raw_flex_fields,
        }

    def _required_ev_power_from_detail(
        self,
        detail: Dict[str, Any],
        charger_min_kw: float,
        charger_max_kw: float,
        control_minutes: float,
        dt_hours: float,
    ) -> float:
        cfg = self.config
        soc = detail.get("soc")
        target_soc = detail.get("target_soc")
        minutes_remaining = detail.get("minutes_remaining")
        gap_kwh = detail.get("energy_gap_kwh")

        detail.update({"eligible": False, "required_kw": cfg.ev_min_connected_kw, "priority": None})

        if soc is None or target_soc is None:
            detail["reason"] = "missing_soc_or_target"
            return cfg.ev_min_connected_kw
        if target_soc <= soc + 1e-6:
            detail["reason"] = "target_already_met"
            return cfg.ev_min_connected_kw
        if gap_kwh is None or gap_kwh <= 1e-9:
            detail["reason"] = "no_energy_gap"
            return cfg.ev_min_connected_kw
        if minutes_remaining is None:
            detail["reason"] = "missing_departure_time"
            return cfg.ev_min_connected_kw

        if minutes_remaining <= control_minutes + 1e-9:
            required_kw = charger_max_kw
        else:
            required_kw = gap_kwh / max(minutes_remaining / 60.0, dt_hours)
        required_kw = _clamp(required_kw, charger_min_kw, charger_max_kw)
        detail.update(
            {
                "eligible": True,
                "required_kw": required_kw,
                "priority": required_kw / charger_max_kw if charger_max_kw > 0 else 1.0,
                "reason": "schedulable",
            }
        )
        return required_kw

    def _ev_action_bounds(self, payload: Dict[str, Any], now: datetime, dt_hours: float) -> EvAggregate:
        cfg = self.config
        control_minutes = max(cfg.control_interval_minutes, MIN_CONTROL_INTERVAL_MINUTES)
        hard_min_total = 0.0
        max_total = 0.0
        connected_any = False
        details: List[Dict[str, Any]] = []

        for charger_id, meta in cfg.chargers.items():
            ev_raw = payload.get(f"charging_sessions.{charger_id}.electric_vehicle")
            ev_id = str(ev_raw).strip() if ev_raw is not None else ""
            if not ev_id:
                continue
            connected_any = True

            charger_min = max(_safe_float(meta.get("min_kw"), cfg.ev_min_connected_kw), cfg.ev_min_connected_kw)
            charger_max = max(_safe_float(meta.get("max_kw"), 22.0), charger_min)
            max_total += charger_max

            detail = self._build_ev_flex_detail(
                payload=payload,
                ev_id=ev_id,
                charger_id=charger_id,
                charger_min_kw=charger_min,
                charger_max_kw=charger_max,
                now=now,
                control_minutes=control_minutes,
                flexibility_enabled=cfg.ev_use_flexibility,
            )

            if cfg.ev_use_flexibility:
                required_kw = self._required_ev_power_from_detail(
                    detail=detail,
                    charger_min_kw=charger_min,
                    charger_max_kw=charger_max,
                    control_minutes=control_minutes,
                    dt_hours=dt_hours,
                )
            else:
                required_kw = charger_min
                detail.update(
                    {
                        "eligible": False,
                        "required_kw": required_kw,
                        "priority": None,
                        "reason": "flexibility_disabled",
                    }
                )
            details.append(detail)
            hard_min_total += max(required_kw, charger_min)

        if not connected_any:
            return EvAggregate(connected=False, hard_min_kw=0.0, max_kw=0.0, details=details)

        hard_min_total = _clamp(hard_min_total, 0.0, max_total)
        return EvAggregate(
            connected=True,
            hard_min_kw=hard_min_total,
            max_kw=max_total,
            details=details,
        )

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

    def _candidate_values(self, low: float, high: float, step: float = 0.1) -> List[float]:
        low = float(low)
        high = float(high)
        if high < low:
            high = low
        if high - low <= 1e-9:
            return [low]

        values = [low, high]
        start_idx = int(math.ceil((low - 1e-9) / step))
        end_idx = int(math.floor((high + 1e-9) / step))
        for idx in range(start_idx, end_idx + 1):
            candidate = round(idx * step, 10)
            if low - 1e-9 <= candidate <= high + 1e-9:
                values.append(candidate)

        unique_sorted = sorted({round(value, 10) for value in values})
        return [float(v) for v in unique_sorted]

    def _battery_energy_delta_kwh(self, battery_kw: float, dt_hours: float) -> float:
        eff = _clamp(self.config.battery_efficiency, 1e-6, 1.0)
        if battery_kw >= 0.0:
            return battery_kw * dt_hours * eff
        return battery_kw * dt_hours / eff

    def _community_battery_target_kw(
        self,
        battery_bounds: BatteryPowerBounds,
        soc: float,
        dt_hours: float,
        community_net_kw: float | None,
        reserve_floor_soc: float,
        charge_target_soc: float,
        price_regime: str,
    ) -> CommunityTargetDecision:
        cfg = self.config
        if not cfg.community_participation_enabled or community_net_kw is None:
            return CommunityTargetDecision(
                raw_target_kw=0.0,
                effective_target_kw=0.0,
                reserve_limited=False,
                soc_recovery_target_kw=0.0,
            )

        deadband_kw = max(cfg.community_energy_deadband_kw, 0.0)
        raw_target_kw = 0.0
        if community_net_kw > deadband_kw + 1e-9:
            community_deficit_kw = community_net_kw - deadband_kw
            raw_target_kw = -community_deficit_kw * max(cfg.community_deficit_weight, 0.0)
        elif community_net_kw < -(deadband_kw + 1e-9):
            community_surplus_kw = (-community_net_kw) - deadband_kw
            raw_target_kw = community_surplus_kw * max(cfg.community_surplus_weight, 0.0)

        cap_kw = max(cfg.community_dispatch_cap_kw, 0.0)
        if cap_kw > 1e-9:
            raw_target_kw = _clamp(raw_target_kw, -cap_kw, cap_kw)

        raw_target_kw = _clamp(raw_target_kw, battery_bounds.min_kw, battery_bounds.max_kw)
        effective_target_kw = raw_target_kw
        reserve_limited = False

        battery_capacity_kwh = max(cfg.battery_capacity_kwh, 1e-6)
        battery_efficiency = _clamp(cfg.battery_efficiency, 1e-6, 1.0)
        interval_hours = max(dt_hours, 1e-9)

        reserve_floor_soc = _clamp(reserve_floor_soc, cfg.battery_soc_min, cfg.battery_soc_max)
        charge_target_soc = _clamp(charge_target_soc, reserve_floor_soc, cfg.battery_soc_max)
        soc_recovery_target_kw = 0.0

        if raw_target_kw < -1e-9:
            reserve_discharge_room_kwh = max(soc - reserve_floor_soc, 0.0) * battery_capacity_kwh
            reserve_discharge_limit_kw = (
                reserve_discharge_room_kwh * battery_efficiency / interval_hours
            )
            allowed_discharge_kw = min(
                max(-battery_bounds.min_kw, 0.0),
                max(reserve_discharge_limit_kw, 0.0),
            )
            effective_target_kw = -min(abs(raw_target_kw), allowed_discharge_kw)
            reserve_limited = abs(effective_target_kw) + 1e-9 < abs(raw_target_kw)
        elif raw_target_kw > 1e-9:
            reserve_charge_room_kwh = max(charge_target_soc - soc, 0.0) * battery_capacity_kwh
            reserve_charge_limit_kw = (
                reserve_charge_room_kwh / max(interval_hours * battery_efficiency, 1e-9)
            )
            allowed_charge_kw = min(
                max(battery_bounds.max_kw, 0.0),
                max(reserve_charge_limit_kw, 0.0),
            )
            effective_target_kw = min(raw_target_kw, allowed_charge_kw)

        if price_regime == "cheap" and charge_target_soc > soc + 1e-9:
            recovery_charge_room_kwh = max(charge_target_soc - soc, 0.0) * battery_capacity_kwh
            recovery_charge_limit_kw = (
                recovery_charge_room_kwh / max(interval_hours * battery_efficiency, 1e-9)
            )
            soc_recovery_target_kw = min(
                max(battery_bounds.max_kw, 0.0),
                max(recovery_charge_limit_kw, 0.0),
            )
            effective_target_kw = max(effective_target_kw, soc_recovery_target_kw)

        effective_target_kw = _clamp(
            effective_target_kw,
            battery_bounds.min_kw,
            battery_bounds.max_kw,
        )
        return CommunityTargetDecision(
            raw_target_kw=raw_target_kw,
            effective_target_kw=effective_target_kw,
            reserve_limited=reserve_limited,
            soc_recovery_target_kw=soc_recovery_target_kw,
        )

    def _community_alignment_penalty(
        self,
        battery_kw: float,
        community_target_kw: float,
        dt_hours: float,
    ) -> float:
        if not self.config.community_participation_enabled:
            return 0.0
        weight = max(self.config.community_dispatch_weight, 0.0)
        if weight <= 1e-9:
            return 0.0
        alignment_error_kw = battery_kw - community_target_kw
        return weight * (alignment_error_kw ** 2) * dt_hours

    def _joint_dispatch_objective(
        self,
        ev_kw: float,
        ev_hard_min_kw: float,
        base_without_battery: float,
        battery_kw: float,
        price_now: float,
        future_avg_price: float,
        grid_import_limit_kw: float,
        grid_export_limit_kw: float,
        dt_hours: float,
        community_target_kw: float,
        forecast_target_kw: float | None,
    ) -> float:
        cfg = self.config
        net_grid_kw = base_without_battery + battery_kw
        imported_kw = max(net_grid_kw, 0.0)
        exported_kw = max(-net_grid_kw, 0.0)

        grid_cost = (
            imported_kw * price_now * dt_hours
            - exported_kw * price_now * cfg.export_price_factor * dt_hours
        )

        battery_delta_kwh = self._battery_energy_delta_kwh(battery_kw, dt_hours)
        battery_wear_cost = abs(battery_delta_kwh) * cfg.battery_degradation_penalty_eur_per_kwh
        battery_future_value = -battery_delta_kwh * future_avg_price

        flex_ev_energy_kwh = max(ev_kw - ev_hard_min_kw, 0.0) * dt_hours
        ev_shift_value = -flex_ev_energy_kwh * future_avg_price

        exported_from_battery_penalty = 0.0
        if battery_kw < -1e-9 and net_grid_kw < -1e-9:
            exported_from_battery_kw = min(abs(battery_kw), exported_kw)
            exported_from_battery_penalty = exported_from_battery_kw * dt_hours * max(price_now, 0.0)

        import_violation = max(net_grid_kw - grid_import_limit_kw, 0.0)
        export_violation = max(-grid_export_limit_kw - net_grid_kw, 0.0)
        hard_violation_penalty = (import_violation + export_violation) * 1000.0 * dt_hours
        community_penalty = self._community_alignment_penalty(
            battery_kw=battery_kw,
            community_target_kw=community_target_kw,
            dt_hours=dt_hours,
        )
        forecast_penalty = 0.0
        if forecast_target_kw is not None:
            forecast_error_kw = battery_kw - forecast_target_kw
            forecast_penalty = max(cfg.forecast_dispatch_weight, 0.0) * (forecast_error_kw ** 2) * dt_hours

        return (
            grid_cost
            + battery_wear_cost
            + battery_future_value
            + ev_shift_value
            + exported_from_battery_penalty
            + hard_violation_penalty
            + community_penalty
            + forecast_penalty
        )

    def _optimize_joint_dispatch(
        self,
        payload: Dict[str, Any],
        ev: EvAggregate,
        battery_soc: float,
        battery_bounds: BatteryPowerBounds,
        non_shiftable_load: float,
        solar_generation: float,
        price_now: float,
        future_avg_price: float,
        grid_import_limit_kw: float,
        grid_export_limit_kw: float,
        dt_hours: float,
        community_target_kw: float,
        forecast_target_kw: float | None,
    ) -> tuple[float, float, float]:
        ev_min_kw = _clamp(ev.hard_min_kw, 0.0, ev.max_kw)
        if not ev.connected:
            ev_values = [0.0]
        else:
            ev_values = self._candidate_values(ev_min_kw, ev.max_kw)

        battery_values = self._candidate_values(battery_bounds.min_kw, battery_bounds.max_kw)

        best_choice: tuple[float, float, float, float, float] | None = None
        # tuple layout:
        # (objective, ev_kw, battery_kw, base_without_battery, net_grid_kw)

        for ev_kw in ev_values:
            base_without_battery = self._estimate_base_without_battery(
                payload=payload,
                ev_kw=ev_kw,
                non_shiftable_load=non_shiftable_load,
                solar_generation=solar_generation,
            )
            for battery_kw in battery_values:
                objective = self._joint_dispatch_objective(
                    ev_kw=ev_kw,
                    ev_hard_min_kw=ev_min_kw,
                    base_without_battery=base_without_battery,
                    battery_kw=battery_kw,
                    price_now=price_now,
                    future_avg_price=future_avg_price,
                    grid_import_limit_kw=grid_import_limit_kw,
                    grid_export_limit_kw=grid_export_limit_kw,
                    dt_hours=dt_hours,
                    community_target_kw=community_target_kw,
                    forecast_target_kw=forecast_target_kw,
                )
                net_grid_kw = base_without_battery + battery_kw
                candidate = (objective, ev_kw, battery_kw, base_without_battery, net_grid_kw)
                if best_choice is None:
                    best_choice = candidate
                    continue
                if objective < best_choice[0] - 1e-9:
                    best_choice = candidate
                    continue
                if abs(objective - best_choice[0]) <= 1e-9:
                    # Tie-breakers: prefer lower export first, then lower import.
                    exported = max(-net_grid_kw, 0.0)
                    best_exported = max(-best_choice[4], 0.0)
                    if exported < best_exported - 1e-9:
                        best_choice = candidate
                        continue
                    if abs(exported - best_exported) <= 1e-9:
                        imported = max(net_grid_kw, 0.0)
                        best_imported = max(best_choice[4], 0.0)
                        if imported < best_imported - 1e-9:
                            best_choice = candidate

        if best_choice is None:
            fallback_ev_kw = ev_min_kw
            fallback_base = self._estimate_base_without_battery(
                payload=payload,
                ev_kw=fallback_ev_kw,
                non_shiftable_load=non_shiftable_load,
                solar_generation=solar_generation,
            )
            fallback_battery_kw = self._battery_dispatch(
                base_without_battery=fallback_base,
                price_now=price_now,
                future_avg_price=future_avg_price,
                bounds=battery_bounds,
            )
            return fallback_ev_kw, fallback_battery_kw, fallback_base

        return best_choice[1], best_choice[2], best_choice[3]

    def _optimize_battery_dispatch_for_fixed_ev(
        self,
        payload: Dict[str, Any],
        ev_kw: float,
        ev_hard_min_kw: float,
        battery_bounds: BatteryPowerBounds,
        non_shiftable_load: float,
        solar_generation: float,
        price_now: float,
        future_avg_price: float,
        grid_import_limit_kw: float,
        grid_export_limit_kw: float,
        dt_hours: float,
        community_target_kw: float,
        forecast_target_kw: float | None,
    ) -> tuple[float, float]:
        base_without_battery = self._estimate_base_without_battery(
            payload=payload,
            ev_kw=ev_kw,
            non_shiftable_load=non_shiftable_load,
            solar_generation=solar_generation,
        )
        battery_values = self._candidate_values(battery_bounds.min_kw, battery_bounds.max_kw)
        if not battery_values:
            return 0.0, base_without_battery

        best_choice: tuple[float, float, float] | None = None
        # tuple: (objective, battery_kw, net_grid_kw)
        for battery_kw in battery_values:
            objective = self._joint_dispatch_objective(
                ev_kw=ev_kw,
                ev_hard_min_kw=ev_hard_min_kw,
                base_without_battery=base_without_battery,
                battery_kw=battery_kw,
                price_now=price_now,
                future_avg_price=future_avg_price,
                grid_import_limit_kw=grid_import_limit_kw,
                grid_export_limit_kw=grid_export_limit_kw,
                dt_hours=dt_hours,
                community_target_kw=community_target_kw,
                forecast_target_kw=forecast_target_kw,
            )
            net_grid_kw = base_without_battery + battery_kw
            candidate = (objective, battery_kw, net_grid_kw)
            if best_choice is None:
                best_choice = candidate
                continue
            if objective < best_choice[0] - 1e-9:
                best_choice = candidate
                continue
            if abs(objective - best_choice[0]) <= 1e-9:
                exported = max(-net_grid_kw, 0.0)
                best_exported = max(-best_choice[2], 0.0)
                if exported < best_exported - 1e-9:
                    best_choice = candidate
                    continue
                if abs(exported - best_exported) <= 1e-9:
                    imported = max(net_grid_kw, 0.0)
                    best_imported = max(best_choice[2], 0.0)
                    if imported < best_imported - 1e-9:
                        best_choice = candidate

        if best_choice is None:
            fallback_battery_kw = self._battery_dispatch(
                base_without_battery=base_without_battery,
                price_now=price_now,
                future_avg_price=future_avg_price,
                bounds=battery_bounds,
            )
            return fallback_battery_kw, base_without_battery

        return best_choice[1], base_without_battery

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
