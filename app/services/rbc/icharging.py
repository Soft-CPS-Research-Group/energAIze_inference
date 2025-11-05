from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set


DEFAULT_FLEX_FIELDS = {
    "soc": "electric_vehicles.{ev_id}.SoC",
    "target_soc": "electric_vehicles.{ev_id}.flexibility.estimated_soc_at_departure",
    "departure_time": "electric_vehicles.{ev_id}.flexibility.estimated_time_at_departure",
}


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric != numeric:  # NaN check
            return None
        return numeric
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw.lower() in {"nan", "none"}:
            return None
        try:
            numeric = float(raw)
        except ValueError:
            return None
        if numeric != numeric:
            return None
        return numeric
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    numeric = _maybe_float(value)
    if numeric is None:
        return default
    return numeric


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(min(value, upper), lower)


def _parse_datetime(raw: Any) -> Optional[datetime]:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or text.lower() in {"", "nan", "none"}:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


@dataclass
class IchargingRuntimeConfig:
    max_board_kw: float
    charger_limit_kw: float
    chargers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    line_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    control_interval_minutes: float = 15.0
    default_capacity_kwh: float = 75.0
    default_departure_buffer_minutes: float = 60.0
    flexible_urgency_threshold: float = 0.7
    non_shiftable_key: str = "non_shiftable_load"
    solar_generation_key: str = "solar_generation"
    vehicle_capacities: Dict[str, float] = field(default_factory=dict)
    flexible_ev_ids: Optional[List[str]] = None
    allow_departure_fallback: bool = False
    flexibility_fields: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FLEX_FIELDS))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IchargingRuntimeConfig":
        data = dict(payload)
        vehicle_capacities = {
            str(key): _safe_float(value, 0.0)
            for key, value in (data.pop("vehicle_capacities", {}) or {}).items()
            if _safe_float(value, 0.0) > 0.0
        }
        flexible_ev_ids_raw = data.pop("flexible_ev_ids", None)
        flexible_ev_ids = None
        if flexible_ev_ids_raw is not None:
            flexible_ev_ids = [str(item) for item in flexible_ev_ids_raw if str(item)]
        flexibility_fields = data.pop("flexibility_fields", None)
        cfg = cls(
            max_board_kw=_safe_float(data.pop("max_board_kw", 0.0)),
            charger_limit_kw=_safe_float(data.pop("charger_limit_kw", 0.0)),
            chargers=dict(data.pop("chargers", {})),
            line_limits=dict(data.pop("line_limits", {})),
            control_interval_minutes=_safe_float(data.pop("control_interval_minutes", 15.0)),
            default_capacity_kwh=_safe_float(data.pop("default_capacity_kwh", 75.0)),
            default_departure_buffer_minutes=_safe_float(
                data.pop("default_departure_buffer_minutes", 60.0)
            ),
            flexible_urgency_threshold=_safe_float(data.pop("flexible_urgency_threshold", 0.7)),
            non_shiftable_key=data.pop("non_shiftable_key", "non_shiftable_load"),
            solar_generation_key=data.pop("solar_generation_key", "solar_generation"),
            vehicle_capacities=vehicle_capacities,
            flexible_ev_ids=flexible_ev_ids,
            allow_departure_fallback=bool(data.pop("allow_departure_fallback", False)),
            flexibility_fields=dict(flexibility_fields or DEFAULT_FLEX_FIELDS),
        )
        return cfg

    def resolve_field(self, key: str, ev_id: str) -> str:
        template = self.flexibility_fields.get(key)
        if not template:
            return ""
        return template.replace("{ev_id}", ev_id)


class IchargingBreakerRuntime:
    def __init__(self, config: IchargingRuntimeConfig):
        self.config = config

    def allocate(self, payload: Dict[str, Any]) -> Dict[str, float]:
        cfg = self.config
        control_minutes = max(cfg.control_interval_minutes, 1.0)
        min_minutes = control_minutes

        chargers_cfg = cfg.chargers
        charger_limit = cfg.charger_limit_kw
        max_board = cfg.max_board_kw

        actions: Dict[str, float] = {}
        min_levels: Dict[str, float] = {}
        max_levels: Dict[str, float] = {}
        charger_priority: Dict[str, float] = {}
        active_chargers: List[str] = []
        planned_chargers: Set[str] = set()

        now = self._current_timestamp(payload)

        flexible_whitelist: Optional[Set[str]] = None
        if cfg.flexible_ev_ids is not None:
            flexible_whitelist = {str(ev_id) for ev_id in cfg.flexible_ev_ids}

        for charger_id, meta in chargers_cfg.items():
            min_kw = _safe_float(meta.get("min_kw", 0.0), 0.0)
            max_kw = _safe_float(meta.get("max_kw", charger_limit), charger_limit)
            min_kw = max(min_kw, 0.0)
            max_kw = max(max_kw, min_kw)

            min_levels[charger_id] = min_kw
            max_levels[charger_id] = max_kw

            ev_id_raw = payload.get(f"charging_sessions.{charger_id}.electric_vehicle")
            ev_id = str(ev_id_raw).strip() if ev_id_raw is not None else ""
            session_power = _safe_float(payload.get(f"charging_sessions.{charger_id}.power"), 0.0)
            session_power = _clamp(session_power, min_kw, max_kw)
            active = bool(meta.get("active_by_default", False)) or bool(ev_id) or session_power > 0

            baseline_power = session_power if active else min_kw
            actions[charger_id] = baseline_power
            charger_priority[charger_id] = baseline_power

            if active:
                active_chargers.append(charger_id)

            if not ev_id:
                continue

            if not meta.get("allow_flex_when_ev", True):
                continue

            if flexible_whitelist is not None and ev_id not in flexible_whitelist:
                continue

            soc_key = cfg.resolve_field("soc", ev_id)
            target_soc_key = cfg.resolve_field("target_soc", ev_id)
            departure_key = cfg.resolve_field("departure_time", ev_id)

            soc = _maybe_float(payload.get(soc_key))
            if soc is None:
                continue
            soc = _clamp(soc, 0.0, 1.0)

            target_soc = _maybe_float(payload.get(target_soc_key))
            if target_soc is None or target_soc <= 0.0:
                continue
            target_soc = _clamp(target_soc, soc, 1.0)
            if target_soc <= soc + 1e-6:
                continue

            capacity = cfg.vehicle_capacities.get(ev_id, cfg.default_capacity_kwh)
            capacity = max(capacity, 0.0)

            energy_gap = max(target_soc - soc, 0.0) * capacity
            if energy_gap <= 1e-6:
                continue

            departure_time = _parse_datetime(payload.get(departure_key))
            if not departure_time:
                if not cfg.allow_departure_fallback:
                    continue
                departure_time = now + timedelta(minutes=cfg.default_departure_buffer_minutes)
            else:
                departure_time = departure_time.astimezone(timezone.utc)

            minutes_remaining = (departure_time - now).total_seconds() / 60.0
            minutes_remaining = max(minutes_remaining, min_minutes)

            if minutes_remaining <= control_minutes:
                required_kw = max_kw
            else:
                required_kw = energy_gap / (minutes_remaining / 60.0)

            required_kw = _clamp(required_kw, min_kw, max_kw)
            min_levels[charger_id] = max(min_levels[charger_id], required_kw)
            actions[charger_id] = max(actions[charger_id], required_kw)
            charger_priority[charger_id] = max(required_kw, charger_priority[charger_id])
            planned_chargers.add(charger_id)

        primary_flex: List[str] = []
        secondary_flex: List[str] = []
        for charger_id in active_chargers:
            max_kw = max_levels.get(charger_id, charger_limit)
            priority = charger_priority.get(charger_id, 0.0)
            if charger_id not in planned_chargers:
                secondary_flex.append(charger_id)
                continue
            urgency_ratio = priority / max_kw if max_kw > 0 else 0.0
            if urgency_ratio < cfg.flexible_urgency_threshold:
                primary_flex.append(charger_id)

        flexible_order = primary_flex + secondary_flex

        base_load = _safe_float(payload.get(cfg.non_shiftable_key)) - _safe_float(
            payload.get(cfg.solar_generation_key)
        )
        base_load = max(base_load, 0.0)

        def reduce_load(target_ids: List[str], amount: float) -> None:
            remaining = amount
            if not target_ids:
                return
            adjustable = [
                cid for cid in target_ids if actions.get(cid, 0.0) - min_levels.get(cid, 0.0) > 1e-6
            ]
            if not adjustable:
                adjustable = [cid for cid in target_ids if cid in actions]
            while remaining > 1e-6 and adjustable:
                adjustable.sort(key=lambda cid: charger_priority.get(cid, 0.0))
                share = remaining / len(adjustable)
                next_adjustable: List[str] = []
                for cid in adjustable:
                    current = actions.get(cid, 0.0)
                    min_level = min_levels.get(cid, 0.0)
                    available = max(current - min_level, 0.0)
                    if available <= 1e-9:
                        continue
                    reduction = min(available, share, remaining)
                    actions[cid] = current - reduction
                    remaining -= reduction
                    if actions[cid] - min_level > 1e-6:
                        next_adjustable.append(cid)
                adjustable = next_adjustable
                if not next_adjustable:
                    break

        total_demand = base_load + sum(actions.values())
        board_overflow = total_demand - max_board
        if board_overflow > 1e-6:
            reduce_load(flexible_order, board_overflow)
            total_demand = base_load + sum(actions.values())
            residual = total_demand - max_board
            if residual > 1e-6:
                reduce_load(list(actions.keys()), residual)

        for line_name, line_cfg in cfg.line_limits.items():
            limit = _safe_float(line_cfg.get("limit_kw"), max_board)
            chargers = [str(cid) for cid in line_cfg.get("chargers", [])]
            line_total = sum(actions.get(cid, 0.0) for cid in chargers)
            overflow = line_total - limit
            if overflow > 1e-6:
                reduce_load([cid for cid in chargers if cid in actions], overflow)
                line_total = sum(actions.get(cid, 0.0) for cid in chargers)
                residual = line_total - limit
                if residual > 1e-6:
                    reduce_load(chargers, residual)

        charger_total = sum(actions.values())
        available_headroom = max_board - base_load - charger_total
        if available_headroom < -1e-6:
            reduce_load(list(actions.keys()), -available_headroom)
            charger_total = sum(actions.values())
            available_headroom = max(max_board - base_load - charger_total, 0.0)

        remaining_headroom = max(available_headroom, 0.0)
        prioritized = sorted(
            planned_chargers, key=lambda cid: charger_priority.get(cid, 0.0), reverse=True
        )
        for cid in prioritized:
            if remaining_headroom <= 1e-6:
                break
            current = actions.get(cid, 0.0)
            max_level = max_levels.get(cid, current)
            if current >= max_level - 1e-6:
                continue
            increment = min(max_level - current, remaining_headroom)
            actions[cid] = current + increment
            remaining_headroom -= increment

        for cid, value in actions.items():
            min_level = min_levels.get(cid, 0.0)
            max_level = max_levels.get(cid, value)
            bounded = _clamp(value, min_level, max_level)
            if bounded != value:
                actions[cid] = bounded

        return actions

    def _current_timestamp(self, payload: Dict[str, Any]) -> datetime:
        candidates = [
            payload.get("timestamp"),
            payload.get("timestamp.$date"),
        ]
        for candidate in candidates:
            parsed = _parse_datetime(candidate)
            if parsed:
                return parsed.astimezone(timezone.utc)
        return datetime.now(timezone.utc)
