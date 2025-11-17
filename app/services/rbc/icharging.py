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
    if not text or text.lower() in {"nan", "none"}:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _current_timestamp(payload: Dict[str, Any]) -> datetime:
    candidates = [
        payload.get("timestamp"),
        payload.get("timestamp.$date"),
    ]
    for candidate in candidates:
        parsed = _parse_datetime(candidate)
        if parsed:
            return parsed.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _line_chargers(cfg: "IchargingRuntimeConfig | BreakerOnlyConfig", line_name: str) -> List[str]:
    info = cfg.line_limits.get(line_name, {})
    preconfigured = info.get("chargers")
    if preconfigured:
        return [str(c) for c in preconfigured]
    derived = []
    for cid, meta in cfg.chargers.items():
        if meta.get("line") == line_name:
            derived.append(str(cid))
    return derived


def _reduce_actions(
    order: List[str],
    amount: float,
    actions: Dict[str, float],
    min_levels: Dict[str, float],
) -> None:
    remaining = amount
    while remaining > 1e-6:
        adjustable = [
            cid for cid in order if actions.get(cid, 0.0) - min_levels.get(cid, 0.0) > 1e-6
        ]
        if not adjustable:
            break
        share = remaining / len(adjustable)
        for cid in adjustable:
            current = actions.get(cid, 0.0)
            min_level = min_levels.get(cid, 0.0)
            reducible = max(current - min_level, 0.0)
            reduction = min(reducible, share, remaining)
            if reduction <= 0:
                continue
            actions[cid] = current - reduction
            remaining -= reduction
            if remaining <= 1e-6:
                break


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
    solar_generation_key: str = "solar_generation"
    min_connected_kw: float = 1.6
    vehicle_capacities: Dict[str, float] = field(default_factory=dict)
    flexible_ev_ids: Optional[List[str]] = None
    allow_departure_fallback: bool = False
    flexibility_fields: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FLEX_FIELDS))
    inactive_power_threshold_kw: float = 0.5

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IchargingRuntimeConfig":
        data = dict(payload)
        vehicle_capacities = {
            str(key): _safe_float(value, 0.0)
            for key, value in (data.pop("vehicle_capacities", {}) or {}).items()
            if _safe_float(value, 0.0) > 0.0
        }
        flexible_ev_ids_raw = data.pop("flexible_ev_ids", None)
        flexible_ev_ids = (
            [str(item) for item in flexible_ev_ids_raw if str(item)] if flexible_ev_ids_raw else None
        )
        flexibility_fields = data.pop("flexibility_fields", None)
        inactive_value = data.pop("inactive_power_threshold_kw", None)
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
            solar_generation_key=data.pop("solar_generation_key", "solar_generation"),
            min_connected_kw=_safe_float(data.pop("min_connected_kw", 1.6)),
            vehicle_capacities=vehicle_capacities or {},
            flexible_ev_ids=flexible_ev_ids,
            allow_departure_fallback=bool(data.pop("allow_departure_fallback", False)),
            flexibility_fields=dict(flexibility_fields or DEFAULT_FLEX_FIELDS),
            inactive_power_threshold_kw=_safe_float(inactive_value, 0.0)
            if inactive_value is not None
            else 0.0,
        )
        if inactive_value is None:
            cfg.inactive_power_threshold_kw = max(cfg.min_connected_kw - 0.2, 0.0)
        return cfg

    def resolve_field(self, key: str, ev_id: str) -> str:
        template = self.flexibility_fields.get(key)
        return template.replace("{ev_id}", ev_id) if template else ""


@dataclass
class ChargerState:
    id: str
    min_kw: float
    max_kw: float
    line: Optional[str]
    ev_id: str
    connected: bool
    allow_flex: bool
    session_power: float
    flexible: bool = False
    required_kw: float = 0.0
    priority: float = 0.0
    is_active_nonflex: bool = False


class IchargingBreakerRuntime:
    def __init__(self, config: IchargingRuntimeConfig):
        self.config = config

    def allocate(self, payload: Dict[str, Any]) -> Dict[str, float]:
        cfg = self.config
        control_minutes = max(cfg.control_interval_minutes, 1.0)
        min_minutes = control_minutes
        solar_kw = max(0.0, _safe_float(payload.get(cfg.solar_generation_key), 0.0))
        effective_board_limit = cfg.max_board_kw

        line_map: Dict[str, str] = {}
        for line_name in cfg.line_limits.keys():
            for cid in _line_chargers(cfg, line_name):
                line_map[cid] = line_name

        actions: Dict[str, float] = {}
        min_levels: Dict[str, float] = {}
        max_levels: Dict[str, float] = {}
        states: Dict[str, ChargerState] = {}
        flexible_chargers: List[str] = []
        charger_priority: Dict[str, float] = {}
        nonflex_by_line: Dict[str, List[str]] = {}

        now = self._current_timestamp(payload)
        flexible_whitelist = (
            {str(ev_id) for ev_id in cfg.flexible_ev_ids} if cfg.flexible_ev_ids else None
        )

        for charger_id, meta in cfg.chargers.items():
            base_min_kw = max(_safe_float(meta.get("min_kw", 0.0), 0.0), 0.0)
            max_kw = max(
                _safe_float(meta.get("max_kw", cfg.charger_limit_kw), cfg.charger_limit_kw), base_min_kw
            )
            line = meta.get("line") or line_map.get(charger_id)
            ev_raw = payload.get(f"charging_sessions.{charger_id}.electric_vehicle")
            ev_id = str(ev_raw).strip() if ev_raw is not None else ""
            session_power = _safe_float(payload.get(f"charging_sessions.{charger_id}.power"), 0.0)
            session_power = _clamp(session_power, 0.0, max_kw)
            connected = bool(ev_id)
            if connected and session_power < cfg.inactive_power_threshold_kw:
                connected = False
                ev_id = ""
            min_kw = base_min_kw
            if connected:
                min_kw = max(min_kw, cfg.min_connected_kw)
            state = ChargerState(
                id=charger_id,
                min_kw=min_kw,
                max_kw=max_kw,
                line=line,
                ev_id=ev_id,
                connected=connected,
                allow_flex=bool(meta.get("allow_flex_when_ev", True)),
                session_power=session_power,
            )
            states[charger_id] = state
            min_levels[charger_id] = min_kw
            max_levels[charger_id] = max_kw
            actions[charger_id] = min_kw
            charger_priority[charger_id] = 0.0

            if not state.connected:
                continue

            if state.allow_flex:
                if self._populate_flexible_state(
                    cfg,
                    payload,
                    state,
                    now,
                    control_minutes,
                    min_minutes,
                    flexible_whitelist,
                ):
                    actions[charger_id] = state.required_kw
                    min_levels[charger_id] = state.required_kw
                    charger_priority[charger_id] = state.priority
                    flexible_chargers.append(charger_id)
                    continue

            state.is_active_nonflex = state.connected
            if state.line and state.is_active_nonflex:
                nonflex_by_line.setdefault(state.line, []).append(charger_id)

        self._distribute_nonflex(cfg, states, nonflex_by_line, actions)
        self._apply_solar_bonus(states, flexible_chargers, solar_kw, actions, max_levels)
        self._enforce_line_limits(cfg, states, actions, min_levels, flexible_chargers)

        board_total = sum(actions.values())
        if board_total - effective_board_limit > 1e-6:
            order = self._ordered_chargers(states, flexible_chargers)
            _reduce_actions(order, board_total - effective_board_limit, actions, min_levels)
            self._enforce_line_limits(cfg, states, actions, min_levels, flexible_chargers)

        return {cid: float(value) for cid, value in actions.items()}

    def _populate_flexible_state(
        self,
        cfg: IchargingRuntimeConfig,
        payload: Dict[str, Any],
        state: ChargerState,
        now: datetime,
        control_minutes: float,
        min_minutes: float,
        whitelist: Optional[Set[str]],
    ) -> bool:
        if not state.allow_flex or not state.connected:
            return False
        if whitelist is not None and state.ev_id not in whitelist:
            return False

        soc_key = cfg.resolve_field("soc", state.ev_id)
        target_key = cfg.resolve_field("target_soc", state.ev_id)
        departure_key = cfg.resolve_field("departure_time", state.ev_id)
        soc = _maybe_float(payload.get(soc_key))
        target_soc = _maybe_float(payload.get(target_key))
        if soc is None or target_soc is None or target_soc <= 0:
            return False
        soc = _clamp(soc, 0.0, 1.0)
        target_soc = _clamp(target_soc, soc, 1.0)
        if target_soc <= soc + 1e-6:
            return False

        capacity = cfg.vehicle_capacities.get(state.ev_id, cfg.default_capacity_kwh)
        energy_gap = max(target_soc - soc, 0.0) * capacity
        if energy_gap <= 1e-6:
            return False

        departure_time = _parse_datetime(payload.get(departure_key))
        if not departure_time:
            if not cfg.allow_departure_fallback:
                return False
            departure_time = now + timedelta(minutes=cfg.default_departure_buffer_minutes)
        else:
            departure_time = departure_time.astimezone(timezone.utc)

        minutes_remaining = max((departure_time - now).total_seconds() / 60.0, min_minutes)
        if minutes_remaining <= control_minutes:
            required_kw = state.max_kw
        else:
            required_kw = energy_gap / (minutes_remaining / 60.0)

        required_kw = _clamp(required_kw, state.min_kw, state.max_kw)
        state.flexible = True
        state.required_kw = required_kw
        state.priority = required_kw / state.max_kw if state.max_kw > 0 else 1.0
        return True

    def _distribute_nonflex(
        self,
        cfg: IchargingRuntimeConfig,
        states: Dict[str, ChargerState],
        nonflex_by_line: Dict[str, List[str]],
        actions: Dict[str, float],
    ) -> None:
        for line_name, charger_ids in nonflex_by_line.items():
            line_cfg = cfg.line_limits.get(line_name, {})
            limit = _safe_float(line_cfg.get("limit_kw"), cfg.max_board_kw)
            line_chargers = _line_chargers(cfg, line_name)
            current = sum(actions.get(cid, 0.0) for cid in line_chargers)
            remaining = max(limit - current, 0.0)
            alloc_queue = [cid for cid in charger_ids if states[cid].max_kw > 1e-6]
            assigned = {cid: 0.0 for cid in alloc_queue}
            while remaining > 1e-6 and alloc_queue:
                share = remaining / len(alloc_queue)
                next_queue: List[str] = []
                for cid in alloc_queue:
                    state = states[cid]
                    capacity = state.max_kw
                    addition = min(share, capacity)
                    assigned[cid] += addition
                    remaining -= addition
                    if addition + 1e-6 < capacity:
                        next_queue.append(cid)
                if next_queue == alloc_queue:
                    break
                alloc_queue = next_queue
            for cid, value in assigned.items():
                actions[cid] = max(actions.get(cid, 0.0), value)

    def _apply_solar_bonus(
        self,
        states: Dict[str, ChargerState],
        flexible_chargers: List[str],
        solar_kw: float,
        actions: Dict[str, float],
        max_levels: Dict[str, float],
    ) -> None:
        if solar_kw <= 1e-6 or not flexible_chargers:
            return
        remaining = solar_kw
        ordered = sorted(flexible_chargers, key=lambda cid: states[cid].priority)
        for cid in ordered:
            headroom = max_levels[cid] - actions.get(cid, 0.0)
            if headroom <= 1e-6:
                continue
            addition = min(headroom, remaining)
            if addition <= 1e-6:
                break
            actions[cid] = actions.get(cid, 0.0) + addition
            remaining -= addition
            if remaining <= 1e-6:
                break

    def _enforce_line_limits(
        self,
        cfg: IchargingRuntimeConfig,
        states: Dict[str, ChargerState],
        actions: Dict[str, float],
        min_levels: Dict[str, float],
        flexible_chargers: List[str],
    ) -> None:
        for line_name, info in cfg.line_limits.items():
            limit = _safe_float(info.get("limit_kw"), cfg.max_board_kw)
            chargers = _line_chargers(cfg, line_name)
            total = sum(actions.get(cid, 0.0) for cid in chargers)
            overflow = total - limit
            if overflow > 1e-6:
                order = self._ordered_chargers(
                    states,
                    [cid for cid in flexible_chargers if states[cid].line == line_name],
                    extra=[cid for cid in chargers if cid in states and not states[cid].flexible],
                )
                _reduce_actions(order, overflow, actions, min_levels)

    def _ordered_chargers(
        self,
        states: Dict[str, ChargerState],
        flex_subset: Optional[List[str]] = None,
        extra: Optional[List[str]] = None,
    ) -> List[str]:
        order: List[str] = []
        flex_ids = flex_subset if flex_subset is not None else [
            cid for cid, st in states.items() if st.flexible
        ]
        order.extend(sorted(flex_ids, key=lambda cid: states[cid].priority))
        extra_ids = extra if extra is not None else [
            cid for cid, st in states.items() if not st.flexible
        ]
        order.extend(extra_ids)
        seen: Set[str] = set()
        filtered: List[str] = []
        for cid in order:
            if cid not in seen:
                seen.add(cid)
                filtered.append(cid)
        return filtered

    def _current_timestamp(self, payload: Dict[str, Any]) -> datetime:
        return _current_timestamp(payload)


@dataclass
class BreakerOnlyConfig:
    max_board_kw: float
    charger_limit_kw: float
    chargers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    line_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    min_connected_kw: float = 1.6

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BreakerOnlyConfig":
        data = dict(payload)
        return cls(
            max_board_kw=_safe_float(data.pop("max_board_kw", 33.0)),
            charger_limit_kw=_safe_float(data.pop("charger_limit_kw", 4.6)),
            chargers=dict(data.pop("chargers", {})),
            line_limits=dict(data.pop("line_limits", {})),
            min_connected_kw=_safe_float(data.pop("min_connected_kw", 1.6)),
        )


class BreakerOnlyRuntime:
    def __init__(self, config: BreakerOnlyConfig):
        self.config = config

    def allocate(self, payload: Dict[str, Any]) -> Dict[str, float]:
        cfg = self.config
        actions: Dict[str, float] = {}
        min_levels: Dict[str, float] = {}
        max_levels: Dict[str, float] = {}
        states: Dict[str, ChargerState] = {}

        line_map: Dict[str, str] = {}
        for line_name in cfg.line_limits.keys():
            for cid in _line_chargers(cfg, line_name):
                line_map[cid] = line_name

        for charger_id, meta in cfg.chargers.items():
            base_min_kw = max(_safe_float(meta.get("min_kw", 0.0), 0.0), 0.0)
            max_kw = max(
                _safe_float(meta.get("max_kw", cfg.charger_limit_kw), cfg.charger_limit_kw), base_min_kw
            )
            line = meta.get("line") or line_map.get(charger_id)
            ev_raw = payload.get(f"charging_sessions.{charger_id}.electric_vehicle")
            ev_id = str(ev_raw).strip() if ev_raw is not None else ""
            session_power = _safe_float(payload.get(f"charging_sessions.{charger_id}.power"), 0.0)
            session_power = _clamp(session_power, 0.0, max_kw)
            connected = bool(ev_id) and session_power >= cfg.min_connected_kw
            min_kw = base_min_kw
            if connected:
                min_kw = max(min_kw, cfg.min_connected_kw)
            state = ChargerState(
                id=charger_id,
                min_kw=min_kw,
                max_kw=max_kw,
                line=line,
                ev_id=ev_id,
                connected=connected,
                allow_flex=False,
                session_power=session_power,
            )
            states[charger_id] = state
            min_levels[charger_id] = min_kw
            max_levels[charger_id] = max_kw
            actions[charger_id] = min_kw

        for line_name, info in cfg.line_limits.items():
            limit = _safe_float(info.get("limit_kw"), cfg.max_board_kw)
            chargers = _line_chargers(cfg, line_name)
            active = [cid for cid in chargers if states.get(cid) and states[cid].connected]
            if not active:
                continue
            share = limit / len(active)
            for cid in active:
                target = min(max_levels[cid], max(share, min_levels[cid]))
                actions[cid] = target

        board_total = sum(actions.values())
        if board_total - cfg.max_board_kw > 1e-6:
            order = [cid for cid, state in states.items() if state.connected]
            if not order:
                order = list(actions.keys())
            _reduce_actions(order, board_total - cfg.max_board_kw, actions, min_levels)

        for cid, value in actions.items():
            actions[cid] = _clamp(value, min_levels[cid], max_levels[cid])
        return {cid: float(val) for cid, val in actions.items()}

    def _current_timestamp(self, payload: Dict[str, Any]) -> datetime:
        return _current_timestamp(payload)
