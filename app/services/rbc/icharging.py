from __future__ import annotations

from dataclasses import dataclass, field
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from app.logging import get_logger


DEFAULT_FLEX_FIELDS = {
    "soc": "electric_vehicles.{ev_id}.SoC",
    "target_soc": "electric_vehicles.{ev_id}.flexibility.estimated_soc_at_departure",
    "departure_time": "electric_vehicles.{ev_id}.flexibility.estimated_time_at_departure",
}
MIN_CONTROL_INTERVAL_MINUTES = 1.0 / 60.0


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


def _normalize_ev_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric != numeric or abs(numeric) <= 1e-9:
            return ""
        if float(int(numeric)) == numeric:
            return str(int(numeric))
        return str(numeric)
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"none", "nan", "null"}:
        return ""
    if text in {"0", "0.0"}:
        return ""
    return text


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


def _round_down_one_decimal(value: float) -> float:
    return math.floor(value * 10.0) / 10.0


def _line_chargers(cfg: "IchargingRuntimeConfig | BreakerOnlyConfig", line_name: str) -> List[str]:
    info = cfg.line_limits.get(line_name, {})
    preconfigured = info.get("chargers")
    if preconfigured:
        return [str(c) for c in preconfigured]
    derived = []
    for cid, meta in cfg.chargers.items():
        phases = meta.get("phases")
        if phases:
            if line_name in phases:
                derived.append(str(cid))
            continue
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
    per_phase_headroom_kw: float = 0.0
    vehicle_capacities: Dict[str, float] = field(default_factory=dict)
    flexible_ev_ids: Optional[List[str]] = None
    allow_departure_fallback: bool = False
    flexibility_fields: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FLEX_FIELDS))
    inactive_power_threshold_kw: float = 0.5
    site_available_headroom_key: Optional[str] = None
    site_available_headroom_fallback_kw: float = 0.0
    site_available_headroom_includes_pv: bool = True
    site_meter_import_key: Optional[str] = None
    site_meter_export_key: Optional[str] = None
    virtual_battery_action_name: Optional[str] = None
    virtual_battery_nominal_power_kw: float = 15.0
    virtual_battery_capacity_kwh: float = 20.0
    virtual_battery_capacity_min_kwh: float = 20.0
    virtual_battery_capacity_max_kwh: float = 20.0
    virtual_battery_charge_power_max_kw: float = 15.0
    virtual_battery_discharge_power_max_kw: float = 15.0
    virtual_battery_soc_key: str = "virtual_battery.soc"
    virtual_battery_soc_fallback_key: str = "electrical_storage.soc"
    virtual_battery_setpoint_key: str = "virtual_battery.setpoint_kw"
    virtual_battery_use_setpoint: bool = False
    virtual_battery_use_community_signals: bool = True
    virtual_battery_community_target_import_key: str = "community.target_net_import_kw"
    virtual_battery_community_current_import_key: str = "community.current_net_import_kw"
    virtual_battery_community_price_prefix: str = "community.price_signal"
    virtual_battery_local_price_prefix: str = "energy_price"
    virtual_battery_soc_min: float = 0.1
    virtual_battery_soc_max: float = 1.0
    session_merge_map: Dict[str, List[str]] = field(default_factory=dict)
    session_merge_power_key: str = "power"
    session_merge_ev_key: str = "electric_vehicle"
    unmanaged_session_groups: Dict[str, List[str]] = field(default_factory=dict)
    unmanaged_session_power_key: str = "power"
    exclusive_charger_groups: List[List[str]] = field(default_factory=list)
    exclusive_group_strategy: str = "highest_power"

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
        site_available_headroom_key = data.pop("site_available_headroom_key", None)
        site_meter_import_key = data.pop("site_meter_import_key", None)
        site_meter_export_key = data.pop("site_meter_export_key", None)
        virtual_battery_action_name = data.pop("virtual_battery_action_name", None)
        session_merge_map_raw = data.pop("session_merge_map", {}) or {}
        session_merge_map: Dict[str, List[str]] = {}
        if isinstance(session_merge_map_raw, dict):
            for key, value in session_merge_map_raw.items():
                target = str(key).strip()
                if not target:
                    continue
                if isinstance(value, list):
                    sources = [str(item).strip() for item in value if str(item).strip()]
                else:
                    source = str(value).strip()
                    sources = [source] if source else []
                if sources:
                    session_merge_map[target] = sources
        unmanaged_session_groups_raw = data.pop("unmanaged_session_groups", {}) or {}
        unmanaged_session_groups: Dict[str, List[str]] = {}
        if isinstance(unmanaged_session_groups_raw, dict):
            for key, value in unmanaged_session_groups_raw.items():
                target = str(key).strip()
                if not target:
                    continue
                if isinstance(value, list):
                    sources = [str(item).strip() for item in value if str(item).strip()]
                else:
                    source = str(value).strip()
                    sources = [source] if source else []
                if sources:
                    unmanaged_session_groups[target] = sources
        exclusive_groups_raw = data.pop("exclusive_charger_groups", []) or []
        exclusive_charger_groups: List[List[str]] = []
        if isinstance(exclusive_groups_raw, list):
            for group in exclusive_groups_raw:
                if not isinstance(group, list):
                    continue
                ids = [str(item).strip() for item in group if str(item).strip()]
                if len(ids) >= 2:
                    exclusive_charger_groups.append(ids)
        exclusive_group_strategy = (
            str(data.pop("exclusive_group_strategy", "highest_power")).strip().lower()
            or "highest_power"
        )
        if exclusive_group_strategy != "highest_power":
            exclusive_group_strategy = "highest_power"
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
            per_phase_headroom_kw=_safe_float(data.pop("per_phase_headroom_kw", 0.0)),
            vehicle_capacities=vehicle_capacities or {},
            flexible_ev_ids=flexible_ev_ids,
            allow_departure_fallback=bool(data.pop("allow_departure_fallback", False)),
            flexibility_fields=dict(flexibility_fields or DEFAULT_FLEX_FIELDS),
            inactive_power_threshold_kw=_safe_float(inactive_value, 0.0)
            if inactive_value is not None
            else 0.0,
            site_available_headroom_key=str(site_available_headroom_key).strip()
            if site_available_headroom_key is not None and str(site_available_headroom_key).strip()
            else None,
            site_available_headroom_fallback_kw=max(
                _safe_float(data.pop("site_available_headroom_fallback_kw", 0.0), 0.0),
                0.0,
            ),
            site_available_headroom_includes_pv=bool(
                data.pop("site_available_headroom_includes_pv", True)
            ),
            site_meter_import_key=str(site_meter_import_key).strip()
            if site_meter_import_key is not None and str(site_meter_import_key).strip()
            else None,
            site_meter_export_key=str(site_meter_export_key).strip()
            if site_meter_export_key is not None and str(site_meter_export_key).strip()
            else None,
            virtual_battery_action_name=str(virtual_battery_action_name).strip()
            if virtual_battery_action_name is not None and str(virtual_battery_action_name).strip()
            else None,
            virtual_battery_nominal_power_kw=max(
                _safe_float(data.pop("virtual_battery_nominal_power_kw", 15.0), 15.0),
                0.0,
            ),
            virtual_battery_capacity_kwh=max(
                _safe_float(data.pop("virtual_battery_capacity_kwh", 20.0), 20.0),
                0.0,
            ),
            virtual_battery_capacity_min_kwh=max(
                _safe_float(data.pop("virtual_battery_capacity_min_kwh", 20.0), 20.0),
                0.0,
            ),
            virtual_battery_capacity_max_kwh=max(
                _safe_float(data.pop("virtual_battery_capacity_max_kwh", 20.0), 20.0),
                0.0,
            ),
            virtual_battery_charge_power_max_kw=max(
                _safe_float(data.pop("virtual_battery_charge_power_max_kw", 15.0), 15.0),
                0.0,
            ),
            virtual_battery_discharge_power_max_kw=max(
                _safe_float(data.pop("virtual_battery_discharge_power_max_kw", 15.0), 15.0),
                0.0,
            ),
            virtual_battery_soc_key=str(
                data.pop("virtual_battery_soc_key", "virtual_battery.soc")
            ),
            virtual_battery_soc_fallback_key=str(
                data.pop("virtual_battery_soc_fallback_key", "electrical_storage.soc")
            ),
            virtual_battery_setpoint_key=str(
                data.pop("virtual_battery_setpoint_key", "virtual_battery.setpoint_kw")
            ),
            virtual_battery_use_setpoint=bool(
                data.pop("virtual_battery_use_setpoint", False)
            ),
            virtual_battery_use_community_signals=bool(
                data.pop("virtual_battery_use_community_signals", True)
            ),
            virtual_battery_community_target_import_key=str(
                data.pop("virtual_battery_community_target_import_key", "community.target_net_import_kw")
            ),
            virtual_battery_community_current_import_key=str(
                data.pop("virtual_battery_community_current_import_key", "community.current_net_import_kw")
            ),
            virtual_battery_community_price_prefix=str(
                data.pop("virtual_battery_community_price_prefix", "community.price_signal")
            ),
            virtual_battery_local_price_prefix=str(
                data.pop("virtual_battery_local_price_prefix", "energy_price")
            ),
            virtual_battery_soc_min=_clamp(
                _safe_float(data.pop("virtual_battery_soc_min", 0.1), 0.1),
                0.0,
                1.0,
            ),
            virtual_battery_soc_max=_clamp(
                _safe_float(data.pop("virtual_battery_soc_max", 1.0), 1.0),
                0.0,
                1.0,
            ),
            session_merge_map=session_merge_map,
            session_merge_power_key=str(data.pop("session_merge_power_key", "power") or "power"),
            session_merge_ev_key=str(data.pop("session_merge_ev_key", "electric_vehicle") or "electric_vehicle"),
            unmanaged_session_groups=unmanaged_session_groups,
            unmanaged_session_power_key=str(
                data.pop("unmanaged_session_power_key", "power") or "power"
            ),
            exclusive_charger_groups=exclusive_charger_groups,
            exclusive_group_strategy=exclusive_group_strategy,
        )
        if cfg.virtual_battery_soc_max < cfg.virtual_battery_soc_min:
            cfg.virtual_battery_soc_max = cfg.virtual_battery_soc_min
        if cfg.virtual_battery_capacity_max_kwh < cfg.virtual_battery_capacity_min_kwh:
            cfg.virtual_battery_capacity_max_kwh = cfg.virtual_battery_capacity_min_kwh
        if cfg.virtual_battery_charge_power_max_kw <= 0.0 and cfg.virtual_battery_nominal_power_kw > 0.0:
            cfg.virtual_battery_charge_power_max_kw = cfg.virtual_battery_nominal_power_kw
        if cfg.virtual_battery_discharge_power_max_kw <= 0.0 and cfg.virtual_battery_nominal_power_kw > 0.0:
            cfg.virtual_battery_discharge_power_max_kw = cfg.virtual_battery_nominal_power_kw
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
    phases: Optional[List[str]] = None
    n_phases: int = 1
    ev_id: str = ""
    connected: bool = False
    allow_flex: bool = True
    session_power: float = 0.0
    flexible: bool = False
    required_kw: float = 0.0
    priority: float = 0.0
    is_active_nonflex: bool = False


class IchargingBreakerRuntime:
    def __init__(self, config: IchargingRuntimeConfig):
        self.config = config

    def _effective_session_state(self, payload: Dict[str, Any], charger_id: str) -> tuple[str, float]:
        cfg = self.config
        sources = cfg.session_merge_map.get(charger_id) or [charger_id]
        session_rows: List[tuple[str, str, float]] = []
        for source_id in sources:
            ev_key = f"charging_sessions.{source_id}.{cfg.session_merge_ev_key}"
            power_key = f"charging_sessions.{source_id}.{cfg.session_merge_power_key}"
            ev_id = _normalize_ev_id(payload.get(ev_key))
            power = max(_safe_float(payload.get(power_key), 0.0), 0.0)
            session_rows.append((source_id, ev_id, power))

        if not session_rows:
            return "", 0.0

        with_ev = [row for row in session_rows if row[1]]
        if with_ev:
            chosen = max(with_ev, key=lambda row: row[2])
        else:
            chosen = max(session_rows, key=lambda row: row[2])
        return chosen[1], chosen[2]

    def _unmanaged_session_load_kw(self, payload: Dict[str, Any]) -> float:
        cfg = self.config
        if not cfg.unmanaged_session_groups:
            return 0.0

        total_kw = 0.0
        for source_ids in cfg.unmanaged_session_groups.values():
            group_peak = 0.0
            for source_id in source_ids:
                power_key = f"charging_sessions.{source_id}.{cfg.unmanaged_session_power_key}"
                power_kw = max(_safe_float(payload.get(power_key), 0.0), 0.0)
                group_peak = max(group_peak, power_kw)
            total_kw += group_peak
        return total_kw

    def _apply_exclusive_groups(self, states: Dict[str, ChargerState]) -> None:
        cfg = self.config
        if not cfg.exclusive_charger_groups:
            return

        log = get_logger()
        for group in cfg.exclusive_charger_groups:
            group_ids = [cid for cid in group if cid in states]
            if len(group_ids) < 2:
                continue

            connected_ids = [cid for cid in group_ids if states[cid].connected]
            if len(connected_ids) <= 1:
                continue

            ranked = sorted(
                connected_ids,
                key=lambda cid: (
                    -states[cid].session_power,
                    0 if states[cid].ev_id else 1,
                    group_ids.index(cid),
                ),
            )
            winner = ranked[0]
            for cid in connected_ids:
                if cid == winner:
                    continue
                st = states[cid]
                st.connected = False
                st.ev_id = ""
                st.flexible = False
                st.required_kw = 0.0
                st.priority = 0.0
                st.is_active_nonflex = False

            log.warning(
                "rbc.exclusive_group_conflict",
                strategy="icharging_breaker",
                group=group_ids,
                winner=winner,
                connected_candidates=[
                    {"charger": cid, "power_kw": states[cid].session_power}
                    for cid in connected_ids
                ],
            )

    def allocate(self, payload: Dict[str, Any]) -> Dict[str, float]:
        cfg = self.config
        control_minutes = max(cfg.control_interval_minutes, MIN_CONTROL_INTERVAL_MINUTES)
        min_minutes = control_minutes
        solar_kw = max(0.0, _safe_float(payload.get(cfg.solar_generation_key), 0.0))
        unmanaged_load_kw = self._unmanaged_session_load_kw(payload)
        if cfg.site_available_headroom_key:
            headroom_raw = _maybe_float(payload.get(cfg.site_available_headroom_key))
            if headroom_raw is None or headroom_raw < 0.0:
                effective_board_limit = max(cfg.site_available_headroom_fallback_kw, 0.0)
                get_logger().warning(
                    "rbc.site_headroom_fallback",
                    strategy="icharging_breaker",
                    key=cfg.site_available_headroom_key,
                    fallback_kw=effective_board_limit,
                )
            else:
                effective_board_limit = headroom_raw
            if not cfg.site_available_headroom_includes_pv:
                effective_board_limit += solar_kw
        else:
            effective_board_limit = cfg.max_board_kw + solar_kw
            site_import = (
                max(_safe_float(payload.get(cfg.site_meter_import_key), 0.0), 0.0)
                if cfg.site_meter_import_key
                else None
            )
            site_export = (
                max(_safe_float(payload.get(cfg.site_meter_export_key), 0.0), 0.0)
                if cfg.site_meter_export_key
                else 0.0
            )
            if site_import is not None:
                net_import = max(site_import - site_export, 0.0)
                effective_board_limit -= net_import
            elif unmanaged_load_kw > 1e-6:
                effective_board_limit -= unmanaged_load_kw
        effective_board_limit = max(effective_board_limit, 0.0)
        per_phase_limit = (
            effective_board_limit / max(len(cfg.line_limits), 1)
            if cfg.line_limits
            else effective_board_limit
        )
        if cfg.line_limits and cfg.per_phase_headroom_kw > 0.0:
            per_phase_limit = max(per_phase_limit - cfg.per_phase_headroom_kw, 0.0)

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
            phases = meta.get("phases")
            n_phases = max(len(phases), 1) if phases else 1
            base_min_kw = max(_safe_float(meta.get("min_kw", 0.0), 0.0), 0.0)
            max_kw = max(
                _safe_float(meta.get("max_kw", cfg.charger_limit_kw), cfg.charger_limit_kw), base_min_kw
            )
            line = meta.get("line") or line_map.get(charger_id)
            ev_id, session_power = self._effective_session_state(payload, charger_id)
            session_power = _clamp(session_power, 0.0, max_kw)
            connected = bool(ev_id)
            min_kw = max(base_min_kw, cfg.min_connected_kw * (max(n_phases, 1) if connected else 1))
            state = ChargerState(
                id=charger_id,
                min_kw=min_kw,
                max_kw=max_kw,
                line=line,
                phases=phases if phases else ([line] if line else None),
                n_phases=n_phases,
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

        self._apply_exclusive_groups(states)

        for charger_id in cfg.chargers.keys():
            state = states[charger_id]
            min_levels[charger_id] = state.min_kw
            actions[charger_id] = state.min_kw

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
                    # `required_kw` is a soft target. Keep the technical minimum as hard floor so
                    # line/board enforcement can still reduce infeasible flex plans.
                    min_levels[charger_id] = state.min_kw
                    charger_priority[charger_id] = state.priority
                    flexible_chargers.append(charger_id)
                    continue

            state.is_active_nonflex = state.connected
            if state.phases and state.is_active_nonflex:
                for ln in state.phases:
                    nonflex_by_line.setdefault(ln, []).append(charger_id)

        self._distribute_nonflex(cfg, states, nonflex_by_line, actions, per_phase_limit)
        self._apply_solar_bonus(states, flexible_chargers, solar_kw, actions, max_levels)
        self._enforce_line_limits(
            cfg, states, actions, min_levels, flexible_chargers, per_phase_limit
        )
        self._fill_remaining_headroom(
            cfg,
            states,
            actions,
            min_levels,
            flexible_chargers,
            per_phase_limit,
            effective_board_limit,
        )

        board_total = sum(
            value for cid, value in actions.items() if states.get(cid) and states[cid].connected
        )
        if board_total - effective_board_limit > 1e-6:
            order = self._ordered_chargers(states, flexible_chargers)
            _reduce_actions(order, board_total - effective_board_limit, actions, min_levels)
            self._enforce_line_limits(
                cfg, states, actions, min_levels, flexible_chargers, per_phase_limit
            )

        flex_shortfalls = {
            cid: state.required_kw - actions.get(cid, 0.0)
            for cid in flexible_chargers
            if (state := states.get(cid)) is not None
            and state.required_kw - actions.get(cid, 0.0) > 1e-6
        }
        if flex_shortfalls:
            get_logger().warning(
                "rbc.flex_requirement_shortfall",
                strategy="icharging_breaker",
                affected=len(flex_shortfalls),
                max_shortfall_kw=round(max(flex_shortfalls.values()), 3),
            )

        quantized: Dict[str, float] = {}
        for cid, value in actions.items():
            if cid not in states:
                quantized[cid] = float(_round_down_one_decimal(value))
                continue
            q = _round_down_one_decimal(value)
            q = _clamp(q, min_levels.get(cid, 0.0), max_levels.get(cid, states[cid].max_kw))
            quantized[cid] = float(q)

        if cfg.virtual_battery_action_name:
            quantized[cfg.virtual_battery_action_name] = self._virtual_battery_dispatch(
                cfg,
                payload,
                quantized,
                states,
                effective_board_limit,
            )

        # Runtime summary for debugging
        try:
            log = get_logger()
            log_actions = quantized
            line_totals: Dict[str, float] = {}
            connected: Dict[str, str] = {}
            for cid, state in states.items():
                if not state.connected:
                    continue
                connected[cid] = state.ev_id
                phases = state.phases or ([state.line] if state.line else [])
                phases = [p for p in phases if p]
                n_phases = max(len(phases), 1)
                per_phase = log_actions.get(cid, 0.0) / n_phases
                for phase in phases or ["unknown"]:
                    line_totals[phase] = line_totals.get(phase, 0.0) + per_phase
            line_totals = dict(sorted(line_totals.items()))
            board_total = sum(log_actions.get(cid, 0.0) for cid in connected)
            flex_summary = {
                cid: {
                    "ev": states[cid].ev_id,
                    "required_kw": _round_down_one_decimal(states[cid].required_kw),
                    "priority": round(states[cid].priority, 3),
                }
                for cid in flexible_chargers
            }
            log.debug(
                "rbc.actions",
                strategy="icharging_breaker",
                actions=log_actions,
                connected=connected,
                flex=flex_summary,
                phase_totals=line_totals,
                board_total=board_total,
            )

            def fmt_kw(value: float) -> str:
                return f"{_round_down_one_decimal(value):.1f}"

            line_chargers: Dict[str, List[str]] = {}
            for cid in cfg.chargers:
                state = states.get(cid)
                if not state:
                    continue
                phases = state.phases or ([state.line] if state.line else [])
                phases = [p for p in phases if p]
                if not phases:
                    phases = ["unknown"]
                for phase in phases:
                    line_chargers.setdefault(phase, []).append(cid)

            ordered_lines = list(cfg.line_limits.keys())
            for phase in line_chargers:
                if phase not in ordered_lines:
                    ordered_lines.append(phase)

            summary_lines = [f"Board total: {fmt_kw(board_total)} kW"]
            for phase in ordered_lines:
                if phase not in line_chargers:
                    continue
                phase_total = line_totals.get(phase, 0.0)
                summary_lines.append(f"{phase} - Phase total: {fmt_kw(phase_total)} kW")
                for cid in line_chargers[phase]:
                    state = states[cid]
                    ev = state.ev_id or "-"
                    connected_flag = "yes" if state.connected else "no"
                    action_kw = log_actions.get(cid, 0.0)
                    action_text = fmt_kw(action_kw)
                    if state.n_phases > 1:
                        per_phase = action_kw / max(state.n_phases, 1)
                        action_text = f"{action_text} ({fmt_kw(per_phase)}/phase)"
                    if state.flexible:
                        flex_text = (
                            f"flex=yes req={fmt_kw(state.required_kw)}"
                            f" prio={state.priority:.2f}"
                        )
                    else:
                        flex_text = "flex=no"
                    summary_lines.append(
                        f"  {cid} - ev={ev} connected={connected_flag} action={action_text} {flex_text}"
                    )

            log.info("rbc.summary\n{}\n", "\n".join(summary_lines))
        except Exception:
            get_logger().exception("rbc.action_logging_failed")

        return quantized

    def _extract_price_series(self, payload: Dict[str, Any], prefix: str) -> List[float]:
        prefixes = [prefix]
        nested_prefix = f"energy_tariffs.OMIE.{prefix}"
        if prefix and not prefix.startswith("energy_tariffs.") and nested_prefix not in prefixes:
            prefixes.append(nested_prefix)

        for candidate in prefixes:
            values: List[tuple[int, float]] = []
            values_prefix = f"{candidate}.values["
            for key, raw in payload.items():
                if not key.startswith(values_prefix):
                    continue
                idx_text = key[len(values_prefix) : -1] if key.endswith("]") else ""
                if not idx_text.isdigit():
                    continue
                value = _maybe_float(raw)
                if value is None:
                    continue
                values.append((int(idx_text), value))
            if values:
                return [value for _, value in sorted(values, key=lambda pair: pair[0])]

            scalar = _maybe_float(payload.get(candidate))
            if scalar is not None:
                return [scalar]

            current = _maybe_float(payload.get(f"{candidate}.current"))
            if current is not None:
                return [current]
        return []

    def _price_based_virtual_battery_dispatch(
        self,
        cfg: IchargingRuntimeConfig,
        payload: Dict[str, Any],
        charge_limit_kw: float,
        discharge_limit_kw: float,
    ) -> float:
        if charge_limit_kw <= 1e-6 and discharge_limit_kw <= 1e-6:
            return 0.0

        series: List[float] = []
        if cfg.virtual_battery_use_community_signals:
            series = self._extract_price_series(payload, cfg.virtual_battery_community_price_prefix)
        if not series:
            series = self._extract_price_series(payload, cfg.virtual_battery_local_price_prefix)
        if not series:
            return 0.0

        current = series[0]
        if len(series) > 1:
            horizon = series[1 : min(len(series), 25)]
            future_avg = sum(horizon) / max(len(horizon), 1)
        else:
            future_avg = current
        delta = future_avg - current
        threshold = max(abs(current) * 0.05, 0.01)
        if delta > threshold:
            return charge_limit_kw
        if delta < -threshold:
            return -discharge_limit_kw
        return 0.0

    def _virtual_battery_dispatch(
        self,
        cfg: IchargingRuntimeConfig,
        payload: Dict[str, Any],
        charger_actions: Dict[str, float],
        states: Dict[str, ChargerState],
        effective_board_limit: float,
    ) -> float:
        charge_cap_kw = max(cfg.virtual_battery_charge_power_max_kw, 0.0)
        discharge_cap_kw = max(cfg.virtual_battery_discharge_power_max_kw, 0.0)
        if charge_cap_kw <= 1e-6 and discharge_cap_kw <= 1e-6:
            return 0.0

        soc = _maybe_float(payload.get(cfg.virtual_battery_soc_key))
        if soc is None:
            soc = _maybe_float(payload.get(cfg.virtual_battery_soc_fallback_key))
        if soc is None:
            soc = 0.5
        if soc > 1.0 and soc <= 100.0:
            soc = soc / 100.0
        soc = _clamp(soc, 0.0, 1.0)

        capacity_kwh = _clamp(
            cfg.virtual_battery_capacity_kwh,
            cfg.virtual_battery_capacity_min_kwh,
            cfg.virtual_battery_capacity_max_kwh,
        )
        if capacity_kwh <= 1e-6:
            return 0.0

        dt_hours = max(cfg.control_interval_minutes, MIN_CONTROL_INTERVAL_MINUTES) / 60.0
        charge_soc_cap = max((cfg.virtual_battery_soc_max - soc) * capacity_kwh / dt_hours, 0.0)
        discharge_soc_cap = max((soc - cfg.virtual_battery_soc_min) * capacity_kwh / dt_hours, 0.0)
        charge_limit_kw = min(charge_cap_kw, charge_soc_cap)
        discharge_limit_kw = min(discharge_cap_kw, discharge_soc_cap)

        board_total_kw = sum(
            charger_actions.get(cid, 0.0)
            for cid, state in states.items()
            if state.connected
        )
        board_headroom_kw = max(effective_board_limit - board_total_kw, 0.0)
        charge_limit_kw = min(charge_limit_kw, board_headroom_kw)

        non_shiftable_load_kw = _safe_float(payload.get("non_shiftable_load"), 0.0)
        solar_kw = max(0.0, _safe_float(payload.get(cfg.solar_generation_key), 0.0))
        local_net_without_battery_kw = non_shiftable_load_kw + board_total_kw - solar_kw

        target_dispatch_kw: Optional[float] = None
        if cfg.virtual_battery_use_community_signals:
            target_import = _maybe_float(payload.get(cfg.virtual_battery_community_target_import_key))
            current_import = _maybe_float(payload.get(cfg.virtual_battery_community_current_import_key))
            if target_import is not None and current_import is not None:
                target_dispatch_kw = -(current_import - target_import)

        price_dispatch_kw = self._price_based_virtual_battery_dispatch(
            cfg,
            payload,
            charge_limit_kw,
            discharge_limit_kw,
        )

        setpoint_dispatch_kw: Optional[float] = None
        if cfg.virtual_battery_use_setpoint:
            setpoint_kw = _maybe_float(payload.get(cfg.virtual_battery_setpoint_key))
            if setpoint_kw is not None:
                setpoint_dispatch_kw = setpoint_kw

        # Solar-first local policy:
        # 1) absorb local PV surplus in battery;
        # 2) when importing from grid, allow discharge when price indicates it is unfavorable.
        if local_net_without_battery_kw < -1e-6:
            solar_first_dispatch_kw = min(charge_limit_kw, -local_net_without_battery_kw)
        elif local_net_without_battery_kw > 1e-6 and price_dispatch_kw < -1e-6:
            solar_first_dispatch_kw = -min(discharge_limit_kw, local_net_without_battery_kw)
        else:
            solar_first_dispatch_kw = 0.0

        dispatch_kw = (
            setpoint_dispatch_kw
            if setpoint_dispatch_kw is not None
            else (target_dispatch_kw if target_dispatch_kw is not None else solar_first_dispatch_kw)
        )
        if target_dispatch_kw is not None and abs(dispatch_kw) < 0.5:
            dispatch_kw = solar_first_dispatch_kw

        if dispatch_kw > 0.0:
            dispatch_kw = min(dispatch_kw, charge_limit_kw)
        elif dispatch_kw < 0.0:
            dispatch_kw = max(dispatch_kw, -discharge_limit_kw)

        dispatch_kw = _round_down_one_decimal(dispatch_kw)
        if dispatch_kw > 0.0:
            dispatch_kw = min(dispatch_kw, charge_limit_kw)
        elif dispatch_kw < 0.0:
            dispatch_kw = max(dispatch_kw, -discharge_limit_kw)

        return float(dispatch_kw)

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
        limit_per_line: float,
    ) -> None:
        for line_name, charger_ids in nonflex_by_line.items():
            limit = limit_per_line
            line_chargers = _line_chargers(cfg, line_name)
            current = sum(
                actions.get(cid, 0.0) / max(states[cid].n_phases, 1)
                for cid in line_chargers
                if cid in states and states[cid].connected
            )
            remaining = max(limit - current, 0.0)
            alloc_queue = [cid for cid in charger_ids if states[cid].max_kw > 1e-6]
            assigned = {cid: 0.0 for cid in alloc_queue}
            while remaining > 1e-6 and alloc_queue:
                share = remaining / len(alloc_queue)
                next_queue: List[str] = []
                for cid in alloc_queue:
                    state = states[cid]
                    per_line_cap = state.max_kw / max(state.n_phases, 1)
                    capacity = per_line_cap
                    addition = min(share, capacity)
                    assigned[cid] += addition
                    remaining -= addition
                    if addition + 1e-6 < capacity:
                        next_queue.append(cid)
                if next_queue == alloc_queue:
                    break
                alloc_queue = next_queue
            for cid, value in assigned.items():
                actions[cid] = max(actions.get(cid, 0.0), value * max(states[cid].n_phases, 1))

    def _fill_remaining_headroom(
        self,
        cfg: IchargingRuntimeConfig,
        states: Dict[str, ChargerState],
        actions: Dict[str, float],
        min_levels: Dict[str, float],
        flexible_chargers: List[str],
        limit_per_line: float,
        board_limit: float,
    ) -> None:
        """
        After flexible requirements are applied, if there is remaining headroom on a line and
        board, top-up non-flex chargers (and flexible chargers up to their max) until limits
        are hit. This prevents leaving unused capacity when a flexible EV requires little power.
        """
        # Board headroom considering connected chargers only
        board_used = sum(actions.get(cid, 0.0) for cid, st in states.items() if st.connected)
        board_remaining = max(board_limit - board_used, 0.0)
        if board_remaining <= 1e-6:
            return

        # Work line by line
        for line_name in cfg.line_limits:
            limit = limit_per_line
            line_chargers = [
                cid for cid, st in states.items() if st.connected and st.line == line_name
            ]
            if not line_chargers:
                continue
            current = sum(actions.get(cid, 0.0) / max(states[cid].n_phases, 1) for cid in line_chargers)
            remaining_line = max(limit - current, 0.0)
            if remaining_line <= 1e-6:
                continue

            # Chargers eligible for top-up: connected chargers on this line with available headroom
            topup_candidates = []
            for cid in line_chargers:
                st = states[cid]
                max_allowed = st.max_kw
                already = actions.get(cid, 0.0)
                if already + 1e-6 >= max_allowed:
                    continue
                topup_candidates.append(cid)
            if not topup_candidates:
                continue

            # Distribute min(board_remaining, remaining_line) across candidates proportionally
            distributable = min(remaining_line, board_remaining)
            while distributable > 1e-6 and topup_candidates:
                share = distributable / len(topup_candidates)
                next_round: List[str] = []
                for cid in topup_candidates:
                    st = states[cid]
                    max_allowed = st.max_kw
                    already = actions.get(cid, 0.0)
                    add = min(share, max_allowed - already)
                    actions[cid] = already + add
                    distributable -= add
                    board_remaining = max(board_remaining - add, 0.0)
                    if board_remaining <= 1e-6:
                        break
                    if already + add + 1e-6 < max_allowed:
                        next_round.append(cid)
                if board_remaining <= 1e-6:
                    break
                if next_round == topup_candidates:
                    # No one could accept more
                    break
                topup_candidates = next_round


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
        limit_per_line: float,
    ) -> None:
        for line_name, info in cfg.line_limits.items():
            limit = limit_per_line
            chargers = _line_chargers(cfg, line_name)
            total = sum(
                actions.get(cid, 0.0) / max(states[cid].n_phases, 1)
                for cid in chargers
                if cid in states and states[cid].connected
            )
            overflow = total - limit
            if overflow > 1e-6:
                order = self._ordered_chargers(
                    states,
                    [cid for cid in flexible_chargers if states[cid].phases and line_name in states[cid].phases],
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
    solar_generation_key: str = "solar_generation"
    per_phase_headroom_kw: float = 0.0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BreakerOnlyConfig":
        data = dict(payload)
        return cls(
            max_board_kw=_safe_float(data.pop("max_board_kw", 33.0)),
            charger_limit_kw=_safe_float(data.pop("charger_limit_kw", 4.6)),
            chargers=dict(data.pop("chargers", {})),
            line_limits=dict(data.pop("line_limits", {})),
            min_connected_kw=_safe_float(data.pop("min_connected_kw", 1.6)),
            solar_generation_key=data.pop("solar_generation_key", "solar_generation"),
            per_phase_headroom_kw=_safe_float(data.pop("per_phase_headroom_kw", 0.0)),
        )


class BreakerOnlyRuntime:
    def __init__(self, config: BreakerOnlyConfig):
        self.config = config

    def allocate(self, payload: Dict[str, Any]) -> Dict[str, float]:
        cfg = self.config
        solar_kw = max(0.0, _safe_float(payload.get(cfg.solar_generation_key), 0.0))
        effective_board_limit = cfg.max_board_kw + solar_kw
        per_phase_limit = (
            effective_board_limit / max(len(cfg.line_limits), 1)
            if cfg.line_limits
            else effective_board_limit
        )
        if cfg.line_limits and cfg.per_phase_headroom_kw > 0.0:
            per_phase_limit = max(per_phase_limit - cfg.per_phase_headroom_kw, 0.0)
        actions: Dict[str, float] = {}
        min_levels: Dict[str, float] = {}
        max_levels: Dict[str, float] = {}
        states: Dict[str, ChargerState] = {}
        nonflex_by_line: Dict[str, List[str]] = {}

        line_map: Dict[str, str] = {}
        for line_name in cfg.line_limits.keys():
            for cid in _line_chargers(cfg, line_name):
                line_map[cid] = line_name

        for charger_id, meta in cfg.chargers.items():
            phases = meta.get("phases")
            n_phases = max(len(phases), 1) if phases else 1
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
            min_kw = max(base_min_kw, cfg.min_connected_kw * (n_phases if connected else 1))
            state = ChargerState(
                id=charger_id,
                min_kw=min_kw,
                max_kw=max_kw,
                line=line,
                phases=phases if phases else ([line] if line else None),
                n_phases=n_phases,
                ev_id=ev_id,
                connected=connected,
                allow_flex=False,
                session_power=session_power,
            )
            states[charger_id] = state
            min_levels[charger_id] = min_kw
            max_levels[charger_id] = max_kw
            actions[charger_id] = min_kw

            if not state.connected:
                continue
            if state.phases:
                for ln in state.phases:
                    nonflex_by_line.setdefault(ln, []).append(charger_id)

        self._distribute_nonflex(cfg, states, nonflex_by_line, actions, per_phase_limit)
        self._enforce_line_limits(cfg, states, actions, min_levels, per_phase_limit)

        board_total = sum(value for cid, value in actions.items() if states.get(cid) and states[cid].connected)
        if board_total - effective_board_limit > 1e-6:
            order = [cid for cid, state in states.items() if state.connected]
            if not order:
                order = list(actions.keys())
            _reduce_actions(order, board_total - effective_board_limit, actions, min_levels)

        for cid, value in actions.items():
            value = _round_down_one_decimal(value)
            actions[cid] = _clamp(value, min_levels[cid], max_levels[cid])
        quantized = {cid: float(val) for cid, val in actions.items()}

        try:
            log = get_logger()
            line_totals: Dict[str, float] = {}
            connected: Dict[str, str] = {}
            for cid, state in states.items():
                if not state.connected:
                    continue
                connected[cid] = state.ev_id
                phases = state.phases or ([state.line] if state.line else [])
                phases = [p for p in phases if p]
                n_phases = max(len(phases), 1)
                per_phase = quantized.get(cid, 0.0) / n_phases
                for phase in phases or ["unknown"]:
                    line_totals[phase] = line_totals.get(phase, 0.0) + per_phase
            line_totals = dict(sorted(line_totals.items()))
            board_total = sum(quantized.get(cid, 0.0) for cid in connected)
            log.info(
                "rbc.actions",
                strategy="breaker_only",
                actions=quantized,
                connected=connected,
                phase_totals=line_totals,
                board_total=board_total,
            )
        except Exception:
            get_logger().exception("rbc.action_logging_failed")

        return quantized

    def _distribute_nonflex(
        self,
        cfg: BreakerOnlyConfig,
        states: Dict[str, ChargerState],
        nonflex_by_line: Dict[str, List[str]],
        actions: Dict[str, float],
        limit_per_line: float,
    ) -> None:
        for line_name, charger_ids in nonflex_by_line.items():
            limit = limit_per_line
            line_chargers = _line_chargers(cfg, line_name)
            current = sum(
                actions.get(cid, 0.0) / max(states[cid].n_phases, 1)
                for cid in line_chargers
                if cid in states and states[cid].connected
            )
            remaining = max(limit - current, 0.0)
            alloc_queue = [cid for cid in charger_ids if states[cid].max_kw > 1e-6]
            assigned = {cid: 0.0 for cid in alloc_queue}
            while remaining > 1e-6 and alloc_queue:
                share = remaining / len(alloc_queue)
                next_queue: List[str] = []
                for cid in alloc_queue:
                    state = states[cid]
                    per_line_cap = state.max_kw / max(state.n_phases, 1)
                    addition = min(share, per_line_cap)
                    assigned[cid] += addition
                    remaining -= addition
                    if addition + 1e-6 < per_line_cap:
                        next_queue.append(cid)
                if next_queue == alloc_queue:
                    break
                alloc_queue = next_queue
            for cid, value in assigned.items():
                actions[cid] = max(actions.get(cid, 0.0), value * max(states[cid].n_phases, 1))

    def _enforce_line_limits(
        self,
        cfg: BreakerOnlyConfig,
        states: Dict[str, ChargerState],
        actions: Dict[str, float],
        min_levels: Dict[str, float],
        limit_per_line: float,
    ) -> None:
        for line_name in cfg.line_limits.keys():
            limit = limit_per_line
            chargers = _line_chargers(cfg, line_name)
            total = sum(
                actions.get(cid, 0.0) / max(states[cid].n_phases, 1)
                for cid in chargers
                if cid in states and states[cid].connected
            )
            overflow = total - limit
            if overflow > 1e-6:
                order = [cid for cid in chargers if cid in states and states[cid].connected]
                if not order:
                    continue
                _reduce_actions(order, overflow, actions, min_levels)

    def _current_timestamp(self, payload: Dict[str, Any]) -> datetime:
        return _current_timestamp(payload)
