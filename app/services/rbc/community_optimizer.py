from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping

from app.logging import get_logger
from app.state import PipelineRecord
from app.utils.flatten import flatten_payload

VALID_SOC_UNIT_MODES = {"auto", "fraction", "percent"}


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
        if not text or text.lower() in {"none", "nan", "null"}:
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


def _normalize_soc(value: float, mode: str = "auto") -> float:
    normalized = value
    unit_mode = str(mode or "auto").strip().lower()
    if unit_mode == "percent":
        normalized = value / 100.0
    elif unit_mode == "auto" and 1.0 < value <= 100.0:
        normalized = value / 100.0
    return _clamp(normalized, 0.0, 1.0)


def _round_one_decimal_towards_zero(value: float) -> float:
    scaled = value * 10.0
    if scaled >= 0.0:
        return math.floor(scaled) / 10.0
    return math.ceil(scaled) / 10.0


def _normalize_ev_id(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    if text.lower() in {"none", "nan", "null"}:
        return ""
    if text in {"0", "0.0"}:
        return ""
    return text


def _extract_vector_from_flat(flat_payload: Mapping[str, Any], prefix: str) -> List[float]:
    items: List[tuple[int, float]] = []
    marker = f"{prefix}["
    for key, raw in flat_payload.items():
        if not isinstance(key, str):
            continue
        if not key.startswith(marker) or not key.endswith("]"):
            continue
        idx_raw = key[len(marker) : -1]
        if not idx_raw.isdigit():
            continue
        numeric = _maybe_float(raw)
        if numeric is None:
            continue
        items.append((int(idx_raw), numeric))
    return [value for _, value in sorted(items, key=lambda pair: pair[0])]


def _normalize_price(value: float, unit_hint: Any = None) -> float:
    normalized = value
    unit = str(unit_hint).strip().lower()
    if "mwh" in unit:
        normalized = value / 1000.0
    elif value > 3.0:
        normalized = value / 1000.0
    return max(normalized, 0.0)


@dataclass
class BatteryAsset:
    agent_index: int
    site_key: str
    action_name: str
    current_kw: float
    low_kw: float
    high_kw: float


@dataclass
class SiteContext:
    agent_index: int
    site_key: str
    strategy: str
    config: Dict[str, Any]
    flat_payload: Dict[str, Any]
    actions: Dict[str, float]
    non_shiftable_load_kw: float
    connected_ev_kw: float
    solar_generation_kw: float
    net_without_battery_kw: float
    battery: BatteryAsset | None


class CommunityOptimizerRuntime:
    """Community-level coordinator that overlays battery dispatch across sites."""

    def allocate(
        self,
        features_payload: Dict[str, Any],
        record: PipelineRecord,
    ) -> Dict[str, Dict[str, float]]:
        if not record.loaded_agent_indices:
            raise RuntimeError("No loaded agents available for community optimization")
        sites = features_payload.get("sites")
        if not isinstance(sites, dict):
            raise KeyError("Missing required object 'features.sites' for community bundle")

        first_pipeline = record.pipelines[record.loaded_agent_indices[0]]
        manifest = first_pipeline.manifest
        top_timestamp = features_payload.get("timestamp")
        top_timestamp_date = features_payload.get("timestamp.$date")
        top_community = features_payload.get("community")

        actions_by_agent: Dict[str, Dict[str, float]] = {}
        contexts: List[SiteContext] = []

        for agent_index in record.loaded_agent_indices:
            pipeline = record.pipelines[agent_index]
            artifact_cfg = manifest.get_artifact(agent_index).config or {}
            site_key = str(artifact_cfg.get("input_site_key") or "").strip()
            if not site_key:
                raise KeyError(f"Agent {agent_index} missing required config.input_site_key")
            site_payload = sites.get(site_key)
            if not isinstance(site_payload, dict):
                raise KeyError(f"Missing required object 'features.sites.{site_key}'")

            selected = dict(site_payload)
            if isinstance(top_community, dict) and "community" not in selected:
                selected["community"] = dict(top_community)
            if "timestamp" not in selected and top_timestamp is not None:
                selected["timestamp"] = top_timestamp
            if (
                "timestamp" not in selected
                and "timestamp.$date" not in selected
                and top_timestamp_date is not None
            ):
                selected["timestamp.$date"] = top_timestamp_date

            runtime_features = self._normalize_agent_input(selected, artifact_cfg)
            flattened = flatten_payload(runtime_features)
            actions_map = pipeline.inference(flattened)
            agent_actions_raw = actions_map.get(str(agent_index), actions_map.get(agent_index, {}))
            if not isinstance(agent_actions_raw, dict):
                raise RuntimeError(f"Invalid actions payload for agent {agent_index}")
            agent_actions = {str(k): float(v) for k, v in agent_actions_raw.items()}
            actions_by_agent[str(agent_index)] = agent_actions

            context = self._build_site_context(
                manifest=manifest,
                agent_index=agent_index,
                site_key=site_key,
                artifact_cfg=artifact_cfg,
                flat_payload=flattened,
                actions=agent_actions,
            )
            contexts.append(context)

        if not contexts:
            return actions_by_agent

        community_enabled = any(
            bool((manifest.get_artifact(ctx.agent_index).config or {}).get("community_optimization_enabled"))
            for ctx in contexts
        )
        if not community_enabled:
            return actions_by_agent

        price_now, future_avg = self._resolve_community_price(features_payload, contexts)
        trade_factor = self._extract_first_factor(contexts, "community_trade_factor", 0.8)
        export_factor = self._extract_first_factor(contexts, "community_export_factor", 0.8)

        baseline_site_net: Dict[str, float] = {}
        for ctx in contexts:
            baseline_site_net[ctx.site_key] = ctx.net_without_battery_kw + (
                ctx.battery.current_kw if ctx.battery else 0.0
            )
        baseline_total_net = sum(baseline_site_net.values())
        baseline_external_cost = self._external_cost(baseline_total_net, price_now, export_factor)

        self._apply_battery_overlay(contexts, price_now, future_avg, actions_by_agent)

        final_site_net: Dict[str, float] = {}
        for ctx in contexts:
            final_battery = 0.0
            if ctx.battery:
                final_battery = _safe_float(
                    actions_by_agent[str(ctx.agent_index)].get(ctx.battery.action_name), 0.0
                )
            final_site_net[ctx.site_key] = ctx.net_without_battery_kw + final_battery

        final_total_net = sum(final_site_net.values())
        final_external_cost = self._external_cost(final_total_net, price_now, export_factor)
        trade_price = trade_factor * price_now
        settlement = self._build_internal_settlement(final_site_net, trade_price)

        get_logger().info(
            "community.optimizer.summary",
            price_now=round(price_now, 6),
            future_avg=round(future_avg, 6),
            baseline_total_net_kw=round(baseline_total_net, 4),
            final_total_net_kw=round(final_total_net, 4),
            baseline_external_cost=round(baseline_external_cost, 6),
            final_external_cost=round(final_external_cost, 6),
            trade_price=round(trade_price, 6),
            site_net_kw={k: round(v, 4) for k, v in final_site_net.items()},
            settlement=settlement,
            actions=actions_by_agent,
        )
        return actions_by_agent

    def _normalize_agent_input(self, selected: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        require_observations = bool(config.get("require_observations_envelope", False))
        if not require_observations:
            return dict(selected)

        observations = selected.get("observations")
        if not isinstance(observations, dict):
            raise KeyError("Missing required object 'observations' for this bundle")

        normalized = dict(observations)
        if "community" in selected and isinstance(selected.get("community"), dict):
            normalized["community"] = dict(selected["community"])
        top_timestamp = selected.get("timestamp")
        if "timestamp" not in normalized and top_timestamp is not None:
            normalized["timestamp"] = top_timestamp
        top_timestamp_date = selected.get("timestamp.$date")
        if (
            "timestamp" not in normalized
            and "timestamp.$date" not in normalized
            and top_timestamp_date is not None
        ):
            normalized["timestamp.$date"] = top_timestamp_date
        return normalized

    def _build_site_context(
        self,
        manifest,
        agent_index: int,
        site_key: str,
        artifact_cfg: Dict[str, Any],
        flat_payload: Dict[str, Any],
        actions: Dict[str, float],
    ) -> SiteContext:
        strategy = str(artifact_cfg.get("strategy") or "")
        non_shiftable = _safe_float(flat_payload.get("non_shiftable_load"), 0.0)
        solar = self._extract_solar_generation(flat_payload, artifact_cfg)

        if strategy in {"icharging_breaker", "breaker_allocation", "breaker_only", "icharging_breaker_v0"}:
            chargers = artifact_cfg.get("chargers", {}) or {}
            connected_ev_kw = 0.0
            for charger_id in chargers:
                cid = str(charger_id)
                if cid.startswith("b_"):
                    continue
                ev_id = _normalize_ev_id(flat_payload.get(f"charging_sessions.{cid}.electric_vehicle"))
                if not ev_id:
                    continue
                connected_ev_kw += _safe_float(actions.get(cid), 0.0)
            net_without_battery = non_shiftable + connected_ev_kw - solar
            battery = self._build_icharging_battery_asset(
                manifest=manifest,
                agent_index=agent_index,
                site_key=site_key,
                artifact_cfg=artifact_cfg,
                flat_payload=flat_payload,
                connected_ev_kw=connected_ev_kw,
                actions=actions,
            )
        elif strategy == "rh1_house_rbc_v1":
            ev_action_name = self._rh1_ev_action_name(artifact_cfg)
            connected_ev_kw = _safe_float(
                actions.get(ev_action_name, actions.get("ev_charge_kw", 0.0)),
                0.0,
            )
            net_without_battery = non_shiftable + connected_ev_kw - solar
            battery = self._build_rh1_battery_asset(
                manifest=manifest,
                agent_index=agent_index,
                site_key=site_key,
                artifact_cfg=artifact_cfg,
                flat_payload=flat_payload,
                net_without_battery_kw=net_without_battery,
                actions=actions,
            )
        else:
            connected_ev_kw = 0.0
            net_without_battery = non_shiftable - solar
            battery = None

        return SiteContext(
            agent_index=agent_index,
            site_key=site_key,
            strategy=strategy,
            config=artifact_cfg,
            flat_payload=flat_payload,
            actions=actions,
            non_shiftable_load_kw=non_shiftable,
            connected_ev_kw=connected_ev_kw,
            solar_generation_kw=solar,
            net_without_battery_kw=net_without_battery,
            battery=battery,
        )

    def _build_icharging_battery_asset(
        self,
        manifest,
        agent_index: int,
        site_key: str,
        artifact_cfg: Dict[str, Any],
        flat_payload: Dict[str, Any],
        connected_ev_kw: float,
        actions: Dict[str, float],
    ) -> BatteryAsset | None:
        action_name = str(artifact_cfg.get("virtual_battery_action_name") or "").strip()
        if not action_name or action_name not in actions:
            return None

        nominal = max(_safe_float(artifact_cfg.get("virtual_battery_nominal_power_kw"), 15.0), 0.0)
        charge_cap = max(
            _safe_float(artifact_cfg.get("virtual_battery_charge_power_max_kw"), nominal),
            0.0,
        )
        discharge_cap = max(
            _safe_float(artifact_cfg.get("virtual_battery_discharge_power_max_kw"), nominal),
            0.0,
        )
        soc_key = str(artifact_cfg.get("virtual_battery_soc_key") or "virtual_battery.soc")
        soc_fallback_key = str(
            artifact_cfg.get("virtual_battery_soc_fallback_key") or "electrical_storage.soc"
        )
        soc = _maybe_float(flat_payload.get(soc_key))
        if soc is None:
            soc = _maybe_float(flat_payload.get(soc_fallback_key))
        if soc is None:
            soc = 0.5
        soc_mode = (
            str(artifact_cfg.get("virtual_battery_soc_unit_mode") or "auto").strip().lower()
            or "auto"
        )
        if soc_mode not in VALID_SOC_UNIT_MODES:
            soc_mode = "auto"
        soc = _normalize_soc(soc, soc_mode)

        soc_min = _clamp(_safe_float(artifact_cfg.get("virtual_battery_soc_min"), 0.1), 0.0, 1.0)
        soc_max = _clamp(_safe_float(artifact_cfg.get("virtual_battery_soc_max"), 1.0), 0.0, 1.0)
        capacity = _safe_float(artifact_cfg.get("virtual_battery_capacity_kwh"), 20.0)
        cap_min = _safe_float(artifact_cfg.get("virtual_battery_capacity_min_kwh"), capacity)
        cap_max = _safe_float(artifact_cfg.get("virtual_battery_capacity_max_kwh"), capacity)
        capacity = _clamp(capacity, min(cap_min, cap_max), max(cap_min, cap_max))
        dt_hours = max(_safe_float(artifact_cfg.get("control_interval_minutes"), 15.0), 1.0 / 60.0) / 60.0

        charge_soc_cap = max((soc_max - soc) * capacity / dt_hours, 0.0)
        discharge_soc_cap = max((soc - soc_min) * capacity / dt_hours, 0.0)
        high_kw = min(charge_cap, charge_soc_cap)
        low_kw = -min(discharge_cap, discharge_soc_cap)

        solar_key = str(artifact_cfg.get("solar_generation_key") or "solar_generation")
        solar_kw = max(0.0, _safe_float(flat_payload.get(solar_key), 0.0))
        site_available_key = artifact_cfg.get("site_available_headroom_key")
        if isinstance(site_available_key, str) and site_available_key.strip():
            headroom_raw = _maybe_float(flat_payload.get(site_available_key.strip()))
            if headroom_raw is None or headroom_raw < 0.0:
                effective_board_limit = max(
                    _safe_float(artifact_cfg.get("site_available_headroom_fallback_kw"), 0.0),
                    0.0,
                )
            else:
                effective_board_limit = headroom_raw
            if not bool(artifact_cfg.get("site_available_headroom_includes_pv", True)):
                effective_board_limit += solar_kw
        else:
            effective_board_limit = max(_safe_float(artifact_cfg.get("max_board_kw"), 0.0), 0.0) + solar_kw

        board_headroom = max(effective_board_limit - connected_ev_kw, 0.0)
        high_kw = min(high_kw, board_headroom)

        bound_low, bound_high = self._action_bounds_for_name(manifest, agent_index, action_name)
        if bound_low is not None:
            low_kw = max(low_kw, bound_low)
        if bound_high is not None:
            high_kw = min(high_kw, bound_high)
        if high_kw < low_kw:
            high_kw = low_kw

        current_kw = _clamp(_safe_float(actions.get(action_name), 0.0), low_kw, high_kw)
        return BatteryAsset(
            agent_index=agent_index,
            site_key=site_key,
            action_name=action_name,
            current_kw=current_kw,
            low_kw=low_kw,
            high_kw=high_kw,
        )

    def _build_rh1_battery_asset(
        self,
        manifest,
        agent_index: int,
        site_key: str,
        artifact_cfg: Dict[str, Any],
        flat_payload: Dict[str, Any],
        net_without_battery_kw: float,
        actions: Dict[str, float],
    ) -> BatteryAsset | None:
        action_name = self._rh1_battery_action_name(artifact_cfg)
        if action_name not in actions:
            if "battery_kw" in actions:
                action_name = "battery_kw"
            else:
                return None

        soc = None
        raw_soc_keys = artifact_cfg.get("battery_soc_keys", ["electrical_storage.soc"])
        soc_keys: List[str] = []
        if isinstance(raw_soc_keys, list):
            soc_keys = [str(k) for k in raw_soc_keys if str(k)]
        elif raw_soc_keys is not None:
            soc_keys = [str(raw_soc_keys)]
        for key in soc_keys:
            numeric = _maybe_float(flat_payload.get(key))
            if numeric is not None:
                soc = numeric
                break
        if soc is None:
            for key in sorted(flat_payload.keys()):
                if not isinstance(key, str):
                    continue
                lowered = key.lower()
                if lowered.startswith("batteries.") and lowered.endswith(".soc"):
                    numeric = _maybe_float(flat_payload.get(key))
                    if numeric is not None:
                        soc = numeric
                        break
        if soc is None:
            soc = 0.5
        soc_mode = str(artifact_cfg.get("soc_unit_mode") or "auto").strip().lower()
        if soc_mode == "percent":
            soc = soc / 100.0
        elif soc_mode == "auto" and 1.0 < soc <= 100.0:
            soc = soc / 100.0
        soc = _clamp(soc, 0.0, 1.0)

        battery_nominal = max(_safe_float(artifact_cfg.get("battery_nominal_power_kw"), 0.0), 0.0)
        capacity_kwh = max(_safe_float(artifact_cfg.get("battery_capacity_kwh"), 4.0), 0.1)
        efficiency = _clamp(_safe_float(artifact_cfg.get("battery_efficiency"), 0.95), 0.1, 1.0)
        soc_min = _clamp(_safe_float(artifact_cfg.get("battery_soc_min"), 0.2), 0.0, 1.0)
        soc_max = _clamp(_safe_float(artifact_cfg.get("battery_soc_max"), 1.0), 0.0, 1.0)
        dt_hours = max(_safe_float(artifact_cfg.get("control_interval_minutes"), 1.0), 1.0 / 60.0) / 60.0

        charge_room_kwh = max(soc_max - soc, 0.0) * capacity_kwh
        discharge_room_kwh = max(soc - soc_min, 0.0) * capacity_kwh
        max_charge_soc_kw = charge_room_kwh / max(dt_hours * efficiency, 1e-6)
        max_discharge_soc_kw = discharge_room_kwh * efficiency / max(dt_hours, 1e-6)
        high_kw = min(battery_nominal, max_charge_soc_kw)
        low_kw = -min(battery_nominal, max_discharge_soc_kw)

        grid_import = _safe_float(flat_payload.get("grid.import_limit_kw"), _safe_float(artifact_cfg.get("grid_import_limit_kw"), 0.0))
        if grid_import <= 0.0:
            grid_import = _safe_float(artifact_cfg.get("grid_import_limit_kw"), 0.0)
        grid_export = _safe_float(flat_payload.get("grid.export_limit_kw"), grid_import)
        grid_export = max(grid_export, 0.0)
        high_kw = min(high_kw, grid_import - net_without_battery_kw)
        low_kw = max(low_kw, -grid_export - net_without_battery_kw)

        bound_low, bound_high = self._action_bounds_for_name(manifest, agent_index, action_name)
        if bound_low is not None:
            low_kw = max(low_kw, bound_low)
        if bound_high is not None:
            high_kw = min(high_kw, bound_high)
        if high_kw < low_kw:
            high_kw = low_kw

        current_kw = _clamp(_safe_float(actions.get(action_name), 0.0), low_kw, high_kw)
        return BatteryAsset(
            agent_index=agent_index,
            site_key=site_key,
            action_name=action_name,
            current_kw=current_kw,
            low_kw=low_kw,
            high_kw=high_kw,
        )

    def _rh1_ev_action_name(self, artifact_cfg: Dict[str, Any]) -> str:
        explicit = str(artifact_cfg.get("ev_action_name") or "").strip()
        if explicit:
            return explicit

        chargers_raw = artifact_cfg.get("chargers")
        if isinstance(chargers_raw, dict):
            for charger_id in chargers_raw:
                candidate = str(charger_id).strip()
                if candidate:
                    return candidate

        return "ev_charge_kw"

    def _rh1_battery_action_name(self, artifact_cfg: Dict[str, Any]) -> str:
        explicit = str(artifact_cfg.get("battery_action_name") or "").strip()
        if explicit:
            return explicit

        raw_soc_keys = artifact_cfg.get("battery_soc_keys", [])
        soc_keys: List[str] = []
        if isinstance(raw_soc_keys, list):
            soc_keys = [str(k) for k in raw_soc_keys if str(k)]
        elif raw_soc_keys is not None:
            key = str(raw_soc_keys).strip()
            if key:
                soc_keys = [key]

        for key in soc_keys:
            parts = [segment for segment in key.split(".") if segment]
            if len(parts) < 3:
                continue
            if parts[0].lower() != "batteries" or parts[-1].lower() != "soc":
                continue
            candidate = parts[1].strip()
            if candidate:
                return candidate

        return "battery_kw"

    def _action_bounds_for_name(self, manifest, agent_index: int, action_name: str) -> tuple[float | None, float | None]:
        action_names = manifest.get_action_names(agent_index)
        bounds = manifest.get_action_bounds(agent_index) or {}
        low_list = bounds.get("low") if isinstance(bounds, dict) else None
        high_list = bounds.get("high") if isinstance(bounds, dict) else None
        if not isinstance(low_list, list) or not isinstance(high_list, list):
            return None, None
        try:
            idx = action_names.index(action_name)
        except ValueError:
            return None, None
        low = _maybe_float(low_list[idx]) if idx < len(low_list) else None
        high = _maybe_float(high_list[idx]) if idx < len(high_list) else None
        return low, high

    def _extract_solar_generation(self, flat_payload: Mapping[str, Any], cfg: Dict[str, Any]) -> float:
        solar_key = str(cfg.get("solar_generation_key") or "solar_generation")
        direct = _maybe_float(flat_payload.get(solar_key))
        if direct is not None:
            return max(direct, 0.0)
        total = 0.0
        found = False
        for key, raw in flat_payload.items():
            if not isinstance(key, str):
                continue
            if not key.startswith("pv_panels.") or not key.endswith(".energy"):
                continue
            numeric = _maybe_float(raw)
            if numeric is None:
                continue
            total += numeric
            found = True
        return max(total, 0.0) if found else 0.0

    def _extract_first_factor(self, contexts: List[SiteContext], field: str, default: float) -> float:
        for ctx in contexts:
            value = _maybe_float(ctx.config.get(field))
            if value is not None:
                return max(value, 0.0)
        return default

    def _resolve_community_price(self, features_payload: Dict[str, Any], contexts: List[SiteContext]) -> tuple[float, float]:
        price_path = "community.price_signal"
        for ctx in contexts:
            configured = str(ctx.config.get("community_price_signal_key") or "").strip()
            if configured:
                price_path = configured
                break

        signal = self._read_nested_dict(features_payload, price_path)
        if isinstance(signal, dict):
            values = signal.get("values")
            if isinstance(values, list):
                cleaned = [float(v) for v in values if _maybe_float(v) is not None]
                if cleaned:
                    unit_hint = signal.get("measurement_unit")
                    normalized = [_normalize_price(v, unit_hint=unit_hint) for v in cleaned]
                    price_now = normalized[0]
                    future = normalized[1:] if len(normalized) > 1 else [price_now]
                    future_avg = sum(future) / len(future)
                    return price_now, future_avg

        community = features_payload.get("community")
        if isinstance(community, dict):
            signal = community.get("price_signal")
            if isinstance(signal, dict):
                values = signal.get("values")
                if isinstance(values, list):
                    cleaned = [float(v) for v in values if _maybe_float(v) is not None]
                    if cleaned:
                        unit_hint = signal.get("measurement_unit")
                        normalized = [_normalize_price(v, unit_hint=unit_hint) for v in cleaned]
                        price_now = normalized[0]
                        future = normalized[1:] if len(normalized) > 1 else [price_now]
                        future_avg = sum(future) / len(future)
                        return price_now, future_avg

        current_values: List[float] = []
        for ctx in contexts:
            series = _extract_vector_from_flat(ctx.flat_payload, "energy_price.values")
            if series:
                unit = ctx.flat_payload.get("energy_price.measurement_unit")
                current_values.append(_normalize_price(series[0], unit))
                continue
            scalar = _maybe_float(ctx.flat_payload.get("energy_price"))
            if scalar is not None:
                current_values.append(_normalize_price(scalar))
        if not current_values:
            return 0.0, 0.0
        avg = sum(current_values) / len(current_values)
        return avg, avg

    def _read_nested_dict(self, data: Dict[str, Any], dotted_path: str) -> Any:
        current: Any = data
        for token in dotted_path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(token)
        return current

    def _apply_battery_overlay(
        self,
        contexts: List[SiteContext],
        price_now: float,
        future_avg: float,
        actions_by_agent: Dict[str, Dict[str, float]],
    ) -> None:
        assets = [ctx.battery for ctx in contexts if ctx.battery is not None]
        if not assets:
            return

        dispatch: Dict[int, float] = {
            asset.agent_index: _clamp(asset.current_kw, asset.low_kw, asset.high_kw) for asset in assets
        }
        by_site: Dict[str, List[BatteryAsset]] = {}
        for asset in assets:
            by_site.setdefault(asset.site_key, []).append(asset)

        def site_net(site_key: str) -> float:
            base = next(ctx.net_without_battery_kw for ctx in contexts if ctx.site_key == site_key)
            battery_sum = sum(dispatch.get(asset.agent_index, 0.0) for asset in by_site.get(site_key, []))
            return base + battery_sum

        def distribute_charge(target_assets: List[BatteryAsset], amount_kw: float) -> float:
            remaining = max(amount_kw, 0.0)
            if remaining <= 1e-9:
                return 0.0
            changed = 0.0
            ranked = sorted(
                target_assets,
                key=lambda asset: (asset.high_kw - dispatch.get(asset.agent_index, 0.0)),
                reverse=True,
            )
            for asset in ranked:
                current = dispatch.get(asset.agent_index, 0.0)
                room = max(asset.high_kw - current, 0.0)
                if room <= 1e-9:
                    continue
                delta = min(room, remaining)
                dispatch[asset.agent_index] = current + delta
                remaining -= delta
                changed += delta
                if remaining <= 1e-9:
                    break
            return changed

        def distribute_discharge(target_assets: List[BatteryAsset], amount_kw: float) -> float:
            remaining = max(amount_kw, 0.0)
            if remaining <= 1e-9:
                return 0.0
            changed = 0.0
            ranked = sorted(
                target_assets,
                key=lambda asset: (dispatch.get(asset.agent_index, 0.0) - asset.low_kw),
                reverse=True,
            )
            for asset in ranked:
                current = dispatch.get(asset.agent_index, 0.0)
                room = max(current - asset.low_kw, 0.0)
                if room <= 1e-9:
                    continue
                delta = min(room, remaining)
                dispatch[asset.agent_index] = current - delta
                remaining -= delta
                changed += delta
                if remaining <= 1e-9:
                    break
            return changed

        # Step 1: solar-first per site (absorb local surplus).
        for site_key, site_assets in by_site.items():
            net_wo = next(ctx.net_without_battery_kw for ctx in contexts if ctx.site_key == site_key)
            if net_wo < -1e-9:
                distribute_charge(site_assets, -net_wo)

        # Step 2: community balancing with price-aware intensity.
        aggregate_net = sum(site_net(ctx.site_key) for ctx in contexts)
        threshold = max(abs(price_now) * 0.05, 0.01)

        if aggregate_net > 1e-9:
            # When current price is cheap vs future, preserve some battery for later.
            intensity = 0.3 if price_now + threshold < future_avg else 1.0
            target_discharge = aggregate_net * intensity
            distribute_discharge(assets, target_discharge)
        elif aggregate_net < -1e-9:
            # When current price is high vs future, keep part of surplus for export now.
            intensity = 0.3 if price_now - threshold > future_avg else 1.0
            target_charge = (-aggregate_net) * intensity
            distribute_charge(assets, target_charge)

        for asset in assets:
            value = _clamp(dispatch.get(asset.agent_index, 0.0), asset.low_kw, asset.high_kw)
            value = _round_one_decimal_towards_zero(value)
            value = _clamp(value, asset.low_kw, asset.high_kw)
            actions_by_agent[str(asset.agent_index)][asset.action_name] = float(value)

    def _external_cost(self, total_net_kw: float, price_now: float, export_factor: float) -> float:
        grid_import = max(total_net_kw, 0.0)
        grid_export = max(-total_net_kw, 0.0)
        return (grid_import * price_now) - (grid_export * price_now * export_factor)

    def _build_internal_settlement(
        self,
        site_net_kw: Dict[str, float],
        trade_price: float,
    ) -> Dict[str, Dict[str, float]]:
        deficits = {site: max(net, 0.0) for site, net in site_net_kw.items()}
        surpluses = {site: max(-net, 0.0) for site, net in site_net_kw.items()}
        total_deficit = sum(deficits.values())
        total_surplus = sum(surpluses.values())
        traded = min(total_deficit, total_surplus)
        settlement: Dict[str, Dict[str, float]] = {
            site: {
                "buy_kw": 0.0,
                "sell_kw": 0.0,
                "buy_cost": 0.0,
                "sell_revenue": 0.0,
            }
            for site in site_net_kw
        }
        if traded <= 1e-9:
            return settlement

        for site, deficit in deficits.items():
            if deficit <= 1e-9:
                continue
            buy_kw = traded * (deficit / total_deficit)
            settlement[site]["buy_kw"] = round(buy_kw, 6)
            settlement[site]["buy_cost"] = round(buy_kw * trade_price, 6)
        for site, surplus in surpluses.items():
            if surplus <= 1e-9:
                continue
            sell_kw = traded * (surplus / total_surplus)
            settlement[site]["sell_kw"] = round(sell_kw, 6)
            settlement[site]["sell_revenue"] = round(sell_kw * trade_price, 6)
        return settlement
