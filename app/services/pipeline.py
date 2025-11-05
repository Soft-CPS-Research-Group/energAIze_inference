"""Inference pipeline that reconstructs preprocessing and runtime execution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import math

import numpy as np
import onnxruntime as ort

from app.services.preprocessor import AgentPreprocessor, build_encoder
from app.settings import settings
from app.utils.manifest import Manifest
from app.logging import get_logger


def _apply_aliases(payload: Dict[str, float], aliases: Dict[str, str]) -> Dict[str, float]:
    """Apply feature alias mapping to the incoming payload."""
    if not aliases:
        return payload
    transformed = dict(payload)
    for alias, target in aliases.items():
        if alias in transformed and target not in transformed:
            transformed[target] = transformed.pop(alias)
    return transformed


class OnnxAgentRuntime:
    """Runtime wrapper around an ONNX model for a single agent."""

    def __init__(
        self,
        index: int,
        session: ort.InferenceSession,
        preprocessor: AgentPreprocessor,
        action_names: List[str],
        feature_aliases: Dict[str, str] | None = None,
    ):
        self.index = index
        self.session = session
        self.preprocessor = preprocessor
        self.action_names = action_names
        self.feature_aliases = feature_aliases or {}
        self.providers = session.get_providers()

    def infer(self, payload: Dict[str, float]) -> Dict[str, float]:
        payload = _apply_aliases(payload, self.feature_aliases)
        features = self.preprocessor.transform(payload)
        log = get_logger()
        log.debug("Running ONNX inference", agent_index=self.index)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: features.reshape(1, -1)})
        actions = outputs[0].squeeze(0)
        mapping: Dict[str, float] = {}
        for i, value in enumerate(actions.tolist()):
            key = self.action_names[i] if i < len(self.action_names) else f"action_{i}"
            mapping[key] = value
        return mapping


class RuleBasedRuntime:
    """Runtime for simple rule-based policies described in JSON."""

    def __init__(
        self,
        index: int,
        config: Dict[str, Any],
        action_names: List[str],
        preprocessor: Optional[AgentPreprocessor] = None,
        feature_aliases: Dict[str, str] | None = None,
    ):
        self.index = index
        self.config = config
        self.action_names = action_names
        self.preprocessor = preprocessor if config.get("use_preprocessor") else None
        self.feature_aliases = feature_aliases or {}
        self.rules = config.get("rules", [])
        default_actions = config.get("default_actions", {})
        if not default_actions and action_names:
            default_actions = {name: 0.0 for name in action_names}
        self.default_actions = default_actions
        self.providers = ["rule_based"]
        self.strategy = config.get("strategy")

    def infer(self, payload: Dict[str, float]) -> Dict[str, float]:
        payload = _apply_aliases(payload, self.feature_aliases)
        raw_payload = payload
        if self.preprocessor:
            features = self.preprocessor.transform(payload)
            raw_payload = {f"f{i}": v for i, v in enumerate(features.tolist())}

        if self.strategy == "breaker_allocation":
            return self._breaker_allocation(payload)

        for rule in self.rules:
            conditions = rule.get("if", {})
            if all(raw_payload.get(k) == v for k, v in conditions.items()):
                actions = rule.get("actions", {})
                get_logger().debug(
                    "Rule matched", agent_index=self.index, conditions=conditions
                )
                return {k: float(v) for k, v in actions.items()}

        get_logger().debug("No rule matched", agent_index=self.index)
        return {k: float(v) for k, v in self.default_actions.items()}

    def _breaker_allocation(self, payload: Dict[str, Any]) -> Dict[str, float]:
        cfg = self.config
        chargers_cfg: Dict[str, Dict[str, Any]] = cfg.get("chargers", {})
        charger_limit = float(cfg.get("charger_limit_kw", 0.0))
        max_board = float(cfg.get("max_board_kw", float("inf")))
        line_limits: Dict[str, Dict[str, Any]] = cfg.get("line_limits", {})
        control_minutes = float(cfg.get("control_interval_minutes", 15))
        control_minutes = max(control_minutes, 1.0)
        default_capacity = float(cfg.get("default_capacity_kwh", 60.0))
        default_target_soc = float(cfg.get("default_target_soc", 1.0))
        default_target_soc = min(max(default_target_soc, 0.0), 1.0)
        default_current_soc = min(max(float(cfg.get("default_current_soc", 0.0)), 0.0), 1.0)
        default_departure_buffer = float(cfg.get("default_departure_buffer_minutes", control_minutes))
        default_departure_buffer = max(default_departure_buffer, control_minutes)
        flexible_threshold = float(cfg.get("flexible_urgency_threshold", 0.6))

        def _maybe_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                numeric = float(value)
                if math.isnan(numeric):
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
                if math.isnan(numeric):
                    return None
                return numeric
            return None

        def _safe_float(value: Any, default: float = 0.0) -> float:
            numeric = _maybe_float(value)
            if numeric is None:
                return default
            return numeric

        def _clamp01(value: float) -> float:
            return min(max(value, 0.0), 1.0)


        def _safe_str(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            return str(value).strip()

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

        def _current_timestamp(data: Dict[str, Any]) -> datetime:
            candidates = [
                data.get("timestamp"),
                data.get("timestamp.$date"),
            ]
            for candidate in candidates:
                parsed = _parse_datetime(candidate)
                if parsed:
                    return parsed.astimezone(timezone.utc)
            return datetime.now(timezone.utc)

        now = _current_timestamp(payload)

        actions: Dict[str, float] = {}
        min_levels: Dict[str, float] = {}
        max_levels: Dict[str, float] = {}
        charger_priority: Dict[str, float] = {}
        baseline_levels: Dict[str, float] = {}
        active_chargers: List[str] = []
        planned_chargers: Set[str] = set()

        vehicle_capacities_cfg = {
            str(key): _safe_float(value, default_capacity)
            for key, value in (cfg.get("vehicle_capacities", {}) or {}).items()
        }
        flexible_ev_whitelist = {str(item) for item in cfg.get("flexible_ev_ids", []) if str(item)}
        allow_departure_fallback = bool(cfg.get("allow_departure_fallback", False))
        ev_charger_map_cfg: Dict[str, Set[str]] = {
            str(key): {str(item) for item in value}
            for key, value in (cfg.get("ev_charger_map", {}) or {}).items()
        }

        def _vehicle_capacity(ev_identifier: str, charger_meta: Dict[str, Any]) -> float:
            if ev_identifier in vehicle_capacities_cfg:
                capacity = vehicle_capacities_cfg[ev_identifier]
                if capacity > 0:
                    return capacity
            capacity_candidates = [
                payload.get(f"electric_vehicles.{ev_identifier}.flexibility.capacity_kwh"),
                payload.get(f"electric_vehicles.{ev_identifier}.capacity_kwh"),
                payload.get(f"electric_vehicles.{ev_identifier}.battery_capacity"),
                charger_meta.get("battery_capacity_kwh"),
                cfg.get("default_capacity_kwh", default_capacity),
            ]
            for candidate in capacity_candidates:
                numeric = _safe_float(candidate, -1.0)
                if numeric > 0:
                    return numeric
            return default_capacity

        for charger_id, meta in chargers_cfg.items():
            min_kw = float(meta.get("min_kw", 0.0))
            max_kw = float(meta.get("max_kw", charger_limit))
            min_kw = max(min_kw, 0.0)
            max_kw = max(max_kw, min_kw)

            min_levels[charger_id] = min_kw
            max_levels[charger_id] = max_kw

            ev_raw = payload.get(f"charging_sessions.{charger_id}.electric_vehicle")
            ev_id = _safe_str(ev_raw)
            session_power = _safe_float(payload.get(f"charging_sessions.{charger_id}.power"))
            session_power = max(min(session_power, max_kw), min_kw)
            active = bool(meta.get("active_by_default", False)) or bool(ev_id) or session_power > 0

            baseline_power = session_power if active else min_kw
            actions[charger_id] = baseline_power
            baseline_levels[charger_id] = baseline_power
            priority = baseline_power

            if active:
                active_chargers.append(charger_id)

            if ev_id:
                if flexible_ev_whitelist and ev_id not in flexible_ev_whitelist:
                    charger_priority[charger_id] = baseline_power
                    continue

                if ev_charger_map_cfg:
                    allowed_chargers = ev_charger_map_cfg.get(ev_id)
                    if not allowed_chargers:
                        charger_priority[charger_id] = baseline_power
                        continue
                    if charger_id not in allowed_chargers:
                        charger_priority[charger_id] = baseline_power
                        continue

                soc_key = f"electric_vehicles.{ev_id}.SoC"
                if soc_key not in payload:
                    charger_priority[charger_id] = baseline_power
                    continue
                soc_value = _maybe_float(payload.get(soc_key))
                if soc_value is None:
                    charger_priority[charger_id] = baseline_power
                    continue
                soc = _clamp01(soc_value)

                target_key = f"electric_vehicles.{ev_id}.flexibility.estimated_soc_at_departure"
                if target_key not in payload:
                    charger_priority[charger_id] = baseline_power
                    continue
                raw_target = _maybe_float(payload.get(target_key))
                if raw_target is None or not (0.0 < raw_target <= 1.0):
                    charger_priority[charger_id] = baseline_power
                    continue
                target_soc = _clamp01(raw_target)
                if target_soc <= soc + 1e-6:
                    charger_priority[charger_id] = baseline_power
                    continue

                capacity = _vehicle_capacity(ev_id, meta)

                energy_gap = max(target_soc - soc, 0.0) * capacity

                departure_raw = payload.get(
                    f"electric_vehicles.{ev_id}.flexibility.estimated_time_at_departure"
                )
                departure_time = _parse_datetime(departure_raw)
                if not departure_time:
                    if not allow_departure_fallback:
                        charger_priority[charger_id] = baseline_power
                        continue
                    departure_time = now + timedelta(minutes=default_departure_buffer)
                else:
                    departure_time = departure_time.astimezone(timezone.utc)

                minutes_remaining = (departure_time - now).total_seconds() / 60.0
                minutes_remaining = max(minutes_remaining, control_minutes)

                required_kw: float
                if energy_gap <= 1e-6:
                    required_kw = baseline_power
                elif minutes_remaining <= control_minutes:
                    required_kw = max_kw
                else:
                    required_kw = energy_gap / (minutes_remaining / 60.0)

                required_kw = max(min(required_kw, max_kw), min_kw)
                min_levels[charger_id] = max(min_levels[charger_id], required_kw)
                actions[charger_id] = max(actions[charger_id], required_kw, baseline_power)
                priority = max(required_kw, baseline_power)

                if energy_gap > 1e-6:
                    planned_chargers.add(charger_id)
                    charger_priority[charger_id] = priority
                else:
                    charger_priority[charger_id] = baseline_power
            else:
                charger_priority[charger_id] = baseline_power

        primary_flex: List[str] = []
        secondary_flex: List[str] = []
        for charger_id in active_chargers:
            max_kw = max_levels.get(charger_id, charger_limit)
            priority = charger_priority.get(charger_id, 0.0)
            if charger_id not in planned_chargers:
                secondary_flex.append(charger_id)
                continue
            if max_kw > 0:
                urgency_ratio = min(priority / max_kw, 1.0)
            else:
                urgency_ratio = 0.0
            if urgency_ratio < flexible_threshold:
                primary_flex.append(charger_id)

        flexible_ids = primary_flex + secondary_flex

        non_shiftable_key = cfg.get("non_shiftable_key", "non_shiftable_load")
        solar_key = cfg.get("solar_generation_key", "solar_generation")
        base_load = _safe_float(payload.get(non_shiftable_key)) - _safe_float(payload.get(solar_key))
        base_load = max(base_load, 0.0)

        def reduce_load(target_ids: List[str], amount: float) -> None:
            remaining = amount
            if not target_ids:
                return
            adjustable = [cid for cid in target_ids if actions.get(cid, 0.0) - min_levels.get(cid, 0.0) > 1e-6]
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
                    new_value = max(current - reduction, min_level)
                    actions[cid] = new_value
                    remaining -= reduction
                    if actions[cid] - min_level > 1e-6:
                        next_adjustable.append(cid)
                adjustable = next_adjustable
                if not next_adjustable:
                    break

        total_demand = base_load + sum(actions.values())
        board_overflow = total_demand - max_board
        if board_overflow > 1e-6:
            reduce_load(flexible_ids, board_overflow)
            total_demand = base_load + sum(actions.values())
            residual = total_demand - max_board
            if residual > 1e-6:
                reduce_load(list(actions.keys()), residual)

        for line_name, line_cfg in line_limits.items():
            limit = float(line_cfg.get("limit_kw", max_board))
            chargers = line_cfg.get("chargers", [])
            line_total = sum(actions.get(cid, 0.0) for cid in chargers)
            overflow = line_total - limit
            if overflow > 1e-6:
                flex_in_line = [cid for cid in chargers if cid in actions]
                reduce_load(flex_in_line, overflow)
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

        active_prioritised = sorted(
            [cid for cid in planned_chargers if cid in actions],
            key=lambda cid: charger_priority.get(cid, 0.0),
            reverse=True,
        )

        remaining_headroom = max(available_headroom, 0.0)
        for cid in active_prioritised:
            if remaining_headroom <= 1e-6:
                break
            current = actions.get(cid, 0.0)
            max_level = max_levels.get(cid, current)
            if current >= max_level - 1e-6:
                continue
            increment = min(max_level - current, remaining_headroom)
            actions[cid] = current + increment
            remaining_headroom -= increment

        battery_actions: Dict[str, float] = {}
        for battery_id, meta in cfg.get("batteries", {}).items():
            battery_actions[battery_id] = 0.0

        for cid, value in actions.items():
            min_level = min_levels.get(cid, 0.0)
            max_level = max_levels.get(cid, value)
            actions[cid] = max(min(value, max_level), min_level)

        ordered = cfg.get("action_order") or (list(chargers_cfg.keys()) + list(battery_actions.keys()))
        result: Dict[str, float] = {}
        for action_id in ordered:
            if action_id in actions:
                result[action_id] = float(actions[action_id])
            elif action_id in battery_actions:
                result[action_id] = float(battery_actions[action_id])
            else:
                result[action_id] = 0.0

        for battery_id, value in battery_actions.items():
            if battery_id not in result:
                result[battery_id] = float(value)

        return result


class InferencePipeline:
    """End-to-end pipeline that handles preprocessing, runtime, and postprocessing."""

    def __init__(
        self,
        manifest: Manifest,
        artifacts_root: Path,
        agent_index: int,
        alias_overrides: Optional[Dict[str, str]] = None,
    ):
        self.manifest = manifest
        self.artifacts_root = artifacts_root
        self.agent_index = agent_index
        self.alias_overrides = alias_overrides or {}
        self._agent = self._build_agent()

    def _build_agent(self):
        env = self.manifest.environment
        action_names = env.action_names or []

        artifact = next(
            (art for art in self.manifest.agent.artifacts if art.agent_index == self.agent_index),
            None,
        )
        if artifact is None:
            raise ValueError(f"Agent index {self.agent_index} not found in manifest")

        get_logger().info(
            "Loading agent",
            agent_index=artifact.agent_index,
            format=artifact.format or "onnx",
            providers=settings.onnx_execution_providers,
        )

        encoder_specs = env.encoders[artifact.agent_index]
        observation_names = env.observation_names[artifact.agent_index]
        encoders = [build_encoder(spec) for spec in encoder_specs]
        preprocessor = AgentPreprocessor(observation_names, encoders)

        artifact_path = self.manifest.resolve_artifact_path(self.artifacts_root, artifact)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        artifact_config = dict(artifact.config or {})
        feature_aliases = dict(self.alias_overrides)

        if artifact.format in (None, "onnx"):
            session = ort.InferenceSession(
                path_or_bytes=artifact_path.as_posix(),
                providers=settings.onnx_execution_providers,
            )
            return OnnxAgentRuntime(
                index=artifact.agent_index,
                session=session,
                preprocessor=preprocessor,
                action_names=action_names,
                feature_aliases=feature_aliases,
            )

        if artifact.format == "rule_based":
            import json

            config = dict(artifact_config)
            if "config_path" in config:
                config_path = self.artifacts_root / config["config_path"]
                with config_path.open("r", encoding="utf-8") as handle:
                    config.update(json.load(handle))
            with artifact_path.open("r", encoding="utf-8") as handle:
                policy_data = json.load(handle)
            config.setdefault("default_actions", policy_data.get("default_actions", {}))
            config.setdefault("rules", policy_data.get("rules", []))
            return RuleBasedRuntime(
                index=artifact.agent_index,
                config=config,
                action_names=action_names,
                preprocessor=preprocessor,
                feature_aliases=feature_aliases,
            )

        raise NotImplementedError(f"Unsupported artifact format '{artifact.format}'")

    def inference(self, payload: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Run inference and map outputs back to named actions."""
        actions = self._agent.infer(payload)
        return {str(self._agent.index): actions}

    @property
    def agent(self):
        return self._agent
