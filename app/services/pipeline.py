"""Inference pipeline that reconstructs preprocessing and runtime execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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

        def _to_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        def _to_str(value: Any) -> str:
            if value is None:
                return ""
            return str(value).strip()

        actions: Dict[str, float] = {}
        min_levels: Dict[str, float] = {}
        max_levels: Dict[str, float] = {}
        allow_flex_map: Dict[str, bool] = {}
        flexible_ids: List[str] = []

        for charger_id, meta in chargers_cfg.items():
            min_kw = float(meta.get("min_kw", 0.0))
            max_kw = float(meta.get("max_kw", charger_limit))
            min_kw = max(min_kw, 0.0)
            max_kw = max(max_kw, min_kw)
            min_levels[charger_id] = min_kw
            max_levels[charger_id] = max_kw
            allow_flex_map[charger_id] = bool(meta.get("allow_flex_when_ev", True))

            session_ev = _to_str(payload.get(f"charging_sessions.{charger_id}.electric_vehicle"))
            session_power = _to_float(payload.get(f"charging_sessions.{charger_id}.power"))
            active = bool(meta.get("active_by_default", False)) or bool(session_ev) or session_power > 0

            actions[charger_id] = max_kw if active else min_kw

            if meta.get("flexible_default", False) and active:
                flexible_ids.append(charger_id)

        if cfg.get("detect_flex_from_ev", True):
            for key, value in payload.items():
                if not isinstance(value, str):
                    continue
                if ".flexibility" not in key:
                    continue
                charger_id = value
                if charger_id in actions and allow_flex_map.get(charger_id, True):
                    flexible_ids.append(charger_id)

        flexible_set = {cid for cid in flexible_ids if actions.get(cid, 0.0) > min_levels.get(cid, 0.0) + 1e-6}
        flexible_ids = list(flexible_set)

        non_shiftable_key = cfg.get("non_shiftable_key", "non_shiftable_load")
        solar_key = cfg.get("solar_generation_key", "solar_generation")
        base_load = _to_float(payload.get(non_shiftable_key)) - _to_float(payload.get(solar_key))
        base_load = max(base_load, 0.0)

        def reduce_load(target_ids: List[str], amount: float) -> None:
            remaining = amount
            if not target_ids:
                return
            adjustable = [cid for cid in target_ids if actions.get(cid, 0.0) - min_levels.get(cid, 0.0) > 1e-6]
            if not adjustable:
                adjustable = [cid for cid in target_ids if cid in actions]
            while remaining > 1e-6 and adjustable:
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
                flex_in_line = [cid for cid in chargers if chargers_cfg.get(cid, {}).get("flexible")]
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

        battery_actions: Dict[str, float] = {}
        remaining_headroom = max(available_headroom, 0.0)
        for battery_id, meta in cfg.get("batteries", {}).items():
            soc_key = meta.get("soc_key")
            charge_threshold = float(meta.get("charge_threshold", 0.5))
            charge_power = float(meta.get("charge_power_kw", 0.0))
            idle_power = float(meta.get("idle_power_kw", 0.0))
            soc = _to_float(payload.get(soc_key)) if soc_key else 0.0
            if soc < charge_threshold and remaining_headroom > 1e-6:
                allocation = min(charge_power, remaining_headroom)
                battery_actions[battery_id] = allocation
                remaining_headroom -= allocation
            else:
                battery_actions[battery_id] = idle_power

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
