"""Inference pipeline that reconstructs preprocessing and runtime execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort

from app.services.preprocessor import AgentPreprocessor, build_encoder
from app.services.rbc import IchargingBreakerRuntime, IchargingRuntimeConfig
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
        self._icharging_runtime: IchargingBreakerRuntime | None = None
        if self.strategy in {"breaker_allocation", "icharging_breaker"}:
            icharging_cfg = IchargingRuntimeConfig.from_dict(config)
            self._icharging_runtime = IchargingBreakerRuntime(icharging_cfg)

    def infer(self, payload: Dict[str, float]) -> Dict[str, float]:
        payload = _apply_aliases(payload, self.feature_aliases)
        raw_payload = payload
        if self.preprocessor:
            features = self.preprocessor.transform(payload)
            raw_payload = {f"f{i}": v for i, v in enumerate(features.tolist())}

        if self._icharging_runtime:
            return self._icharging_runtime.allocate(payload)

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
