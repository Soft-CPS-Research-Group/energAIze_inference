from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from loguru import logger

from app.services.preprocessor import AgentPreprocessor, build_encoder
from app.utils.manifest import Manifest


class OnnxAgentRuntime:
    def __init__(self, index: int, session: ort.InferenceSession, preprocessor: AgentPreprocessor, action_names: List[str]):
        self.index = index
        self.session = session
        self.preprocessor = preprocessor
        self.action_names = action_names

    def infer(self, payload: Dict[str, float]) -> Dict[str, float]:
        features = self.preprocessor.transform(payload)
        logger.debug("Running ONNX inference for agent %s", self.index)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: features.reshape(1, -1)})
        actions = outputs[0].squeeze(0)
        mapping: Dict[str, float] = {}
        for i, value in enumerate(actions.tolist()):
            if i < len(self.action_names):
                key = self.action_names[i]
            else:
                key = f"action_{i}"
            mapping[key] = value
        return mapping


class RuleBasedRuntime:
    def __init__(self, index: int, config: Dict[str, Any], action_names: List[str], preprocessor: AgentPreprocessor | None = None):
        self.index = index
        self.config = config
        self.action_names = action_names
        self.preprocessor = preprocessor if config.get("use_preprocessor") else None
        self.rules = config.get("rules", [])
        default_actions = config.get("default_actions", {})
        if not default_actions and action_names:
            default_actions = {name: 0.0 for name in action_names}
        self.default_actions = default_actions

    def infer(self, payload: Dict[str, float]) -> Dict[str, float]:
        raw_payload = payload
        if self.preprocessor:
            features = self.preprocessor.transform(payload)
            raw_payload = {f"f{i}": v for i, v in enumerate(features.tolist())}

        for rule in self.rules:
            conditions = rule.get("if", {})
            if all(raw_payload.get(k) == v for k, v in conditions.items()):
                actions = rule.get("actions", {})
                logger.debug("Rule matched for agent %s: %s", self.index, conditions)
                return {k: float(v) for k, v in actions.items()}

        logger.debug("No rule matched for agent %s; using default actions", self.index)
        return {k: float(v) for k, v in self.default_actions.items()}


class InferencePipeline:
    def __init__(self, manifest: Manifest, artifacts_root: Path, agent_index: int):
        self.manifest = manifest
        self.artifacts_root = artifacts_root
        self.agent_index = agent_index
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

        logger.info(
            "Loading agent index {} (format={})", artifact.agent_index, artifact.format or "onnx"
        )

        encoder_specs = env.encoders[artifact.agent_index]
        observation_names = env.observation_names[artifact.agent_index]
        encoders = [build_encoder(spec) for spec in encoder_specs]
        preprocessor = AgentPreprocessor(observation_names, encoders)

        artifact_path = self.manifest.resolve_artifact_path(self.artifacts_root, artifact)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        if artifact.format in (None, "onnx"):
            session = ort.InferenceSession(
                path_or_bytes=artifact_path.as_posix(),
                providers=["CPUExecutionProvider"],
            )
            return OnnxAgentRuntime(
                index=artifact.agent_index,
                session=session,
                preprocessor=preprocessor,
                action_names=action_names,
            )

        if artifact.format == "rule_based":
            import json

            config = dict(artifact.config)
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
            )

        raise NotImplementedError(f"Unsupported artifact format '{artifact.format}'")

    def inference(self, payload: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        actions = self._agent.infer(payload)
        return {str(self._agent.index): actions}

    @property
    def agent(self):
        return self._agent
