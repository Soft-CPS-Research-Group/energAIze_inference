"""Build concrete runtimes from the current manifest and local artifacts."""

from pathlib import Path
from typing import Dict

from app.core.contracts import AgentRuntime
from app.adapters.onnx_backend import TENSORRT_PROVIDER, requested_providers, ensure_available
from app.logging import get_logger
from app.services.preprocessor import AgentPreprocessor, build_encoder
from app.services.runtimes.onnx import OnnxAgentRuntime
from app.services.runtimes.rule_based import RuleBasedRuntime
from app.utils.manifest import Manifest


class ArtifactRuntimeFactory:
    def __init__(self, providers: list[str], tensorrt_enabled: bool = False) -> None:
        self.providers = list(providers)
        self.tensorrt_enabled = tensorrt_enabled

    def build(
        self, manifest: Manifest, artifacts_root: Path, agent_index: int,
        alias_overrides: Dict[str, str],
    ) -> AgentRuntime:
        action_names = manifest.get_action_names(agent_index)
        artifact = manifest.get_artifact(agent_index)

        get_logger().info(
            "Loading agent",
            agent_index=artifact.agent_index,
            format=artifact.format or "onnx",
            execution_device=(artifact.config or {}).get("execution_device", "deployment_default"),
        )

        encoder_specs = manifest.get_encoder_specs(artifact.agent_index)
        observation_names = manifest.get_observation_names(artifact.agent_index)
        encoders = [build_encoder(spec) for spec in encoder_specs]
        preprocessor = AgentPreprocessor(observation_names, encoders)

        artifact_path = manifest.resolve_artifact_path(artifacts_root, artifact)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        artifact_config = dict(artifact.config or {})
        feature_aliases = dict(alias_overrides)

        if artifact.format in (None, "onnx"):
            providers = requested_providers(artifact_config, self.providers)
            ensure_available(providers, self.tensorrt_enabled)
            if providers == [TENSORRT_PROVIDER]:
                from app.adapters.tensorrt_session import TensorRTSession
                session = TensorRTSession(artifact_path)
            else:
                import onnxruntime as ort
                session = ort.InferenceSession(
                    path_or_bytes=artifact_path.as_posix(), providers=providers)
            if session.get_providers()[0] != providers[0] or any(
                    p not in session.get_providers() for p in providers):
                raise ValueError("Runtime silently fell back from the requested providers")
            bounds = manifest.get_action_bounds(artifact.agent_index)
            return OnnxAgentRuntime(
                index=artifact.agent_index,
                session=session,
                preprocessor=preprocessor,
                action_names=action_names,
                feature_aliases=feature_aliases,
                action_bounds=bounds,
            )

        if artifact.format == "rule_based":
            import json

            config = dict(artifact_config)
            if "config_path" in config:
                config_path = artifacts_root / config["config_path"]
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
