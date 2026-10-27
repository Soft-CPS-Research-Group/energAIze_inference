"""Pipeline facade: delegate execution to an injected runtime factory."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Protocol, TYPE_CHECKING

from app.core.contracts import Actions, AgentRuntime, Features

if TYPE_CHECKING:
    from app.utils.manifest import Manifest


class RuntimeFactory(Protocol):
    def build(
        self, manifest: Manifest, artifacts_root: Path, agent_index: int,
        alias_overrides: Dict[str, str],
    ) -> AgentRuntime: ...


class InferencePipeline:
    """End-to-end pipeline that handles preprocessing, runtime, and postprocessing."""

    def __init__(
        self,
        manifest: Manifest,
        artifacts_root: Path,
        agent_index: int,
        alias_overrides: Optional[Dict[str, str]] = None,
        *,
        runtime_factory: RuntimeFactory | None = None,
    ):
        self.manifest = manifest
        self.artifacts_root = artifacts_root
        self.agent_index = agent_index
        self.alias_overrides = alias_overrides or {}
        if runtime_factory is None:
            # Default retained for existing scripts; production composition injects it.
            from app.adapters.runtime_factory import ArtifactRuntimeFactory
            from app.settings import settings

            runtime_factory = ArtifactRuntimeFactory(settings.onnx_execution_providers)
        self._agent = runtime_factory.build(
            manifest, artifacts_root, agent_index, self.alias_overrides
        )

    def inference(self, payload: Features) -> Actions:
        """Run inference and map outputs back to named actions."""
        actions = self._agent.infer(payload)
        return {str(self._agent.index): actions}

    @property
    def agent(self) -> AgentRuntime:
        return self._agent


def __getattr__(name: str):
    """Preserve legacy runtime imports without eagerly importing execution engines."""
    if name == "OnnxAgentRuntime":
        from app.services.runtimes.onnx import OnnxAgentRuntime
        return OnnxAgentRuntime
    if name == "RuleBasedRuntime":
        from app.services.runtimes.rule_based import RuleBasedRuntime
        return RuleBasedRuntime
    raise AttributeError(name)
