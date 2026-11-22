"""Ports used by the application, without importing concrete runtime libraries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

Features = dict[str, Any]
Actions = dict[str, dict[str, float]]


class Artifact(Protocol):
    config: dict[str, Any] | None


class ManifestView(Protocol):
    manifest_version: int
    metadata: dict[str, Any]
    algorithm: dict[str, Any]
    topology: dict[str, Any]

    def get_artifact(self, agent_index: int) -> Artifact: ...
    def get_action_names(self, agent_index: int) -> list[str]: ...


class AgentRuntime(Protocol):
    index: int
    providers: list[str]

    def infer(self, payload: Features) -> dict[str, float]: ...
    def propose(self, payload: Features) -> dict[str, float]: ...


class TensorSession(Protocol):
    """Shared session boundary for ONNX Runtime and native TensorRT adapters."""

    def get_providers(self) -> list[str]: ...
    def get_inputs(self) -> list[Any]: ...
    def get_outputs(self) -> list[Any]: ...
    def run(self, output_names: list[str] | None, feed: dict[str, Any]) -> list[Any]: ...


class Pipeline(Protocol):
    manifest: ManifestView
    agent_index: int

    @property
    def agent(self) -> AgentRuntime: ...

    def inference(self, payload: Features) -> Actions: ...


@dataclass
class PipelineRecord:
    """One completely constructed set of pipelines, published together."""

    pipelines: dict[int, Pipeline]
    manifest_path: Path
    artifacts_dir: Path
    agent_index: int
    loaded_agent_indices: list[int]
    alias_mapping_path: Path | None
    loaded_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def pipeline(self) -> Pipeline:
        return self.pipelines[self.agent_index]

    @property
    def default_agent_index(self) -> int:
        return self.agent_index


class BundleLoader(Protocol):
    def load(
        self,
        manifest_path: Path,
        artifacts_dir: Path | None,
        agent_index: int | None = None,
        alias_mapping_path: Path | None = None,
    ) -> PipelineRecord: ...


class ActivePipelines(Protocol):
    def is_configured(self) -> bool: ...
    def get_pipeline(self, agent_index: int | None = None) -> Pipeline: ...
    def get_record(self) -> PipelineRecord | None: ...
    def activate(self, record: PipelineRecord) -> None: ...
    def unload(self) -> None: ...
    def get_startup_time(self) -> datetime: ...


PrepareInput = Callable[[Features, Pipeline], Features]
ObserveActions = Callable[[Pipeline, Features, Actions], None]
