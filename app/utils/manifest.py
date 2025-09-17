from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EncoderSpec(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ArtifactSpec(BaseModel):
    agent_index: int
    path: str
    observation_dimension: int | None = None
    action_dimension: int | None = None
    format: str | None = None
    config: Dict[str, Any] = Field(default_factory=dict)


class AgentMetadata(BaseModel):
    format: str = "onnx"
    artifacts: List[ArtifactSpec]


class RewardMetadata(BaseModel):
    name: Optional[str]
    params: Dict[str, Any] = Field(default_factory=dict)


class EnvironmentMetadata(BaseModel):
    observation_names: List[List[str]]
    encoders: List[List[EncoderSpec]]
    action_bounds: List[Optional[Dict[str, Any]]]
    action_names: Optional[List[str]] = None
    reward_function: RewardMetadata


class Manifest(BaseModel):
    manifest_version: int = 1
    metadata: Dict[str, Any]
    simulator: Dict[str, Any]
    training: Dict[str, Any]
    topology: Dict[str, Any]
    algorithm: Dict[str, Any]
    environment: EnvironmentMetadata
    agent: AgentMetadata

    def resolve_artifact_path(self, root: Path, artifact: ArtifactSpec) -> Path:
        candidate = root / artifact.path
        return candidate.resolve()


def load_manifest(path: Path) -> Manifest:
    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Manifest.parse_obj(payload)
