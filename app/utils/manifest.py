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
    action_bounds: List[Any] = Field(default_factory=list)
    action_names: Optional[List[str]] = None
    action_names_by_agent: Optional[Dict[str, List[str]] | List[List[str]]] = None
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

    def get_artifact(self, agent_index: int) -> ArtifactSpec:
        artifact = next(
            (item for item in self.agent.artifacts if item.agent_index == agent_index),
            None,
        )
        if artifact is None:
            raise ValueError(f"Agent index {agent_index} not found in manifest")
        return artifact

    def get_observation_names(self, agent_index: int) -> List[str]:
        values = self._resolve_agent_value(self.environment.observation_names, agent_index, "observation_names")
        if not isinstance(values, list):
            raise ValueError("observation_names entry must be a list of feature names")
        return [str(item) for item in values]

    def get_encoder_specs(self, agent_index: int) -> List[EncoderSpec]:
        values = self._resolve_agent_value(self.environment.encoders, agent_index, "encoders")
        if not isinstance(values, list):
            raise ValueError("encoders entry must be a list")
        return values

    def get_action_bounds(self, agent_index: int) -> Dict[str, Any] | None:
        if not self.environment.action_bounds:
            return None
        values = self._resolve_agent_value(self.environment.action_bounds, agent_index, "action_bounds")
        if isinstance(values, list):
            if not values:
                return None
            first = values[0]
            return first if isinstance(first, dict) else None
        if isinstance(values, dict):
            return values
        return None

    def get_action_names(self, agent_index: int) -> List[str]:
        by_agent = self.environment.action_names_by_agent
        if isinstance(by_agent, dict):
            names = by_agent.get(str(agent_index))
            if names is None:
                names = by_agent.get(agent_index)  # type: ignore[arg-type]
            if names:
                return [str(item) for item in names]
        elif isinstance(by_agent, list):
            values = self._resolve_agent_value(by_agent, agent_index, "action_names_by_agent")
            if isinstance(values, list):
                return [str(item) for item in values]

        names = self.environment.action_names or []
        return [str(item) for item in names]

    @staticmethod
    def _resolve_agent_value(values: List[Any], agent_index: int, field_name: str) -> Any:
        if not values:
            raise ValueError(f"Manifest field '{field_name}' is empty")
        if 0 <= agent_index < len(values):
            return values[agent_index]
        if len(values) == 1:
            return values[0]
        raise ValueError(
            f"Manifest field '{field_name}' has no entry for agent_index={agent_index} "
            f"(entries={len(values)})"
        )


def load_manifest(path: Path) -> Manifest:
    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Manifest.parse_obj(payload)
