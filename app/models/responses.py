from __future__ import annotations

from typing import Dict, Any, List

from pydantic import BaseModel


class InferenceResponse(BaseModel):
    actions: Dict[str, Dict[str, float]]


class RewardResponse(BaseModel):
    rewards: Dict[str, float]


class InfoResponse(BaseModel):
    algorithm: str
    manifest_version: int
    metadata: Dict[str, Any]
    topology: Dict[str, Any]
    service_version: str
    agent_index: int
    action_names: List[str] | None = None
    uptime_seconds: float
    loaded_at: str | None = None
    alias_mapping_path: str | None = None
