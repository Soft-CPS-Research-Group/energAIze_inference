from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ..., description="Feature name/value pairs for the agent (supports nested objects)"
    )


class RewardRequest(BaseModel):
    observations: Dict[str, Any] = Field(
        ..., description="Observation data required for reward calculation"
    )
