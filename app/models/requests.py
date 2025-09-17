from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature name/value pairs for the agent")


class RewardRequest(BaseModel):
    observations: Dict[str, float] = Field(
        ..., description="Observation data required for reward calculation"
    )
