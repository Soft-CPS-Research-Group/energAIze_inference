from __future__ import annotations

from typing import Any, Dict, Union

from pydantic import BaseModel, Field


ScalarValue = Union[float, int, str, bool]


class InferenceRequest(BaseModel):
    features: Dict[str, ScalarValue] = Field(
        ..., description="Feature name/value pairs for the agent (strings, numbers, booleans)"
    )


class RewardRequest(BaseModel):
    observations: Dict[str, Any] = Field(
        ..., description="Observation data required for reward calculation"
    )
