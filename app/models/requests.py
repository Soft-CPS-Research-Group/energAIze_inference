from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    agent_index: int | None = Field(
        default=None,
        ge=0,
        description="Optional target agent index. When omitted, the default loaded agent is used.",
    )
    features: Dict[str, Any] = Field(
        ..., description="Feature name/value pairs for the agent (supports nested objects)"
    )
