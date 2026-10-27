"""Local activation and cycle fencing; coordination with other containers is external."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field, conint, root_validator, validator
from app.contracts.v1.common import Contract, Name
from app.contracts.v1.installation import InstallationContext
from app.contracts.v1.evaluation import ExecutionReference, InputEvidence, ProposedAction
from .safety import EvaluationResultV2, FinalAction, SafetyAudit


class InstallationUpdate(Contract):
    schema_version: Literal["installation-update.v1-draft"]
    expected_context_id: Optional[Name] = Field(...)
    expected_activation_id: Optional[Name] = Field(...)
    installation: InstallationContext


class Activate(Contract):
    schema_version: Literal["activation.v2-draft"]
    candidate_id: Name
    timeout_seconds: conint(strict=True, ge=1, le=3600) = 120
    expected_activation_id: Optional[Name] = Field(...)
    expected_context_id: Name


class CycleRequest(Contract):
    features: Dict[str, Any]
    expected_activation_id: Name
    expected_context_id: Name
    cycle_id: Name

    @validator("features", pre=True)
    def object_only(cls, value):
        if not isinstance(value, dict):
            raise ValueError("features must be an object")
        return value


class ActivationReference(Contract):
    activation_id: Name
    context_id: Name
    candidate_id: Name
    previous_activation_id: Optional[Name] = Field(...)
    activated_at: Name


class Decision(Contract):
    schema_version: Literal["decision.v2-draft"]
    activation: ActivationReference
    cycle_id: Name
    context: ExecutionReference
    proposed_actions: List[ProposedAction] = Field(..., min_items=1)
    actions: List[FinalAction] = Field(..., min_items=1)
    safety: SafetyAudit
    input_evidence: InputEvidence
    providers: List[Name]
    request_id: Optional[Name] = None

    @root_validator(skip_on_failure=True)
    def consistent(cls, values):
        evaluation = {k: v for k, v in values.items() if k not in {"schema_version", "activation", "cycle_id"}}
        EvaluationResultV2(schema_version="evaluation.v2-draft", evaluation_only=True, **evaluation)
        if values["activation"].candidate_id != values["context"].candidate_id:
            raise ValueError("Activation and execution context refer to different candidates")
        return values
