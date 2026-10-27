"""Candidate proposal evaluation; deliberately distinct from an active decision."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, conint, constr, validator

from .common import Contract, Name, Number


class EvaluationRequest(Contract):
    features: Dict[str, Any]

    @validator("features", pre=True)
    def message_object(cls, value):
        if not isinstance(value, dict):
            raise ValueError("features must be a JSON object")
        return value


class ExecutionReference(Contract):
    candidate_id: Name
    site_id: Name
    configuration_revision: Name
    installation_revision: Name
    bindings_revision: Name
    bundle_id: Name
    bundle_revision: Name
    manifest_sha256: constr(strict=True, regex=r"^[a-f0-9]{64}$")
    agent_index: conint(strict=True, ge=0)
    message_profile: Literal["percepta.mongo-message.v0"]
    adapter_revision: Literal["channels-1"]
    producer_revision: Optional[Name] = Field(...)
    safety_policy_revision: Name
    service_version: Name


class ProposedAction(Contract):
    action_name: Name
    channel_id: Name
    equipment_id: Optional[Name] = Field(...)
    unit: Name
    proposed_value: Number


class SignalEvidence(Contract):
    signal_name: Name
    channel_id: Name
    source_unit: Name
    unit: Name
    unit_source: Literal["configuration", "message"]
    interval_seconds: Optional[Number] = Field(...)
    interval_source: Literal["configuration", "message", "unknown"]
    converted: bool


class InputEvidence(Contract):
    timestamp: Optional[Name] = Field(...)
    freshness: Literal["unknown", "not_assessed"]
    inputs: List[SignalEvidence]


class EvaluationResult(Contract):
    schema_version: Literal["evaluation.v1-draft"]
    proposal_only: Literal[True]
    context: ExecutionReference
    proposed_actions: List[ProposedAction] = Field(..., min_items=1)
    input_evidence: InputEvidence
    providers: List[Name]
    request_id: Optional[Name] = None
