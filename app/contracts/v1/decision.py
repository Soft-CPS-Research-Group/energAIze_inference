"""Single-site decision and safety audit, independent of Percepta persistence."""

from typing import Dict, List, Literal, Optional

from pydantic import Field, StrictBool, conint, constr, root_validator

from .common import Contract, Name, Number, unique


class ActivationReference(Contract):
    activation_id: Name
    site_id: Name
    configuration_revision: Name
    installation_revision: Name
    bindings_revision: Name
    bundle_id: Name
    bundle_revision: Name
    manifest_sha256: constr(strict=True, regex=r"^[a-f0-9]{64}$")
    agent_index: conint(strict=True, ge=0)
    message_profile: Literal["percepta.mongo-message.v0"]
    adapter_revision: Name
    safety_policy_revision: Name
    service_version: Name


class SafetyChange(Contract):
    action_name: Name
    channel_id: Name
    equipment_id: Optional[Name] = Field(...)  # Explicit null for a site-owned channel.
    unit: Name
    proposed_value: Number
    corrected_value: Number
    applied: StrictBool
    rule_ids: List[Name] = Field(..., min_items=1)
    reason: Name

    @root_validator(skip_on_failure=True)
    def actual_change(cls, values):
        if values["proposed_value"] == values["corrected_value"]:
            raise ValueError("A safety change must describe different values")
        if len(values["rule_ids"]) != len(set(values["rule_ids"])):
            raise ValueError("Duplicate safety rule")
        return values


class SafetyReport(Contract):
    mode: Literal["off", "observe", "intervene"]
    changes: List[SafetyChange]


class DecisionResult(Contract):
    schema_version: Literal["decision.v1-draft"]
    request_id: Name
    activation: ActivationReference
    proposed_actions: Dict[Name, Number]
    actions: Dict[Name, Number]
    safety: SafetyReport

    @root_validator(skip_on_failure=True)
    def safety_consistency(cls, values):
        proposed, final, safety = values["proposed_actions"], values["actions"], values["safety"]
        if not proposed or proposed.keys() != final.keys():
            raise ValueError("Proposed and final actions must have the same nonempty action set")
        unique(safety.changes, "action_name", "safety change for action")
        unique(safety.changes, "channel_id", "safety change for channel")
        if safety.mode == "off" and safety.changes:
            raise ValueError("Safety off cannot report corrections")
        expected = dict(proposed)
        for change in safety.changes:
            if change.action_name not in proposed or proposed[change.action_name] != change.proposed_value:
                raise ValueError("Safety change does not match proposed action")
            if change.applied != (safety.mode == "intervene"):
                raise ValueError("Only intervene mode applies corrections")
            if change.applied:
                expected[change.action_name] = change.corrected_value
        if final != expected:
            raise ValueError("Final actions must match exactly the reported applied corrections")
        return values

    def legacy_response(self):
        """Explicit lossy projection; the current Percepta discards the safety audit."""
        return {"actions": {str(self.activation.agent_index): dict(self.actions)}, "request_id": self.request_id}
