"""Explicit pilot profile: scalar bounds, decimal precision and one upper aggregate."""

from typing import List, Literal, Optional, Union
from pydantic import Field, StrictBool, conint, root_validator
from app.contracts.v1.common import Contract, Name, Number, unique
from app.contracts.v1.execution import ConfigurationDraft
from app.contracts.v1.decision import SafetyChange
from app.contracts.v1.evaluation import ExecutionReference, InputEvidence, ProposedAction


class ActionRule(Contract):
    rule_id: Name
    channel_id: Name
    decimal_places: conint(strict=True, ge=0, le=6)


class AggregateRule(Contract):
    rule_id: Name
    constraint_id: Name
    curtailment_order: List[Name] = Field(..., min_items=1)
    infeasible_behavior: Literal["minimum_actions_with_report"]

    @root_validator(skip_on_failure=True)
    def distinct(cls, values):
        order = values["curtailment_order"]
        if len(order) != len(set(order)):
            raise ValueError("Curtailment order must contain distinct channels")
        return values


class SafetyPolicyV2(Contract):
    schema_version: Literal["safety-policy.v2-draft"]
    revision: Name
    mode: Literal["off", "observe", "intervene"]
    profile_id: Literal["physical-limits.v1"]
    action_rules: List[ActionRule] = Field(..., min_items=1)
    aggregate_rule: Optional[AggregateRule] = None

    @root_validator(skip_on_failure=True)
    def distinct(cls, values):
        unique(values["action_rules"], "channel_id", "action rule channel")
        ids = [r.rule_id for r in values["action_rules"]]
        if values.get("aggregate_rule"):
            ids.append(values["aggregate_rule"].rule_id)
        if len(ids) != len(set(ids)):
            raise ValueError("Safety rule identities must be distinct")
        return values


class ConfigurationV2(ConfigurationDraft):
    schema_version: Literal["configuration.v2-draft"]
    safety: SafetyPolicyV2


Configuration = Union[ConfigurationDraft, ConfigurationV2]


def parse_configuration(value):
    model = ConfigurationV2 if value.get("schema_version") == "configuration.v2-draft" else ConfigurationDraft
    return model.parse_obj(value)


class FinalAction(Contract):
    action_name: Name
    channel_id: Name
    equipment_id: Optional[Name] = Field(...)
    unit: Name
    value: Number


class ResidualConstraint(Contract):
    rule_id: Name
    constraint_id: Name
    unit: Name
    value: Number
    upper: Number
    excess: Number
    reason: Name

    @root_validator(skip_on_failure=True)
    def positive_excess(cls, values):
        if values["excess"] <= 0 or values["value"] < values["upper"]:
            raise ValueError("Residual constraint must report positive excess")
        return values


class SafetyAudit(Contract):
    mode: Literal["off", "observe", "intervene"]
    policy_revision: Name
    profile_id: Literal["physical-limits.v1"]
    checked: StrictBool
    proposed_feasible: Optional[StrictBool] = Field(...)
    suggested_feasible: Optional[StrictBool] = Field(...)
    final_feasible: Optional[StrictBool] = Field(...)
    changes: List[SafetyChange]
    unresolved_constraints: List[ResidualConstraint]

    @root_validator(skip_on_failure=True)
    def assessment(cls, values):
        flags = [values[n] for n in ("proposed_feasible", "suggested_feasible", "final_feasible")]
        if values["mode"] == "off":
            if values["checked"] or any(f is not None for f in flags) or values["changes"] or values["unresolved_constraints"]:
                raise ValueError("Off mode cannot claim assessment or corrections")
        else:
            if not values["checked"] or any(f is None for f in flags):
                raise ValueError("Enabled safety requires feasibility assessment")
            if values["suggested_feasible"] != (not values["unresolved_constraints"]):
                raise ValueError("Suggested feasibility must reflect unresolved constraints")
            expected = values["suggested_feasible"] if values["mode"] == "intervene" else values["proposed_feasible"]
            if values["final_feasible"] != expected:
                raise ValueError("Final feasibility must reflect the selected mode")
        return values


class EvaluationResultV2(Contract):
    schema_version: Literal["evaluation.v2-draft"]
    evaluation_only: Literal[True]
    context: ExecutionReference
    proposed_actions: List[ProposedAction] = Field(..., min_items=1)
    actions: List[FinalAction] = Field(..., min_items=1)
    safety: SafetyAudit
    input_evidence: InputEvidence
    providers: List[Name]
    request_id: Optional[Name] = None

    @root_validator(skip_on_failure=True)
    def consistent(cls, values):
        proposed, final, audit = values["proposed_actions"], values["actions"], values["safety"]
        for items in (proposed, final, audit.changes):
            unique(items, "action_name", "action name")
            unique(items, "channel_id", "action channel")
        before, after = {a.action_name: a for a in proposed}, {a.action_name: a for a in final}
        if before.keys() != after.keys():
            raise ValueError("Proposed and final action sets differ")
        expected = {n: a.proposed_value for n, a in before.items()}
        if audit.policy_revision != values["context"].safety_policy_revision:
            raise ValueError("Safety revision differs from captured context")
        for change in audit.changes:
            action = before.get(change.action_name)
            if action is None or change.proposed_value != action.proposed_value:
                raise ValueError("Correction differs from original proposal")
            if any(getattr(change, k) != getattr(action, k) for k in ("channel_id", "equipment_id", "unit")):
                raise ValueError("Correction identity differs from proposal")
            if change.applied != (audit.mode == "intervene"):
                raise ValueError("Only intervention applies corrections")
            if change.applied:
                expected[change.action_name] = change.corrected_value
        for name, action in after.items():
            if any(getattr(action, k) != getattr(before[name], k) for k in ("channel_id", "equipment_id", "unit")) or action.value != expected[name]:
                raise ValueError("Final action differs from the audited correction")
        return values
