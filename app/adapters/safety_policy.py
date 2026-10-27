"""Validate boundary capabilities and compile an immutable physical safety plan."""

from app.contracts.v2.safety import ConfigurationV2
from app.core.safety import ActionLimit, AggregateLimit, SafetyPlan, decimal


class SafetyConfigurationError(ValueError):
    def __init__(self, code, detail, path="/safety"):
        super().__init__(detail)
        self.code, self.detail, self.path = code, detail, path


def require(condition, code, detail, path="/safety"):
    if not condition:
        raise SafetyConfigurationError(code, detail, path)


def compile_safety(config):
    if not isinstance(config, ConfigurationV2):
        require(config.safety.mode == "off" and config.safety.profile_id == "off.v1" and not config.safety.rule_ids,
                "safety_profile_unsupported", "Use configuration.v2-draft and physical-limits.v1 for safety assessment")
        return None
    require(config.message.adapter_revision == "channels-1", "safety_adapter_unsupported", "Physical safety requires channels-1")
    channels = {c.channel_id: c for c in config.installation.channels}
    outputs = {b.channel_id: b.signal_name for b in config.bindings.outputs}
    inputs = {b.channel_id for b in config.bindings.inputs}
    rules = {r.channel_id: r for r in config.safety.action_rules}
    require(rules.keys() == outputs.keys(), "safety_coverage_invalid", "Action rules must cover exactly all output channels")
    actions = []
    for channel_id, name in outputs.items():
        channel, rule = channels[channel_id], rules[channel_id]
        bounds = channel.physical_bounds
        path = f"/installation/channels/{channel_id}/physical_bounds"
        require(channel.meaning.value_type == "number", "safety_type_unsupported", "Safety actions must be scalar numbers", path)
        require(bounds is not None and bounds.lower is not None and bounds.upper is not None,
                "safety_bounds_incomplete", "Both finite action bounds are required", path)
        require(max(abs(bounds.lower), abs(bounds.upper)) <= 1_000_000_000,
                "safety_bounds_unsupported", "Action bounds must be within +/-1e9 for this decimal profile", path)
        action = ActionLimit(name, channel_id, rule.rule_id, decimal(bounds.lower), decimal(bounds.upper), rule.decimal_places)
        require(action.minimum_tick <= action.maximum_tick, "safety_grid_empty", "No representable action lies inside its physical bounds", path)
        actions.append(action)
    limits, aggregate_rule = config.installation.aggregate_limits, config.safety.aggregate_rule
    require(len(limits) <= 1, "safety_aggregate_unsupported", "This profile supports at most one aggregate constraint")
    require(bool(limits) == (aggregate_rule is not None), "safety_aggregate_coverage", "Configure a rule for the installation aggregate constraint, and no extra rule")
    aggregate = None
    if limits:
        limit = limits[0]
        path = f"/installation/aggregate_limits/{limit.constraint_id}"
        require(aggregate_rule.constraint_id == limit.constraint_id, "safety_aggregate_coverage", "Aggregate rule references another constraint", path)
        require(limit.bounds.lower is None and limit.bounds.upper is not None,
                "safety_aggregate_unsupported", "Only an aggregate upper bound is supported", path)
        observed, controlled = [], []
        minimum = decimal(0)
        by_channel = {a.channel_id: a for a in actions}
        for term in limit.terms:
            coefficient, channel = decimal(term.coefficient), channels[term.channel_id]
            if term.channel_id in outputs:
                require(coefficient > 0, "safety_coefficient_unsupported", "Controlled aggregate coefficients must be positive", path)
                action = by_channel[term.channel_id]
                controlled.append((action.name, coefficient))
                minimum += coefficient * action.minimum_tick * action.quantum
            else:
                require(term.channel_id in inputs and channel.role == "observation" and channel.meaning.value_type == "number"
                        and channel.meaning.value_semantics == "instantaneous", "safety_observation_unsupported",
                        "Uncontrolled aggregate terms must be bound scalar instantaneous observations", path)
                bounds = channel.physical_bounds
                require(bounds is not None and bounds.lower is not None and bounds.upper is not None,
                        "safety_observation_bounds", "Uncontrolled terms need both physical bounds for static feasibility", path)
                observed.append((term.channel_id, coefficient))
                minimum += coefficient * decimal(bounds.lower if coefficient > 0 else bounds.upper)
        expected_channels = {t.channel_id for t in limit.terms if t.channel_id in outputs}
        require(bool(controlled) and set(aggregate_rule.curtailment_order) == expected_channels,
                "safety_priority_invalid", "Priority must contain exactly the controlled aggregate channels", path)
        require(minimum <= decimal(limit.bounds.upper), "safety_statically_infeasible", "Even the minimum possible aggregate exceeds its upper limit", path)
        aggregate = AggregateLimit(aggregate_rule.rule_id, limit.constraint_id, limit.unit, decimal(limit.bounds.upper),
                                   tuple(observed), tuple(controlled), tuple(outputs[c] for c in aggregate_rule.curtailment_order))
    return SafetyPlan(config.safety.revision, config.safety.mode, config.safety.profile_id, tuple(actions), aggregate)
