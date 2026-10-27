"""Deterministic bounded correction, independent of HTTP, storage and model engines."""

from dataclasses import dataclass
from fractions import Fraction
import math

from .errors import ExecutionError


def decimal(value):
    return Fraction(str(value))


@dataclass(frozen=True)
class ActionLimit:
    name: str
    channel_id: str
    rule_id: str
    lower: Fraction
    upper: Fraction
    decimal_places: int

    @property
    def quantum(self):
        return Fraction(1, 10 ** self.decimal_places)

    @property
    def minimum_tick(self):
        return math.ceil(self.lower / self.quantum)

    @property
    def maximum_tick(self):
        return math.floor(self.upper / self.quantum)


@dataclass(frozen=True)
class AggregateLimit:
    rule_id: str
    constraint_id: str
    unit: str
    upper: Fraction
    # Canonical channel identities, not encoded features or equipment aliases.
    observations: tuple[tuple[str, Fraction], ...]
    actions: tuple[tuple[str, Fraction], ...]
    order: tuple[str, ...]


@dataclass(frozen=True)
class SafetyPlan:
    revision: str
    mode: str
    profile_id: str
    actions: tuple[ActionLimit, ...]
    aggregate: AggregateLimit | None = None


def json_number(value):
    try:
        result = float(value)
    except OverflowError as exc:
        raise ExecutionError("safety_numeric_range", "Safety result exceeds the JSON numeric range", "/safety") from exc
    if not math.isfinite(result) or (value != 0 and result == 0):
        raise ExecutionError("safety_numeric_range", "Safety result must be finite", "/safety")
    return result


def correct(plan, proposal, observations):
    """Return final actions and an audit. Physical infeasibility is reported, not raised."""
    audit = dict(mode=plan.mode, policy_revision=plan.revision, profile_id=plan.profile_id,
                 checked=plan.mode != "off", proposed_feasible=None, suggested_feasible=None,
                 final_feasible=None, changes=[], unresolved_constraints=[])
    if plan.mode == "off":
        return dict(proposal), audit
    limits = {a.name: a for a in plan.actions}
    original = {name: decimal(value) for name, value in proposal.items()}
    corrected, rules, reasons = {}, {n: [] for n in proposal}, {n: [] for n in proposal}
    proposed_ok = True
    for name, limit in limits.items():
        value = original[name]
        on_grid = (value / limit.quantum).denominator == 1
        in_bounds = limit.lower <= value <= limit.upper
        proposed_ok &= on_grid and in_bounds
        tick = math.floor(abs(value) / limit.quantum + Fraction(1, 2)) * (-1 if value < 0 else 1)
        tick = max(limit.minimum_tick, min(limit.maximum_tick, tick))
        corrected[name] = tick * limit.quantum
        if corrected[name] != value:
            rules[name].append(limit.rule_id)
            reasons[name].append("physical bounds" if not in_bounds else "decimal precision")
    aggregate = plan.aggregate
    unresolved = []
    if aggregate:
        base = sum((decimal(observations[channel]) * coefficient for channel, coefficient in aggregate.observations), Fraction())
        coefficients = dict(aggregate.actions)

        def total(values):
            return base + sum((values[name] * coefficient for name, coefficient in aggregate.actions), Fraction())

        proposed_ok &= total(original) <= aggregate.upper
        excess = total(corrected) - aggregate.upper
        # Each action is visited at most once. Ceil reduction preserves the upper
        # constraint after quantization, including unequal positive coefficients.
        for name in aggregate.order:
            if excess <= 0:
                break
            limit = limits[name]
            available = int(corrected[name] / limit.quantum) - limit.minimum_tick
            reduction = min(available, math.ceil(excess / (coefficients[name] * limit.quantum)))
            if reduction:
                corrected[name] -= reduction * limit.quantum
                excess -= reduction * coefficients[name] * limit.quantum
                rules[name].append(aggregate.rule_id)
                reasons[name].append("aggregate upper limit; explicit curtailment priority")
        # Recompute after the last quantized change rather than relying on the
        # running budget alone. Fractions need no permissive numeric tolerance.
        excess = total(corrected) - aggregate.upper
        if excess > 0:
            unresolved.append(dict(rule_id=aggregate.rule_id, constraint_id=aggregate.constraint_id,
                unit=aggregate.unit, value=json_number(total(corrected)), upper=json_number(aggregate.upper),
                excess=json_number(excess), reason="Configured action minima reached; uncontrolled contribution leaves a residual excess"))
    # Validate the final grid and bounds again, then verify JSON serialization
    # preserves every corrected action exactly in the decimal model.
    suggested = {}
    for name, value in corrected.items():
        limit = limits[name]
        if not (limit.lower <= value <= limit.upper and (value / limit.quantum).denominator == 1):
            raise ExecutionError("safety_invariant_failed", "Corrected action violates its configured grid or bounds", "/safety")
        suggested[name] = json_number(value)
        if decimal(suggested[name]) != value:
            raise ExecutionError("safety_precision_unrepresentable", "Corrected action cannot be represented without changing its value", "/safety")
        if value != original[name]:
            audit["changes"].append(dict(action_name=name, proposed_value=proposal[name], corrected_value=suggested[name],
                applied=plan.mode == "intervene", rule_ids=rules[name], reason="; ".join(reasons[name])))
    suggested_ok = not unresolved
    audit.update(proposed_feasible=bool(proposed_ok), suggested_feasible=suggested_ok,
                 final_feasible=suggested_ok if plan.mode == "intervene" else bool(proposed_ok),
                 unresolved_constraints=unresolved)
    return (suggested if plan.mode == "intervene" else dict(proposal)), audit
