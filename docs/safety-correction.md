# Safety correction — iteration 5

The candidate evaluator now applies the configured physical protection **after**
the model's raw proposal. It preserves original observations, raw actions and
identities, and returns the final evaluated actions alongside every net correction.
Physical violations covered by this profile produce a result and audit, including
an explicit residual when the configured controls cannot satisfy the aggregate.
Missing data, invalid outputs and engine failures remain technical errors.

This is local candidate evaluation, not activation or physical dispatch. The
response says `evaluation_only: true`; it does not invent an activation ID. The
legacy `/inference` path retains its existing response. [Iteration 6](local-lifecycle.md) now publishes and recovers the complete
configuration/bundle/safety combination through a separate active route. Delivery and
persistence through Percepta/Mongo are iteration 7.

## Separation of responsibilities

```text
Percepta message -> bound canonical inputs -> model -> raw proposed actions
                         |                                |
                         +-> physical safety plan <-------+
                                      |
                              final actions + audit
```

- `app/contracts/v2/safety.py`: versioned boundary and response consistency checks.
- `app/adapters/safety_policy.py`: capability checks and compilation from the
  installation, bindings and policy into an immutable plan.
- `app/core/safety.py`: deterministic correction without HTTP, storage, backend
  access, Pydantic or model-engine dependencies.
- `app/application/evaluation.py`: captures canonical observations before calling
  the runtime, validates its proposal, invokes correction and attaches identities.

Safety uses physical values after declared unit conversion and **before** model
encoding. The original message is never rewritten. Input conversion evidence and
action corrections occupy separate fields in the response. Runtime continues to
supply the installation snapshot; inference does not query equipment APIs per cycle.

## Explicit configuration and response versions

The administrative routes retain the `/admin/v1` prefix. Their document schemas
are separately versioned. PUT/GET configuration and startup files support both:

| Configuration | Policy | Evaluation response |
| --- | --- | --- |
| `configuration.v1-draft` | Existing `off.v1`, no rules | `evaluation.v1-draft`, proposal only |
| `configuration.v2-draft` | `safety-policy.v2-draft`, `physical-limits.v1` | `evaluation.v2-draft`, proposal, final actions and audit |

Old documents and exported v1 schemas are unchanged. There is no silent migration.
To enable protection, supply a new immutable configuration revision using v2,
prepare its candidate, then evaluate it. Altering a stored revision still conflicts.
For v2, model proposals may exceed physical limits in any of the three modes: the
returned mode and assessment explicitly describe whether correction was applied.
The legacy v1 preparation restrictions on declared output ranges remain intact.

Policy example (complete configuration in `examples/safety_demo/configuration.json`):

```json
{
  "schema_version": "safety-policy.v2-draft",
  "revision": "safety-1",
  "mode": "intervene",
  "profile_id": "physical-limits.v1",
  "action_rules": [
    {"rule_id": "charger-1-bounds", "channel_id": "charger-1-setpoint", "decimal_places": 1},
    {"rule_id": "charger-2-bounds", "channel_id": "charger-2-setpoint", "decimal_places": 1}
  ],
  "aggregate_rule": {
    "rule_id": "site-budget",
    "constraint_id": "site-import",
    "curtailment_order": ["charger-2-setpoint", "charger-1-setpoint"],
    "infeasible_behavior": "minimum_actions_with_report"
  }
}
```

Limits and aggregate coefficients belong to the installation context. The policy
selects precision and curtailment order; its first channel is reduced first.
Every output must have exactly one action rule. Rule identities must be distinct.
Equipment identity and units come from the captured bindings, not model aliases.

## Modes and response meaning

| Mode | Final evaluated actions | Audit |
| --- | --- | --- |
| `off` | Original proposal | `checked: false`, no changes; feasibility fields are null |
| `observe` | Original proposal | Suggested changes with `applied: false`; feasibility of both original and suggested result |
| `intervene` | Corrected result | Actual changes with `applied: true`, and any unresolved aggregate excess |

`proposed_actions` always contains the raw model values. `actions` contains the
values selected by the mode. `safety.changes` has one record per net changed action:
action name, channel, equipment (or explicit null), unit, before/after values,
application flag, rule IDs and reason. No-op corrections are omitted.

`proposed_feasible`, `suggested_feasible` and `final_feasible` refer only to the
configured scalar bounds, precision grid and supported aggregate equation. In
observe mode, a feasible suggestion does not imply the unchanged final proposal
is feasible. `unresolved_constraints` describes residuals **after the suggested
correction**, with rule/constraint, unit, value, upper bound, excess and reason.
The context includes installation, configuration, bindings, bundle/hash, adapter,
safety-policy and service revisions. This is a calculation against supplied data,
not proof of measured actuation or equipment availability.

## Supported physical profile

1. Every action is a scalar with finite lower and upper bounds within ±1e9 in its
   declared unit. Precision is 0–6 decimal places, using a grid anchored at zero.
   At least one grid point must lie within the bounds.
2. Each action is rounded to its nearest grid point (ties away from zero), then
   clamped to the innermost representable lower/upper bounds. A non-grid physical
   boundary is never exceeded because of rounding.
3. The installation may declare **one** linear aggregate upper limit, or none.
   Every controlled term must have a positive coefficient and be listed exactly
   once in the priority. Other terms must be bound scalar instantaneous
   observations with both physical bounds declared. Negative observation
   coefficients are allowed, for example an explicitly signed generation term.
   All terms share the aggregate unit, as required by the installation contract.
4. Preparation rejects an aggregate that is impossible even with the best
   observation bounds and all controlled actions at their representable minima.
   Multiple aggregates, aggregate lower limits, negative controlled coefficients,
   forecast terms, missing coverage or ambiguous priority are unsupported.
5. At evaluation, reduce actions in priority order by enough whole grid steps to
   satisfy the budget. Visit each action at most once. Recheck action bounds,
   precision and aggregate total after correction; there is no convergence loop.
6. If the uncontrollable contribution makes the limit impossible, reduce all
   aggregate controls to their **configured representable minima** and report the
   remaining excess. Those minima may be nonzero. The required
   `minimum_actions_with_report` policy makes this fallback explicit; there is no
   universal zero command or reuse of a previous action.

Arithmetic uses exact rational values derived from the decimal representations
of the inputs. No positive tolerance silently permits exceeding the modeled upper
limit. Corrections are checked again for exact decimal preservation when converted
to JSON numbers. Unsupported numeric overflow/underflow produces a technical error,
not nonfinite JSON or a fabricated successful correction. Residual numbers are
serialized as finite floats; their feasibility flag is computed before that
serialization, so very small excesses can coexist with displayed totals that round
to the same number as the limit.

Observation bounds support static feasibility checks. This profile does not
replace or clip sensor readings at runtime, infer missing measurements, apply a
freshness threshold, or model equipment dynamics, ramp limits or per-phase physics.
It is not an optimizer for arbitrary interacting constraints. Configuration is
validated even in off mode so the selected profile remains unambiguous.

## Reproduce the two-charger example

The example is synthetic; its 11 kW limit and 1 kW minima are not hardware settings.
From the repository root, select a private local token and run:

```sh
export ADMIN_API_TOKEN='choose-a-local-token'
export ALLOWED_BUNDLE_ROOT="$PWD/examples/safety_demo"
export CONFIGURATION_STORE_DIR="$(mktemp -d)"
export INSTALLATION_SITE_ID=example-house
export CONFIGURATION_FILE="$PWD/examples/safety_demo/configuration.json"
export PREPARATION_REQUEST_FILE="$PWD/examples/safety_demo/prepare.json"
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Use an environment without a legacy `MODEL_MANIFEST_PATH`. Startup stores and prepares
the configuration through the same validation path. In another terminal with the
same token, retrieve a candidate receipt (this does not activate anything):

```sh
curl -sS -X POST http://127.0.0.1:8000/admin/v1/candidates \
  -H "Authorization: Bearer $ADMIN_API_TOKEN" \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: safety-demo' \
  --data-binary @examples/safety_demo/prepare.json
```

Read `/admin/v1/operations/{operation_id}` until succeeded, then
`/admin/v1/candidates/{resource_id}` and require `state: ready`. Use that candidate:

```sh
curl -sS -X POST http://127.0.0.1:8000/admin/v1/candidates/CANDIDATE_ID/evaluations \
  -H "Authorization: Bearer $ADMIN_API_TOKEN" -H 'Content-Type: application/json' \
  --data-binary @examples/safety_demo/request.json
```

The model proposes 6 + 6 kW. With base load 3 kW and upper limit 11 kW, the result is
6 + 2 kW: charger 2 is reduced first. `final_feasible` is true. Repeating with
`request-residual.json` gives base load 12 kW, actions 1 + 1 kW, a total of 14 kW,
residual excess 3 kW and `final_feasible: false`. Both evaluations return HTTP 200
and an auditable result. No command is sent to a device.

For observe/off, create another configuration revision and a preparation request
referencing it. Reusing an immutable revision with a different mode is rejected.
Container mounts remain the same: bundle input in read-only storage, private
persistent state in writable storage; see [message execution](message-execution.md).

## Verification and remaining integration

`tests/test_safety.py` covers modes, scalar boundaries, negative values, half-step
rounding, unequal coefficients, explicit priority, nonzero fallback minima and
60 combinations of proposals/observations checked against independent invariants.
It also covers authenticated real API evaluation, ONNX, units before normalization,
original-message preservation, runtime mutation isolation, unsupported preparation
profiles, technical failures, startup files, restart and audit contract consistency.
The committed two-charger example is exercised through the actual evaluator.

Export/check the three additional schemas with:

```sh
.venv/bin/python scripts/export_safety_schemas.py --check
.venv/bin/python scripts/export_contract_schemas.py --check
.venv/bin/python -m pytest tests/test_safety.py -q
```

Verification on 2026-10-27: **358 passed, 4 skipped**, including **44 new safety
tests**. The four skips and the multipart deprecation warning predate this change.

Independent Draft 7 validation also covers all three policy modes and six real
API responses (ordinary and residual cases). JSON Schema validates structure;
Pydantic and preparation additionally enforce cross-field and physical invariants.

[Iteration 6](local-lifecycle.md) now implements activation/recovery and reuse of
a loaded runtime. The
specialized RH1/iCharging strategies still need preparation capability adapters.
The current Percepta consumer does not yet preserve this safety envelope in Mongo;
that consumer integration remains explicit iteration 7 work.
