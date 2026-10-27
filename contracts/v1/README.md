# Draft contracts for inference configuration and decisions

These contracts define the integration boundary. Installation context and safety audit have concrete schemas and
semantic validation. The algorithm interface is a **provisional sidecar**, not a
replacement for `artifact_manifest.json` or an agreed Algorithms export format.
Each document explicitly carries its draft version so later changes are visible.

The current Percepta/Mongo message remains the input. No schema revision or
installation revision is invented on behalf of the producer. The original iteration
2 introduced documents only. [Iteration 3](../../docs/candidate-preparation.md) now
uses configuration and compatibility documents in an opt-in administrative API;
it does not fetch backend data or activate a configuration. [Iteration 5](../../docs/safety-correction.md)
adds safety correction using explicit v2 configuration and evaluation documents;
these v1 schemas remain unchanged.

## Ownership and separation

| Document | Producer | Meaning |
| --- | --- | --- |
| `installation.v1-draft` | Runtime compiler using Community inventory/catalogues | Reduced snapshot of one site, equipment, channels, meanings, physical constraints and declared access |
| `bundle-interface.v1-draft` | Provisional adapter around an export; eventually Algorithms | Algorithm input/output names and order, semantics, target site, runtime and topology capability |
| `configuration.v1-draft` | Runtime | Installation, bundle interface, explicit bindings, input profile and safety policy selected together |
| `safety-policy.v1-draft` | Runtime configuration | Mode, profile and rule identifiers supported by the inference implementation |
| `decision.v1-draft` | Inference | Proposal, delivered actions, activation identity and explanation of safety changes |
| `compatibility.v1-draft` | Candidate preparation/evaluation use case | What was checked, unresolved issues and required changes |

Source schemas are in `app/contracts/v1`. JSON Schemas are in `schemas/` and
examples in `examples/`. The core application does not import these boundary
models. The preparation service uses them at its own boundary.

## Installation context

`site_id` identifies the installation, independent of container location or
display name. `revision` identifies an immutable snapshot. `sources` records the
systems, references and revisions from which Runtime compiled it. This is not a
connection configuration: credentials, URLs for device access and SSH details
have no place in this document.

Each equipment entry has a stable ID, kind, optional label, phases and typed
numeric technical values. The initial technical-value representation supports
scalar facts such as capacity and rated power; curves and richer equipment models
require an explicit schema extension. A missing fact stays missing.

Each channel describes:

- A stable channel ID and owning equipment, or a null equipment ID for a channel
  resolved at site scope, including externally supplied community inputs.
- Its role: observation, forecast or action.
- Quantity, unit, value type, temporal semantics, positive direction and interval
  where relevant. Units and meanings are compared exactly; no implicit scaling
  or conversion between energy and power occurs in these contracts.
- Whether it is enabled and whether access is declared as read, write, both,
  unavailable or unknown. Declared access is not a credential, live connectivity
  check or proof of authorization. Unknown must remain explicit.
- For inputs, the path below `features` as a list of exact object keys, for example
  `["observations", "charging_sessions", "EVC01", "power"]`. Dots in IDs are not
  path separators. The structural path avoids confusing a database ID with an
  algorithm feature name. Null means an unresolved mapping, requiring preparation.
- Optional physical bounds, in the channel's unit. Bounds apply componentwise to
  numeric vectors. Omit the bounds object when limits are unknown.

Aggregate limits express a weighted sum of scalar channels in a common unit.
Coefficients are dimensionless and signed; a referenced observation can represent
base demand while an action represents controllable power. This can describe
site or phase constraints. Checking timing alignment, physical feasibility,
coverage and the installed safety profile's ability to enforce them belongs to
candidate preparation and safety execution.

Backend fields that support compilation already include asset identity and
technical characteristics, capability ID, quantity, unit, value semantics,
collection role and parameter/entity keys. The mapping is a compilation step:
the backend's review snapshot marked `executable: false` is **not** this runnable
context, and the current Percepta collection compiler does not prove actuation
access. No backend compiler was changed in this iteration.

## Provisional bundle and explicit binding

The sidecar describes one selected agent for one target site. `inputs` and
`outputs` are ordered algorithm interface declarations. `bindings` separately
associate those logical names with installation channel IDs. A display-name
change does not reorder inputs; adding a key to a message does not add a model
dimension or action.

The sidecar requires the target site, selected agent and a declared runtime.
`topology_support` is fixed, dynamic or unspecified. Dynamic is a capability
claim to validate, not permission to accept arbitrary equipment. A missing
manifest hash is retained as missing; it is not manufactured from a path.
Preparation must verify the actual manifest and every artifact. A manifest hash
alone does not verify artifact bytes.

The schema rejects cross-site pairing, missing/duplicate bindings and differences
in units, meaning, direction, interval or type. Draft v1 has no implicit conversion
language. A mismatch needs a reviewed adapter/schema change or corrected source
metadata. It is not repaired by guessing a multiplier.

Decision cadence still comes from the selected runtime's artifact configuration
in this draft. A channel's measurement/forecast interval is not automatically the
decision interval. Preparation must resolve and validate both; a common explicit
control-period field is part of the exporter/runtime alignment still to close.

Passing these checks does **not** prove runtime support, access, complete control
coverage, safety feasibility or physical compatibility. The exact exporter
fields, entity-based tensor shapes, runtime capability declarations and artifact
identity mechanism remain alignment points with Algorithms. They can evolve in
a new explicitly supported version without changing the installation identity.

## Current Percepta input

The configured profile is `percepta.mongo-message.v0`. This is a name assigned by
the inference integration to the current producer format, not a version present
in messages. `adapter_revision` belongs to the serving implementation;
`producer_revision` remains null unless an actual producer revision is known.

`POST /inference` continues to receive `{"features": message}` with optional
`agent_index`. The message contains observations, forecasts and timestamp, and
may contain community values. Percepta converts its timestamp to an ISO string
before posting. Existing stored examples can also use Mongo-style dates, as
supported by existing adapters. New measurements and forecasts are request data,
not installation revisions.

The real example in `examples/legacy-request.json` is the first record of
`examples/rh1_bundle/exemplos_mensagem_R-H-01_2303.json`, copied without field
changes. It is a repository sample, not a newly queried live Mongo document.
Other examples here are synthetic and do not assert actual equipment limits.

For future adaptation, configured channel paths select the required inputs.
Mongo `_id`, saved `decisions`, diagnostics and activation metadata are not model
features unless a separate explicit supported contract requires them. Missing
required inputs produce a named technical error; unsupported fields do not create
equipment. A site-like field inside `features` never chooses the installation.
The currently deployed route's permissive handling is unchanged in this iteration.

The [iteration 4 `channels-1` adapter](../../docs/message-execution.md) now implements
this selected-input path through candidate evaluation. It also supports canonical
array-index components when traversing arrays, while object components remain
literal keys. The original `legacy-1` profile is not silently migrated. Proposals
use the additive `evaluation.v1-draft` response, without pretending to have an
activation identity or delivered safety-corrected actions.

## Safety result independent of Percepta

The result describes one agent's `proposed_actions` and final `actions`. The
activation reference identifies site, configuration, installation, bindings,
bundle, input adapter, safety policy and service versions separately. Versions
do not need to be equal; the activation records their validated combination.

| Mode | Delivered actions | Audit |
| --- | --- | --- |
| `off` | Original proposal | No corrections |
| `observe` | Original proposal | Suggested corrections with `applied: false` |
| `intervene` | Corrected actions | Actual corrections with `applied: true` |

A correction records action name, equipment/channel IDs, unit, original and
corrected values, rule IDs and reason. One consolidated change per action records
all contributing rules. In observation mode `corrected_value` is a suggestion;
the delivered value remains the proposal. Untouched actions remain identical.
Schema validation rejects unreported changes, incorrect original values and
mode/applied inconsistencies. It does not certify physical feasibility or check
IDs against a live installation; execution must verify them against the captured
activation/context and its safety policy.

A physical-limit violation triggers correction and explanation, not a safety
veto. Original measurements are preserved. Malformed data, nonfinite outputs or
runtime failures are technical errors, not successful decisions containing zeros
or fabricated safety reports. The correction engine, fallback policy and handling
of the supported aggregate constraint are implemented and tested in [iteration 5](../../docs/safety-correction.md),
using a new evaluation envelope without an invented activation identity.

`DecisionResult.legacy_response()` explicitly projects final actions to the
existing `actions[agent_index]` shape. It loses the audit. The inspected Percepta
consumer currently extracts only `actions["0"]`, forwards them and saves them
under `decisions`. It does not persist new top-level safety fields. Until that
consumer is updated, do not claim end-to-end storage of the safety report; neither
the old response nor the producer was changed here.

## Planned administrative API

These names and semantics were the draft target for iterations 3 and 6. Configuration,
candidate and operation routes are now implemented; activation and active-state
routes remain planned. The executable preparation request uses the separate
`preparation.v2-draft` schema to include local bundle addressing; the v1 request
below is retained as the original draft, not accepted by the candidate route.
See the [implemented API and exact states](../../docs/candidate-preparation.md).
Admin calls require a trusted
authenticated Runtime caller; the inference data path does not receive backend
credentials or fetch backend APIs for every decision.

| Operation | Request | Response |
| --- | --- | --- |
| `PUT /admin/v1/configurations/{revision}` | `ConfigurationDraft`, matching path revision | `201` new or `200` identical existing revision; storing does not activate |
| `GET /admin/v1/configurations/{revision}` | None | `200` stored draft, `404` missing |
| `POST /admin/v1/candidates` | `PreparationRequest` plus `Idempotency-Key` | `202 OperationReceipt`; load/validate candidate without replacing active |
| `GET /admin/v1/candidates/{candidate_id}` | None | `200` candidate ID, configuration revision, state (`preparing`, `ready`, `rejected`) and compatibility report when assessed |
| `POST /admin/v1/activations` | `ActivationRequest` plus `Idempotency-Key` | `202 OperationReceipt`; switch only from expected active ID to a ready candidate |
| `GET /admin/v1/active` | None | `200` activation reference or null when unconfigured |
| `GET /admin/v1/operations/{operation_id}` | None | `200 OperationReceipt`; status pending, running, succeeded, failed or uncertain |

Requests with bad syntax/contract use `422`; authentication/authorization failures
use `401`/`403`; unknown references use `404`; revision reuse with different
content, stale activation preconditions or a non-ready candidate use `409`;
temporary preparation infrastructure failure uses `503`. Errors use
`ContractError` with code, document path and detail. Asynchronous failures record
the same error information with the operation. No error replaces the active
record. A timeout is not proof of failure: retry/query using the same idempotency
key/operation ID. `expected_activation_id` is required; explicit null means the
caller expects no active configuration.

Existing inference/info/health payloads retain their contract. Iteration 3 adds
optional bearer authentication to legacy admin routes and disables direct loading
and unloading when candidate preparation is enabled.

## Composition change matrix

| Change | Required assessment and possible outcome |
| --- | --- |
| Label only | Confirm IDs, bindings and semantics unchanged; configuration revision may suffice |
| Extra observation not consumed | Check combined constraints/coverage; leave existing input order unchanged |
| Second controllable charger | Verify observation/action/binding/safety coverage; fixed bundle may require a new export and validation |
| Removed channel or charger | Reject dangling bindings; do not fill fixed model inputs with invented zeros |
| Same dimension but changed unit, direction or meaning | Schema binding check detects mismatch; review adapter/bundle requirements |
| Different message structure | Update configured mapping if supported, otherwise adapter/service version |
| Physical limit or access change | Revalidate policy feasibility and access; no assumption that the algorithm remains suitable |
| Unknown semantics or unavailable evidence | `insufficient_information`, never implicit compatibility |

The one/two-charger examples retain old IDs. The second installation alone is
structurally valid, while the old bundle still has only one output. The example
impact report requires a new bundle/binding for that scenario. It is an expected
assessment fixture, **not output of an implemented compatibility evaluator**.
The implemented evaluator is described in the iteration 3 guide; automatic retraining or coordinated deployment of
Percepta and Algorithms is outside the inference service.

## Validation and evolution

```sh
python scripts/validate_contract.py configuration contracts/v1/examples/configuration.json
python scripts/validate_contract.py decision contracts/v1/examples/decision-intervene.json
python scripts/export_contract_schemas.py --check
python -m pytest -q tests/test_contracts_v1.py
```

JSON Schema validates structure and scalar restrictions. Cross-field rules such
as binding references, ordering of bounds, safety consistency and identity
uniqueness are implemented by the Python models and must also be enforced by
other consumers. The CLI applies both layers. Successful contract validation is
explicitly not an activation check. The schema exporter preserves nullable fields
that Pydantic 1 otherwise omits from JSON Schema.

Unknown document fields and unsupported schema versions are rejected. A new field
or interpretation requires an explicit supported draft revision/migration;
changing a meaning while keeping its old version is not a compatible update.
Schemas, examples and semantic tests move together. No contract file changes the
running system until its adapter and lifecycle integration are installed.

Verification on 27 October 2026: **204 tests passed, 4 skipped**, including 59
new contract tests and the existing dataset replays. The same pre-existing
`python_multipart` warning remains. All ten exported schemas and their example
documents were also checked with an independent JSON Schema Draft 7 validator;
explicit null activation preconditions were included. Schema export consistency,
syntax compilation and whitespace checks passed. The running OpenAPI document
is unchanged. This verifies the draft contracts, not deployment or physical safety.

Local sources checked on 27 October 2026: Percepta `bab86ac`,
`app/predictor.py`; Community/Runtime backend `9f53d54`,
`Community_Infrastructure_Context` asset/channel models and
`Runtime_Management_Context/app/configuration.py` and `percepta/compiler.py`.
These sources establish the producer and available inventory fields, not agreement
on a future exporter contract.
