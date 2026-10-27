# Provisional Percepta adaptation and candidate evaluation

[Iteration 6](local-lifecycle.md) now adds a reusable active-generation execution
path with explicit activation/context references and restart recovery. Candidate
evaluation described here remains separate.

Iteration 4 uses the existing Percepta/Mongo message as a provisional input
format. It introduces `channels-1`, an explicit adapter for
`percepta.mongo-message.v0`. The producer does not need to add a version field:
the Runtime chooses the profile in configuration. A later producer format must
receive an explicitly supported adapter/profile instead of changing old semantics
silently. Message parsing is separated from algorithm execution and channel identity.

The legacy `/inference` request/response shape remains available. The new path
can be exercised through authenticated candidate evaluation before activation is
implemented. Evaluation produces an identified **proposal only**; it neither
changes the active runtime nor sends a physical command.

## From message to algorithm

Runtime supplies the installation channels, signal meanings and bindings.
`channels-1` selects each required input through its `message_path` below
`features`, then names it with the bound algorithm signal. The adapter never
recursively copies the message into features. History, `_id`, decisions, unrelated
observations, diagnostics and site-like fields do not become inputs or choose a
different installation. New keys do not create equipment.

Paths are arrays of exact JSON keys. A dot in an equipment ID remains a literal
dot. While traversing a JSON array, a canonical nonnegative decimal component
selects its zero-based element; while traversing an object, it selects the literal
key. For example:

```json
["forecasts", "ConsumptionForecastService", "consumption_total", "values", "0"]
```

This behavior belongs to the new adapter revision; the legacy flattening profile
is not reinterpreted. Alias files must be migrated into these explicit bindings
and removed from a `channels-1` bundle. Preparation rejects conflicting aliases,
rather than applying alias resolution twice or guessing which source wins.
The new adapter reads directly below `features`; it does not repeat the legacy
`require_observations_envelope` transformation or flattening inside the engine.

Required inputs must be present and have the declared scalar type. Null, numeric
strings supplied for numeric signals, booleans supplied as numbers and nonfinite
values are errors. Unknown one-hot categories are errors, not all-zero vectors.
Missing required forecasts fail explicitly; unrelated or unconsumed forecasts do
not affect evaluation. Vector/tensor entity interfaces still require their own
capability adapter.

Scalar observations in the current producer format have no per-value unit label.
Their units come from the declared channel context, with `unit_source:
configuration` in the evidence. This is an assumption supplied by Runtime, not a
measurement independently verified by inference.

For selected forecast `values`, the adapter checks the containing descriptor's
`measurement_unit`, `frequency_seconds` and `horizon_seconds` when present:

- Matching units pass unchanged. Explicit power scaling between W/kW/MW and
  energy scaling between Wh/kWh/MWh are supported and reported. Energy-to-power,
  sign reversal, currency and arbitrary unit conversions are not guessed.
- A declared frequency must be finite, positive and match the configured interval
  when one exists. A horizon must be positive; if the message also declares the
  frequency, horizon must equal frequency times the values-array length.
- Missing unit/frequency metadata is distinguished from evidence present in the
  message. It uses the explicit channel declaration where available.

The adapter applies supported unit conversion once. ONNX then applies the manifest
encoder once. Simple equality policies consume the selected canonical values
without encoding; preparation rejects other encoder declarations for that profile.

An optional timestamp accepts a timezone-qualified ISO string or Mongo `$date`
(including epoch milliseconds/`$numberLong`) and is normalized to UTC as evidence.
Malformed timestamps fail explicitly. A missing timestamp remains unknown. There
is no invented maximum age, synchronization guarantee, interpolation or imputation;
freshness is `unknown` or `not_assessed`, never implicitly certified as fresh.

## Proposals and execution failures

The captured context fixes site, configuration, installation, bindings, bundle,
selected agent and adapter revisions for the whole evaluation. It is compiled
into immutable data detached from mutable boundary documents. The legacy active
inference service also captures its record once per request, so an overlapping
replacement cannot change its selected runtime halfway through execution.

Each proposal includes algorithm action name, channel ID, optional equipment ID,
unit and proposed value. Equipment IDs no longer have to equal algorithm action
names in `channels-1`; site-owned channels can use a null equipment ID. Units are
declared by the bundle/channel contract, not inferred from output tensor shape.

ONNX exposes its raw output proposal before the old clamping/flooring presentation.
Rule engines expose their own proposed actions. Common evaluation checks require
exact output names and finite numeric values. A physical limit violation in a
proposal does **not** trigger a safety veto: the proposal is returned unchanged
for inspection and the future safety stage. It is not a delivered command or a
claim that the value is safe to actuate. [Iteration 5](safety-correction.md) adds
explicit v2 configuration and evaluation with corrections; the v1 proposal-only
response described here remains unchanged.

The legacy ONNX `/inference` path still applies its configured clamp and decimal
flooring, now followed by a final clamp. Thus narrow, negative or non-decimal
bounds cannot be crossed by flooring. An infeasible decimal grid can result in
a boundary value with more than one decimal place. Candidate preparation no
longer rejects bounds merely because they do not align to tenths.

ONNX execution exceptions now return a technical error, never a successful zero
action map. Malformed/nonfinite outputs and incorrect output width are errors.
Input encoding failures are distinguished from native engine failures. This
intentionally changes error behavior while preserving successful reference cases.
Legacy specialized RBC arithmetic is unchanged; those strategies still need
preparation capability adapters before entering the new candidate path.

## Evaluate a prepared candidate

1. Store a configuration with `message.adapter_revision: channels-1`.
2. Prepare its local bundle through the iteration 3 API and require a ready candidate.
3. Submit `POST /admin/v1/candidates/{candidate_id}/evaluations`, with the same
   bearer authentication and a body containing `{"features": <Percepta message>}`.

The candidate selects the site and agent. An `agent_index` or site selector is
not accepted as an extra top-level evaluation request field. The producer message
itself remains untouched. The response carries `schema_version:
evaluation.v1-draft`, `proposal_only: true`, context, proposed actions, input
evidence, actual providers and request ID. It intentionally has no delivered
`actions` map and no invented activation ID. Schemas are in
[evaluation](../contracts/v1/schemas/evaluation.schema.json) and
[evaluation-request](../contracts/v1/schemas/evaluation-request.schema.json).

Responses: `200` proposal; `400` invalid consumed input; `401` authentication;
`404` unknown candidate; `409` candidate not ready or wrong adapter; `422` snapshot
or compatibility revalidation failure; `500` runtime construction/execution failure;
`503` preparation mode disabled. Errors contain `detail: {code, path, detail}`;
request schema errors retain standard FastAPI validation details. No failure
publishes a new active model or supplies substitute actions.

Each evaluation reloads the immutable configuration, verifies all captured file
hashes, makes a temporary isolated copy and constructs the runtime from it. It
does not reread the mutable delivery bundle. Temporary files are cleaned after
the operation. This is an administrative validation path, not a low-latency active
serving endpoint or a durable evaluation-results journal. Activation, engine
reuse and bounded scheduling remain iteration 6.

## Executable example

[The evaluation demo](../examples/evaluation_demo/configuration.json) binds the
repository message's battery SoC to `battery_soc` and translates `charger_dispatch`
to the charger setpoint channel. Its context and policy are synthetic; they do not
assert actual access, electrical limits or suitability for that house. The input
message is the existing [Percepta fixture](../contracts/v1/examples/legacy-request.json).
This sample's energy-like load values are deliberately not relabelled as power.

Enable preparation mode as described in the
[configuration guide](candidate-preparation.md), using
`examples/evaluation_demo` as the bundle root. Store `configuration.json`, prepare
`prepare.json`, then POST the existing legacy-request fixture to the evaluation
route. The reference response proposes **2.5 kW** for `charger-1-setpoint`, owned
by `EVC01`; it reports adapter `channels-1` and remains proposal-only.

## Filesystem and container mounts

The service needs two configurable local areas, regardless of whether the container
runs on an edge device or a cloud server:

| Area | Container example | Access/lifetime |
| --- | --- | --- |
| Delivered bundles | `/srv/inference/bundles` | Read-only; exporter/deployment supplies files here |
| Configuration, snapshots and temporary evaluation files | `/var/lib/inference` | Private, writable and persistent across restarts |

Host paths can differ at every destination. API bundle paths are relative to
`ALLOWED_BUNDLE_ROOT` **inside the container**; host absolute paths are not sent
in requests. State retains the site identity, immutable configurations and candidate
evidence. Losing that volume loses these records. Source bundles need not remain
available after their snapshot has been prepared, but the approved root directory
must still exist at startup. API-only configuration needs no third directory;
optional bootstrap JSON can live in an additional read-only configuration mount.
Logs can go to stdout, as in the supplied compose file.

[compose.preparation.yml](../compose.preparation.yml) is a standalone example with
explicit `BUNDLES_HOST_PATH`, `STATE_HOST_PATH`, `INSTALLATION_SITE_ID` and
`ADMIN_API_TOKEN` inputs. It uses one worker, mounts delivery read-only, and binds
the HTTP port to localhost. Create the host directories with permissions appropriate
to the container user. Administrative access beyond localhost requires the normal
TLS/gateway setup. It does not inherit the legacy compose file's auto-load setting.
`.dockerignore` excludes local secrets and development state from the image build.
Keep the persistent state directory outside the repository/build context; the
ignore file cannot identify an arbitrarily named state directory chosen by an operator.

The compose configuration was checked locally, without building an image, starting
a container or configuring hardware. The repository's current CUDA-based image
is retained; CPU image packaging, resource limits and actual destination deployment
remain integration work. No claim of Raspberry/Jetson/GPU validation follows from
these local CPU tests.

## Code boundaries and verification

`app/adapters/channel_message.py` owns this provisional message format and unit
conversion. `execution_context.py` compiles boundary contracts into immutable core
bindings. `app/application/evaluation.py` owns proposal execution and output checks.
The preparation service verifies and constructs the selected candidate; HTTP routes
translate requests/errors. Engines do not fetch house data or discover equipment.

The new tests cover the repository Percepta record, exact nested paths, array
selection, missing/invalid data, units/intervals/horizons, Mongo timestamps,
one-pass encoding, identified proposals, real ONNX sessions, native failures,
rounding boundaries, snapshot tampering, restart and concurrent record replacement.
The legacy inference/reference tests remain regression checks. Run:

```sh
python scripts/export_contract_schemas.py --check
python scripts/export_preparation_schema.py --check
python -m pytest -q tests/test_execution.py tests/test_preparation.py
```

Delivery verification (2026-10-27): **314 passed, 4 skipped** in the full suite,
including 60 new execution tests and two additional schema-export cases. Independent
Draft 7 validation passed for the demo configuration, request and actual evaluation
response. Schema checks, Python compilation, documentation links and Compose
configuration also passed. The existing four standby ONNX bundle tests remain
skipped; the new ONNX tests execute real CPU sessions. No image build or device
deployment was performed.
