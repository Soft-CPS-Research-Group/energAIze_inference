# Local configuration and candidate preparation

Iteration 3 adds an opt-in administrative path for one installation. Runtime
delivers a reduced installation context and a reference to a local algorithm
bundle. Inference stores the configuration, assesses compatibility, snapshots
the files and constructs the selected runtime in isolation. It never publishes
that runtime to the active inference store.

This is local preparation, not deployment or activation. There is no backend
query, equipment discovery, automatic retraining or device command. The exporter
interface remains provisional. [Version 1 contracts](../contracts/v1/README.md)
still define configuration and compatibility; the executable preparation request
is explicitly **`preparation.v2-draft`**, because the earlier request lacked a
local bundle reference. Its [schema](../contracts/v2/preparation.schema.json)
does not replace the original v1 schema.

## Enable the mode

| Setting | Meaning |
| --- | --- |
| `CONFIGURATION_STORE_DIR` | Persistent private directory for SQLite revisions, operation receipts and candidate snapshots; enables this mode |
| `INSTALLATION_SITE_ID` | Operator-selected site identity, pinned to the store on first use |
| `ALLOWED_BUNDLE_ROOT` | Existing approved directory containing delivered bundles |
| `ADMIN_API_TOKEN` | Required bearer secret of at least 32 characters; supply through the deployment secret mechanism |
| `CONFIGURATION_FILE` | Optional configuration JSON to store at startup |
| `PREPARATION_REQUEST_FILE` | Optional v2 request to prepare at startup, after storing configuration |
| `MAX_CANDIDATE_BYTES` | Total captured-file byte limit per candidate; default 512 MiB |
| `ONNX_EXECUTION_PROVIDERS` | Requested ONNX providers; unavailable providers or silent fallback reject preparation |

Run one worker per store. A Linux process lock prevents two owners from recovering
or preparing against the same directory. The directory is bound to one site and
cannot be reused for another site merely by changing an environment variable.
Mount state on persistent storage and source bundles read-only. The process
creates private directories; the operator must also protect pre-existing paths.
Use TLS at the gateway and restrict administrative network access. Bearer
authentication does not implement multiple identities or role-based permissions.

`MODEL_MANIFEST_PATH` is forbidden in this mode. `/admin/load` and `/admin/unload`
return `409` after authentication, so they cannot bypass preparation. Without
`CONFIGURATION_STORE_DIR`, legacy loading remains available; setting
`ADMIN_API_TOKEN` also protects those legacy administrative routes.

## Administrative requests

All routes below require `Authorization: Bearer <ADMIN_API_TOKEN>`.

| Route | Result |
| --- | --- |
| `PUT /admin/v1/configurations/{revision}` | `201` for a new configuration, `200` for an identical existing document; `activated: false` |
| `GET /admin/v1/configurations/{revision}` | The stored `configuration.v1-draft` document |
| `POST /admin/v1/candidates` | `202` operation receipt; requires `Idempotency-Key` |
| `GET /admin/v1/operations/{id}` | Durable `pending`, `running`, `succeeded`, `failed` or `uncertain` receipt |
| `GET /admin/v1/candidates/{id}` | Candidate state, configuration revision, compatibility, impact, captured hashes and provider evidence |
| `GET /admin/v1/status` | Site, configuration/completed-candidate counts, latest submitted operation ID/status (or null), and `activation_supported: false` |

Example preparation request:

```json
{
  "schema_version": "preparation.v2-draft",
  "configuration_revision": "configuration-1",
  "bundle": {
    "directory": "bundle",
    "manifest_file": "artifact_manifest.json",
    "file_hashes": {}
  },
  "baseline_configuration_revision": null
}
```

`directory` is relative to `ALLOWED_BUNDLE_ROOT`; manifest, aliases, artifacts,
config files and file-hash keys are relative to that bundle directory. ONNX
external-data locations are relative to their model file. Paths must be canonical
POSIX paths. Absolute paths, parent traversal, symlinks and non-regular files are
rejected, including symlinks that point inside the root. The local runtime
performs these filesystem checks in addition to JSON Schema validation.

Configuration path/body revisions must match and belong to the operator's site.
A configuration revision cannot be overwritten with different content (`409`).
The entire document is immutable under that root revision; individual nested
component revision labels are not separate storage keys. Producers should change
their labels when their content changes; impact assessment compares actual content.

An identical idempotency key and canonical request returns the original operation,
including after restart or source-file changes. A different request with that
key returns `409`. To intentionally re-evaluate files, submit a new key. Unknown
references return `404`; invalid request contracts return `422`; absent/wrong
credentials return `401`; the administrative mode being disabled returns `503`.
Domain errors carry `detail: {code, path, detail}`. FastAPI syntax/validation errors
retain its standard validation response.

Preparation runs outside the event loop using the application thread pool. An
operation can succeed with a **rejected** candidate: the evaluation completed and
found incompatibility. `failed` denotes an unexpected technical failure. Completed
candidates contain `compatible`, `incompatible` or `insufficient_information`
reports with checked stages and localized issues. Missing knowledge never produces
a ready candidate. Pending/running candidates have no compatibility report yet.

## What is checked

- Configuration/bundle/site identity, explicit input/output bindings, matching
  declared units, direction, temporal semantics and value types; required channel
  access and physical action bounds; coverage of enabled action channels.
- Explicit manifest version 1, selected agent, artifact count when declared,
  runtime format, feature/action names and order, encoder parameters and dimensions.
- The expected manifest SHA-256 from configuration and every supplied file hash.
  All referenced artifacts, aliases, config files and ONNX external tensor files
  are captured before constructing a native runtime. Observed hashes cover every
  captured file; an observed hash alone is not proof of publisher authenticity.
  Supply all expected file hashes when artifact integrity must be verified against
  the producer, rather than only recorded for this snapshot.
- Construction from captured files, requested/actual providers, and actual ONNX
  input/output tensor signatures. Sessions are released after validation; the
  candidate persists evidence and files, not a live engine.
- Compatibility with the implemented message and safety capabilities, rather than
  just accepting a well-formed declaration of an unsupported capability.

The optional baseline is an explicitly selected **stored configuration**, not an
active-generation precondition. There is no active configuration in this new
lifecycle yet. Impact lists changed sections, added/removed/changed channels,
semantic, equipment and constraint changes. It classifies the need for
configuration, bundle or adapter review. These classifications are guidance;
the compatibility report determines whether the candidate is ready. Changing
declared algorithm semantics/order while retaining the same manifest hash is
rejected pending reviewed bundle evidence. The service cannot independently prove
the physical meaning of a model from tensor dimensions or a sidecar declaration.

## Supported preparation profiles and limits

The initial profile supports fixed flat interfaces with `legacy-1` message
adaptation. A bound input's message path must resolve through the existing
flattening/envelope/alias behavior. Literal dots inside path components and
vector inputs need a future adapter. With the observations envelope enabled,
only observations and the existing forwarded community/forecast fields can be
bound by the current channel schema. Output names must currently equal the bound equipment IDs; general
channel-to-action routing belongs to iteration 4.

- **ONNX:** one float32 input and output, rank two, fixed feature/action width,
  batch dimension 1 or an accepted dynamic batch marker; supported scalar encoders;
  complete finite bounds within installation limits. Bounds must align to tenths
  until the legacy post-processing rounding fix in iteration 4.
- **Simple rules:** JSON `default_actions` and equality `rules`, with explicit
  finite scalar values for every action in every branch. Conditions use declared
  input names; encoded rule conditions are not supported. Values must fit the
  installation bounds.
- **Safety:** only explicit mode `off`, profile `off.v1`, empty rules. Observation
  and intervention are rejected as unsupported until iteration 5 implements them.

Specialized RH1/iCharging strategies, dynamic/entity interfaces and unknown runtime
configuration fields are rejected by this new path until their capability adapters
are implemented. Their legacy loading behavior is preserved. A ready candidate
does not certify model accuracy, real device access, semantic truth, joint physical
feasibility or successful inference for every possible message. Aggregate limits
have structural/reference checks only; they are not enforced by `off.v1`.

This configuration rejection is separate from the future safety layer's operational
behavior: that layer will correct and report physical violations, not veto decisions.

## Reproducible local example

[The demo](../examples/preparation_demo/configuration.json) contains synthetic
installation data, a simple rule bundle and matching hashes. From the repository
root, with dependencies installed:

```sh
unset MODEL_MANIFEST_PATH
export INSTALLATION_SITE_ID=example-house
export ALLOWED_BUNDLE_ROOT="$PWD/examples/preparation_demo"
export CONFIGURATION_STORE_DIR="$PWD/.local-preparation-demo"
export CONFIGURATION_FILE="$ALLOWED_BUNDLE_ROOT/configuration.json"
export PREPARATION_REQUEST_FILE="$ALLOWED_BUNDLE_ROOT/prepare.json"
export ADMIN_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

The settings also read `.env`; remove any legacy `MODEL_MANIFEST_PATH` entry there
before using this example. Keep the generated token out of version control.
The same demo configuration and preparation JSON can instead be submitted via
the PUT/POST routes with the two startup-file settings unset. Both paths call
the same service and persistence operations. The test suite exercises this demo.

Startup may have no configurations, stored configurations, or a ready/rejected
candidate. `/health` remains `configured: false` in a fresh preparation-only
instance and `/inference` remains unavailable, because no activation occurred.
Use authenticated status and candidate/operation reads to inspect preparation;
`latest_operation.operation_id` also identifies the startup candidate when one exists.
Startup file requests use a deterministic idempotency key; changing only source
bytes does not silently trigger re-preparation.

On restart, unfinished operations become `uncertain` with
`preparation_interrupted`; they are not automatically retried. Inspect the receipt
and use a new key to revalidate. Completed evidence and snapshots remain readable.
Any future activation must revalidate/reload; every candidate records
`requires_revalidation_before_activation: true`.

## Next integration boundaries

Iteration 4 implements explicit Percepta/channel adaptation and execution error
handling. Iteration 5 implements supported safety correction profiles. Iteration 6
adds bounded scheduling, activation preconditions, atomic local publication and
recovery of an active combination. This iteration has per-candidate size limits,
but no total storage quota, retention policy or deployment-wide job scheduler.
The Runtime context compiler, Algorithms exports, Percepta audit persistence and
real deployment remain external integration work.

Code is separated into HTTP translation (`app/routers/preparation.py`), preparation
orchestration, compatibility assessment, filesystem capture and transactional
storage (`app/preparation/`). The active application service and inference route
do not depend on this preparation service.

Validate schemas and the local feature with:

```sh
python scripts/export_contract_schemas.py --check
python scripts/export_preparation_schema.py --check
python -m pytest -q tests/test_preparation.py
```

Delivery verification (2026-10-27): full suite **252 passed, 4 skipped**, including
48 preparation tests and the existing inference/replay regressions. Both schema
export checks, Python compilation and independent Draft 7 validation of the demo
configuration/request passed. Verification was local with CPU ONNX Runtime; no
device deployment or GPU execution is claimed.
