# Inference Serving API

## Purpose
Serve inference decisions for any algorithm (rule-based policies, ONNX models, or custom runtimes) behind a stable HTTP API. Bundles configure the runtime so models and policies can be swapped without code changes.

## Design Decisions
- Bundle-driven configuration: a manifest plus artifacts define features, encoders, actions, and runtime behavior.
- Stateless requests: each `/inference` call is self-contained.
- Single deployment runtime: one execution path for both rule-based and model-backed inference, one agent decision per request.
- Flat feature keys: nested input is flattened into dot notation for consistent access.
- Observability-first: request IDs, structured logs, and optional file sink.
- Safe outputs: actions are clamped and rounded as configured by the bundle.

## Architecture
- HTTP API (FastAPI)
- Payload flattening and optional alias mapping
- Inference pipeline
  - Preprocessor (encoders)
  - Runtime (e.g., `onnx` or `rule_based` artifacts)
  - Post-processing (clamps/rounding)
- Response mapping (actions per agent)

## Bundle Format
A bundle is a directory with three groups of files:

- X: `artifact_manifest.json` (required)
- Y: artifact files (required, one per agent, e.g., `model.onnx` or `policy_agent_0.json`)
- Z: sidecar files (optional, e.g., `aliases.json`, extra configs referenced by the manifest)

Example layout:
```
bundle/
  artifact_manifest.json
  model.onnx
  policy_agent_0.json
  aliases.json
  assets/
```

`artifact_manifest.json` must define:
- `topology`: number of agents.
- `environment`: observation names, encoders, action names, and action bounds.
- `agent`: list of artifacts with `path`, `format`, and optional `config`.

Supported artifact formats:
- `onnx`: load ONNX Runtime session.
- `rule_based`: load JSON policy and optional config.

## Domain Model
- Bundle: a directory that contains the manifest and artifacts.
- Manifest: the configuration for the pipeline (environment, topology, artifacts).
- Agent: a logical inference target that produces an action vector.
- Artifact: a model or policy definition (ONNX or JSON rule set).
- Features: request input payload, flattened to dot keys.
- Actions: response output, keyed by agent index.
- Aliases: optional mapping to normalize feature names.

Diagram:
```
Bundle
  |-- Manifest
  |     |-- Environment (observations, encoders, actions)
  |     `-- Agents
  |          `-- Artifact(s)
  `-- Sidecars (aliases, extra config)

Request
  -> Flatten payload (dot keys, array indices)
  -> Apply feature aliases (optional)
  -> Preprocess (encoders)
  -> Runtime (onnx | rule_based)
  -> Post-process (clamp/round)
  -> Actions (per agent)
  -> Response
```
Explanation: every inference call is normalized into flat feature keys, optionally remapped via aliases, transformed by encoders, executed by the selected runtime, then post-processed before returning action values per agent.

## API
- `POST /admin/load` – load manifest and artifacts; can select default agent.
- `POST /admin/unload` – unload current runtime state.
- `POST /inference` – payload includes `features` and optional `agent_index`.
- `GET /info` – metadata for loaded runtime and default agent context.
- `GET /health` – service readiness and loaded-model diagnostics.

Full request/response contracts and error semantics:

- `docs/api_contract.md`

## Payload Contract
- Wrap input in `{"features": {...}}` and optionally include `agent_index`.
- Nested objects are flattened using dot notation: `a.b.c`.
- Arrays use indices: `list[0]`, `list[1]`.
- Aliases can map external field names to internal ones.
- Two request modes are supported:
  - Single-agent mode: `features` contains direct feature fields.
  - Community mode: `features.sites.<input_site_key>` is selected for the target agent.

Example:
```json
{
  "agent_index": 0,
  "features": {
    "timestamp": "2026-01-01T12:00:00Z",
    "sensor": {"temperature": 22.4},
    "meters": [{"power": 1.2}]
  }
}
```

Community example:
```json
{
  "agent_index": 1,
  "features": {
    "timestamp": "2026-02-22T12:00:00Z",
    "sites": {
      "boavista": {
        "non_shiftable_load": 7.0,
        "solar_generation": 2.0
      },
      "sao_mamede": {
        "site": {"pt_available_kw": 80.0},
        "solar_generation": 12.0
      }
    }
  }
}
```

## Running Locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Docker
Simple local run (CPU, no NVIDIA runtime required):
```bash
docker compose up --build
```

If your Docker daemon has NVIDIA as default runtime, force CPU runtime:
```bash
DOCKER_RUNTIME=runc docker compose up --build
```

Default startup bundle in `docker-compose.yml`:
- `MODEL_MANIFEST_PATH=/data/icharging_boavista_with_flex/artifact_manifest.json`
- `FEATURE_ALIAS_PATH=/data/icharging_boavista_with_flex/aliases.json`

Override for another bundle:
```bash
export MODEL_MANIFEST_PATH="/data/your_bundle/artifact_manifest.json"
export FEATURE_ALIAS_PATH="/data/your_bundle/aliases.json"  # optional
export MODEL_AGENT_INDEX=0
docker compose up --build
```

Optional GPU run (if NVIDIA runtime/driver is available):
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Environment Variables
- `MODEL_MANIFEST_PATH` (optional; if missing, load via `/admin/load`)
- `MODEL_ARTIFACTS_DIR` (optional base path for artifacts)
- `MODEL_AGENT_INDEX` (optional default agent index)
- `FEATURE_ALIAS_PATH` (optional aliases file)
- `LOG_LEVEL` (INFO/DEBUG)
- `LOG_JSON` (true/false)
- `LOG_FILE` (optional path for file logging)
- `LOG_FILE_ROTATION` (e.g., `50 MB` or `1 day`)
- `LOG_FILE_RETENTION` (e.g., `7 days`)
- `ONNX_EXECUTION_PROVIDERS` (e.g., `CPUExecutionProvider`)

## Logging
- Console logging is always enabled.
- `LOG_JSON=true` switches to JSON logs (single-line entries).
- `LOG_FILE` enables file logging in addition to console output.
- Rotation and retention are controlled by `LOG_FILE_ROTATION` and `LOG_FILE_RETENTION`.
- In Docker, mount a host folder to persist logs (e.g., `./logs:/var/log/energaize`).

## Tests
```bash
./.venv/bin/python -m pytest -q tests/test_api_inference.py
```
The test suite covers pipeline loading, API inference behavior, and example bundle scenarios. Use `-k` to run a focused test.

Bundle and multi-agent suite example:
```bash
./.venv/bin/python -m pytest -q \
  tests/test_multi_agent_store_api.py \
  tests/bundles/test_community_boavista_sao_mamede_bundle.py \
  tests/bundles/test_community_boavista_sao_mamede_rh1_bundle.py \
  tests/bundles/test_icharging_boavista_with_flex_bundle.py \
  tests/bundles/test_icharging_boavista_without_flex_bundle.py \
  tests/bundles/test_icharging_sao_mamede_bundle.py \
  tests/bundles/test_rh1_bundle.py
```

## Postman
Collection: `postman/EnergyFlexibilityInference.postman_collection.json`.
- Includes sample requests for loading bundles and running inference.
- Supports `x-request-id` for tracing.
- ONNX sample requests use the envelope contract: `features.timestamp + features.observations` (with optional `features.forecasts`).

## Observability
- Console logs by default (optionally JSON).
- Optional file logging via `LOG_FILE`.
- Request IDs are accepted via `x-request-id` or generated automatically.
