# Inference Serving API

## Purpose
Serve inference decisions for any algorithm (rule-based policies, ONNX models, or custom runtimes) behind a stable HTTP API. Bundles configure the runtime so models and policies can be swapped without code changes.

## Design Decisions
- Bundle-driven configuration: a manifest plus artifacts define features, encoders, actions, and runtime behavior.
- Stateless requests: each `/inference` call is self-contained.
- Single pipeline: one execution path for both rule-based and model-backed inference.
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
- `POST /inference` – payload `{"features": {...}}` → `{"actions": {"0": {...}}, "request_id": "..."}`
- `GET /info` – manifest metadata for the loaded bundle.
- `GET /health` – readiness and environment checks (includes GPU availability).
- `POST /admin/load` – load a bundle by path.
- `POST /admin/unload` – unload the current bundle.

## Payload Contract
- Wrap input in `{"features": {...}}`.
- Nested objects are flattened using dot notation: `a.b.c`.
- Arrays use indices: `list[0]`, `list[1]`.
- Aliases can map external field names to internal ones.

Example:
```json
{
  "features": {
    "timestamp": "2026-01-01T12:00:00Z",
    "sensor": {"temperature": 22.4},
    "meters": [{"power": 1.2}]
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
```bash
export BUNDLE_PATH="$PWD/examples"
export MODEL_MANIFEST_PATH="/data/your_bundle/artifact_manifest.json"
export MODEL_AGENT_INDEX=0
export FEATURE_ALIAS_PATH="/data/your_bundle/aliases.json"  # optional

docker compose up --build
```

## Environment Variables
- `MODEL_MANIFEST_PATH` (required to auto-load on startup)
- `MODEL_ARTIFACTS_DIR` (optional base path for artifacts)
- `MODEL_AGENT_INDEX` (agent to serve)
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
pytest tests/test_api_inference.py
```
The test suite covers pipeline loading, API inference behavior, and example bundle scenarios. Use `-k` to run a focused test.

## Postman
Collection: `postman/EnergyFlexibilityInference.postman_collection.json`.
- Includes sample requests for loading bundles and running inference.
- Supports `x-request-id` for tracing.

## Observability
- Console logs by default (optionally JSON).
- Optional file logging via `LOG_FILE`.
- Request IDs are accepted via `x-request-id` or generated automatically.
