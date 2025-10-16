# Energy Flexibility Inference API

FastAPI service for running trained energy flexibility agents exported from the
training platform (artifact manifests + ONNX models). The service exposes
endpoints to perform inference, compute rewards, and inspect metadata for a
model bundle.

## Features

- Loads `artifact_manifest.json` and associated artefacts (ONNX or rule-based)
  produced by the training platform.
- Reconstructs preprocessing pipelines (normalisation, one-hot encoding, etc.)
  directly from manifest metadata.
- Serves a single agent per container (selectable via `MODEL_AGENT_INDEX`) to
  support edge deployments—start one container per building/agent. When no
  model is configured the service stays idle until `/admin/load` is called.
- Optional reward calculation endpoint mirroring the training reward logic
  (supports the baseline `RewardFunction` and `V2GPenaltyReward`).
- Health and information endpoints for deployment observability.

## Architecture

```
Clients (edge/fog) ---> FastAPI Routers (info/inference/reward/admin)
                                 |
                                 v
                       Pipeline Store (OTA load/unload)
                                 |
                                 v
                       InferencePipeline (preprocess + runtime)
                                 |
                                 v
                   Artefact Bundle (manifest + model per agent)
```

- **FastAPI routers** expose public endpoints as well as `/admin/*` for
  on-the-fly model management.
- **Pipeline store** keeps the currently active agent (if any) and metadata
  about when it was loaded.
- **InferencePipeline** reconstructs encoders, invokes the appropriate runtime
  (ONNX or rule-based), and maps outputs back to named actions.
- **Artefact bundles** are produced by the training platform and mounted or
  downloaded at runtime. One container is expected to host exactly one agent
  from the bundle.

## Getting Started

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set environment variables**
   - `MODEL_MANIFEST_PATH` (optional): path to `artifact_manifest.json`. If omitted,
     the service boots without a model and waits for `/admin/load`.
   - `MODEL_ARTIFACTS_DIR` (optional): root directory containing the manifest and
     artefacts. If set, the manifest path can be relative to this directory.
   - `MODEL_AGENT_INDEX` (optional): choose which agent from the manifest to serve
     in this container. Required when `MODEL_MANIFEST_PATH` is provided.
   - `LOG_LEVEL` (optional, default `INFO`): controls log verbosity (handled by
     Loguru).
   - `FEATURE_ALIAS_PATH` (optional): path to a JSON file with feature alias
     overrides per agent.

3. **Run the service**
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Test endpoints**
   - `GET /info`: model metadata (returns HTTP 503 until a model is configured)
   - `POST /inference`: supply feature dictionary and receive actions
   - `POST /reward`: optionally compute reward for given observations
   - `GET /health`: readiness probe (always 200; payload includes whether a model
     is configured)
   - `POST /admin/load`: load a manifest dynamically (`{"manifest_path": "...", "agent_index": 0}`)
   - `POST /admin/unload`: unload the current pipeline

## Tests

```bash
pytest
```

## Deployment

A simple Docker entrypoint is provided via FastAPI. For reproducible deployments:

1. Bundle the manifest and artefacts in a directory (per agent/run) or use
   MLflow to download them at startup.
2. Build an image that copies the bundle and sets environment variables, or
   mount the artefact directory into the container at runtime.
3. Run the container with Uvicorn/Gunicorn (one container per agent if serving
   at the edge).

### Rule-based Policies

The manifest can describe rule-based controllers by setting `format` to
`"rule_based"` and pointing `artifact.path` to a JSON definition:

```json
{
  "default_actions": {"action": 1.0},
  "rules": [
    {"if": {"mode": "idle"}, "actions": {"action": 0.0}}
  ]
}
```

Optional configuration can be provided via `artifact.config` (e.g.,
`{"use_preprocessor": true}` or `{"config_path": "policies/agent.json"}`).

The inference service instantiates an appropriate runtime automatically based on
`format`.

### Feature Aliases

If real-world feature names differ from the training manifest, provide a JSON
sidecar file referenced by `FEATURE_ALIAS_PATH`. The file must be a flat
mapping of `alias -> canonical` feature names (keys starting with `_` are
ignored so you can drop short comments). Example:

```json
{
  "temperature": "indoor_temperature",
  "hvac_kw": "hvac_power_kw"
}
```

Incoming payloads can use the alias names, which are converted to the manifest
feature names before preprocessing.

## Configuration

See `app/settings.py` for configurable environment variables.

- `MODEL_MANIFEST_PATH` (required): location of manifest.
- `MODEL_ARTIFACTS_DIR` (optional): base directory for relative paths.
- `LOG_LEVEL` (optional): log level for structured logs (default `INFO`).
- `LOG_JSON` (optional): when set to `true`, emit JSON-formatted logs.

## API Contract

Request/response models are defined in `app/models/requests.py` and
`app/models/responses.py`. Example inference payload (feature keys must match
the manifest `observation_names` for the configured agent):

```json
{
  "features": {
    "feat1": 0.2,
    "feat2": 0.5
  }
}
```

Reward payload:

```json
{
  "observations": {
    "net_electricity_consumption": 5.0,
    "mode": "idle"
  }
}
```

`GET /health` returns `{"status": "ok", "configured": true/false, "agent_index": idx}`.

## Tests & CI

Run the unit tests locally with `pytest`. The repository includes a GitHub
Actions workflow (`.github/workflows/ci.yml`) that executes
`python -m compileall` and `pytest` on pushes and pull requests—push your branch
to GitHub to trigger the pipeline. On pushes to `main` the workflow also builds
and pushes a Docker image (requires the `DOCKERHUB_USERNAME` and
`DOCKERHUB_TOKEN` secrets).

## Example Bundles & Manual Testing

Sample artefacts live under `examples/`:

- `examples/rule_based/`: minimal manifest + rule-based policy that drives the
  `hvac` action based on a `mode` feature. Use it to validate `/admin/load` and
  `/inference` without training assets. Includes `aliases.json` to demonstrate
  runtime feature renaming.
- `scripts/generate_identity_bundle.py`: helper that emits a one-feature ONNX
  identity model and companion manifest. Run
  `python scripts/generate_identity_bundle.py` to create
  `examples/identity_bundle/artifact_manifest.json`.

To try the API end-to-end:

1. Start the service (`uvicorn app.main:app --reload`).
2. Import `postman/EnergyFlexibilityInference.postman_collection.json` into
   Postman (or run with Newman). Update collection variables:
   - `rbcManifestPath` → rule-based manifest path inside the service (defaults to
     `/data/rule_based/artifact_manifest.json`).
   - `aliasMappingPath` → optional feature-alias JSON (defaults to
     `/data/rule_based/aliases.json`).
   - `onnxManifestPath` → ONNX manifest path inside the service (defaults to
     `/data/identity_bundle/artifact_manifest.json`).
   - `baseUrl` → URL of the running service.
3. Execute the numbered requests in order: `01 - Health Check` through
   `07 - Admin Unload`.

The collection walks through loading the rule-based sample, exercising
inference/reward, validating feature aliases, unloading, then repeating the flow
with the generated identity ONNX bundle.

### Docker Compose

Run the stack with the provided compose file:

```bash
docker compose up --build
```

Environment overrides (set via `export VAR=value` or a `.env` file):

- `BUNDLE_PATH` → host directory containing the manifest bundles (defaults to
  `./examples`).
- `MODEL_MANIFEST_PATH` → manifest path as seen inside the container (defaults to
  `/data/rule_based/artifact_manifest.json`).
- `MODEL_AGENT_INDEX` → which agent from the bundle to serve (defaults to `0`).
- `ONNX_EXECUTION_PROVIDERS` → e.g. `CUDAExecutionProvider,CPUExecutionProvider`.

Quick example before calling `docker compose up`:

```bash
export BUNDLE_PATH="$PWD/examples"
export MODEL_MANIFEST_PATH="/data/rule_based/artifact_manifest.json"
export MODEL_AGENT_INDEX=0
export ONNX_EXECUTION_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"
docker compose --compatibility up --build
```

To boot straight into the ONNX identity sample instead, set
`MODEL_MANIFEST_PATH="/data/identity_bundle/artifact_manifest.json"` before
launching or perform the swap at runtime via the Postman collection.

On GPU hosts install the NVIDIA Container Toolkit and run with compatibility
mode so Compose honours the device reservation:

```bash
docker compose --compatibility up --build
```

If no GPU is present the container continues on CPU without changes.

> **Note:** The Dockerfile uses `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, so
> the first build downloads the CUDA + cuDNN runtime (large image) and requires
> the NVIDIA Container Toolkit on the host for GPU access.

### GPU Execution

To execute ONNX models on GPU:

1. Uninstall the CPU-only runtime and install the CUDA build:
   ```bash
   pip uninstall onnxruntime
   pip install onnxruntime-gpu
   ```
2. Ensure NVIDIA drivers / CUDA libraries are present on the host or Docker base
   image.
3. Set `ONNX_EXECUTION_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"`
   (order defines the fallback sequence). The service reads this variable and
   passes it directly to ONNX Runtime when instantiating the session.

If GPU is not available, the provider list automatically falls back to
`CPUExecutionProvider`.

## Roadmap

- Support additional reward functions beyond the baseline set.
- Add caching/pooling for ONNXRuntime sessions (GPU execution providers).
- Provide CLI tooling to fetch bundles from MLflow automatically.
- Implement hot-reload/admin endpoints for OTA model swaps.
