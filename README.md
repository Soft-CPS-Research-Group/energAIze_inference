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

## Configuration

See `app/settings.py` for configurable environment variables.

- `MODEL_MANIFEST_PATH` (required): location of manifest.
- `MODEL_ARTIFACTS_DIR` (optional): base directory for relative paths.

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

## Roadmap

- Support additional reward functions beyond the baseline set.
- Add caching/pooling for ONNXRuntime sessions (GPU execution providers).
- Provide CLI tooling to fetch bundles from MLflow automatically.
- Implement hot-reload/admin endpoints for OTA model swaps.
