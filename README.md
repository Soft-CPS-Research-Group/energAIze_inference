# Energy Flexibility Inference API

FastAPI service to serve charging control policies. Supports:
- Rule-based controllers (icharging breaker with flexibility, breaker-only limiter).
- ONNX models (sample bundle included; plug in real RL exports).

## What the service does
- Enforces electrical limits: 11 kW per phase, 33 kW board, per-charger max (4.6 kW single-phase, 9 kW tri-phase `BB000018_1`).
- Minimum power: any connected charger (EV id present) emits at least 1.6 kW (per phase for tri-phase) even if observed power is lower.
- Empty chargers emit 1.6 kW but are excluded from board/phase accounting.
- Outputs are floored to 1 decimal place and clamped to limits.
- Flexibility (icharging_breaker only): prioritises flexible EVs by SoC/target/departure; solar adds bonus headroom but does not lift board cap.
- Request IDs: optional `x-request-id` is propagated to logs and responses.

## API
- `POST /inference` – payload: `{"features": {...}}`; responds with `{"actions": {"0": {...}}, "request_id": "..."}`.
- `GET /info` – manifest metadata.
- `GET /health` – readiness (includes whether a model is loaded).
- `POST /admin/load` – load a bundle: `{"manifest_path": "...", "agent_index": 0, "artifacts_dir": null, "alias_mapping_path": "..."}`.
- `POST /admin/unload` – unload current model.

### Payload contract (RBC and ONNX)
- Wrap in `{"features": {...}}`.
- `charging_sessions.<charger_id>.power` (float) and `.electric_vehicle` (string/int or empty). Chargers listed in the manifest should appear; extras are tolerated.
- `electric_vehicles.<ev_id>.SoC` (0–1) and optional flexibility fields; EV ids are dynamic.
- Optional globals: `timestamp`, `solar_generation`, `non_shiftable_load`, `energy_price`, `pv_panels`, `grid_meters`, `batteries`.

## Bundles included
- `examples/ichargingusecase_rule_based/` – icharging_breaker (full flexibility + limits, control interval 1 min).
- `examples/ichargingusecase_v0/` – breaker_only (limits only, 1 min).
- `examples/ichargingusecase_onnx/` – ONNX sample (for testing, not production).

## Running locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Or via Docker Compose:
```bash
export BUNDLE_PATH="$PWD/examples"
export MODEL_MANIFEST_PATH="/data/ichargingusecase_rule_based/artifact_manifest.json"
export FEATURE_ALIAS_PATH="/data/ichargingusecase_rule_based/aliases.json"
export MODEL_AGENT_INDEX=0
docker compose up --build
```
Environment vars (see `app/settings.py`):
- `MODEL_MANIFEST_PATH` (required to auto-load)  
- `MODEL_ARTIFACTS_DIR` (optional base path)  
- `MODEL_AGENT_INDEX` (agent to serve)  
- `LOG_LEVEL` (INFO/DEBUG)  
- `LOG_JSON` (true/false)  
- `FEATURE_ALIAS_PATH` (alias mapping)  
- `ONNX_EXECUTION_PROVIDERS` (e.g., `CUDAExecutionProvider,CPUExecutionProvider`)

## Tests
- Full suite: `pytest tests/test_api_inference.py`.
- Includes replay of real logs: `dados_de_inferência_IC_11.11.2025_a_14.11.2025.json` and `dados_de_inferência_IC_14.11.2025_a_18.11.2025.json`.
- ONNX sample bundle is covered by `test_onnx_icharging_sample_bundle`.

## Postman
Collection: `postman/EnergyFlexibilityInference.postman_collection.json`.
- Steps for rule-based, breaker-only, and ONNX sample.
- `requestIdExample` variable sets `x-request-id` in sample requests.
Update paths to match your deployment (e.g., `/data/ichargingusecase_rule_based/artifact_manifest.json`).

## Observability
- Structured logs (JSON if `LOG_JSON=true`, otherwise console).  
- Per-request log: actions, connected chargers, phase totals, board total, request_id.
- To see request IDs in responses, send `x-request-id`; otherwise one is generated.

## Notes / Next improvements
- Input validation against the manifest (charger set/types) is not yet enforced.
- No metrics/alerts; add Prometheus/StatsD for latency and clamp/failure counts.
- No ONNX → RBC fallback; currently ONNX errors return zeros.
- Provide real ONNX exports for production; the included one is a stub.
