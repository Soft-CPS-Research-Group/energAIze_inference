from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from app.models.requests import InferenceRequest
from app.models.responses import InferenceResponse
from app.logging import get_logger
from app.state import store
from app.utils.flatten import flatten_payload

router = APIRouter(prefix="/inference", tags=["inference"])


def get_runtime_pipeline():
    try:
        return store.get_pipeline()
    except RuntimeError:
        get_logger().warning("Inference requested before model configuration")
        raise HTTPException(status_code=503, detail="Model not configured") from None


@router.post("", response_model=InferenceResponse)
async def run_inference(payload: InferenceRequest, request: Request, pipeline = Depends(get_runtime_pipeline)):
    """Run inference for the configured agent using the supplied feature dict."""
    log = get_logger()
    request_id = None
    if request is not None:
        request_id = getattr(request.state, "request_id", None) or request.headers.get("x-request-id")
    try:
        flattened = flatten_payload(payload.features)
        actions = pipeline.inference(flattened)
    except KeyError as exc:
        log.warning("Inference payload missing data", missing=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference execution failed") from exc

    # Emit per-request action summary with phase/board totals for debugging.
    try:
        manifest = pipeline.manifest
        agent_cfg = manifest.agent.artifacts[pipeline.agent_index].config or {}
        chargers_cfg = agent_cfg.get("chargers", {})
        actions_for_agent = actions.get(str(pipeline.agent_index), actions.get(pipeline.agent_index, {}))
        connected = {
            cid: bool(str(flattened.get(f"charging_sessions.{cid}.electric_vehicle", "")).strip())
            for cid in actions_for_agent
            if not str(cid).startswith("b_")
        }
        line_totals: dict[str, float] = {}
        for cid, value in actions_for_agent.items():
            if str(cid).startswith("b_"):
                continue
            if not connected.get(cid):
                continue
            meta = chargers_cfg.get(cid, {})
            phases = meta.get("phases") or ([meta.get("line")] if meta.get("line") else [])
            phases = [p for p in phases if p]
            n_phases = max(len(phases), 1)
            per_phase = value / n_phases
            for phase in phases or ["unknown"]:
                line_totals[phase] = line_totals.get(phase, 0.0) + per_phase
        board_total = sum(v for k, v in actions_for_agent.items() if connected.get(k))
        line_totals = dict(sorted(line_totals.items()))
        log.info(
            "inference.actions",
            actions=actions_for_agent,
            connected=connected,
            phase_totals=line_totals,
            board_total=board_total,
        )
    except Exception:
        log.exception("inference.action_logging_failed")

    return InferenceResponse(actions=actions, request_id=request_id)
