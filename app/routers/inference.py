from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from app.models.requests import InferenceRequest
from app.models.responses import InferenceResponse
from app.logging import get_logger
from app.application.service import InferenceService
from app.dependencies import get_service

router = APIRouter(prefix="/inference", tags=["inference"])


def _ensure_configured(service: InferenceService = Depends(get_service)):
    if not service.store.is_configured():
        get_logger().warning("Inference requested before model configuration")
        raise HTTPException(status_code=503, detail="Model not configured")


@router.post("", response_model=InferenceResponse)
async def run_inference(
    payload: InferenceRequest,
    request: Request,
    _ = Depends(_ensure_configured),
    service: InferenceService = Depends(get_service),
):
    """Run inference for the configured agent using the supplied feature dict."""
    log = get_logger()
    request_id = None
    if request is not None:
        request_id = getattr(request.state, "request_id", None) or request.headers.get("x-request-id")
    try:
        actions = service.infer(payload.features, payload.agent_index)
    except KeyError as exc:
        log.warning("Inference payload missing data", missing=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc)
        status_code = 503 if "not configured" in detail else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference execution failed") from exc

    return InferenceResponse(actions=actions, request_id=request_id)
