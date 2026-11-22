from __future__ import annotations

import hmac
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from app.models.requests import InferenceRequest
from app.models.responses import InferenceResponse
from app.logging import get_logger
from app.application.service import InferenceService
from app.dependencies import get_service
from app.core.errors import ExecutionError, InputError
from app.contracts.v2.lifecycle import CycleRequest, Decision
from app.routers.preparation import bearer, manager, translate
from app.core.state import PipelineNotConfigured

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("", response_model=InferenceResponse)
async def run_inference(
    payload: InferenceRequest,
    request: Request,
    service: InferenceService = Depends(get_service),
):
    """Run inference for the configured agent using the supplied feature dict."""
    if getattr(request.app.state, "preparation", None) is not None:
        raise HTTPException(409, detail="Managed lifecycle requires /inference/v2 with explicit activation and context fences")
    log = get_logger()
    request_id = None
    if request is not None:
        request_id = getattr(request.state, "request_id", None) or request.headers.get("x-request-id")
    try:
        actions = service.infer(payload.features, payload.agent_index)
    except PipelineNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InputError as exc:
        raise HTTPException(status_code=400, detail=exc.document()) from exc
    except ExecutionError as exc:
        log.exception("Inference runtime failed", code=exc.code)
        raise HTTPException(status_code=500, detail=exc.document()) from exc
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


@router.post("/v2", response_model=Decision)
def run_managed_inference(body: CycleRequest, request: Request, service=Depends(manager)):
    """Capture one generation for the whole cycle; this route runs off the event loop."""
    return managed_decision(body, request, service)


def managed_decision(body, request, service):
    try:
        result = translate(lambda: service.lifecycle.infer(body))
    except InputError as exc:
        raise HTTPException(400, detail=exc.document()) from exc
    except ExecutionError as exc:
        raise HTTPException(500, detail=exc.document()) from exc
    result["request_id"] = getattr(request.state, "request_id", None)
    return result


class PerceptaObservation(BaseModel):
    # Percepta's native dispatcher expects scalar setpoints keyed by entity.
    # A structured observation is archived by Percepta and carries no setpoints to dispatch.
    decision: Decision


def period_timestamp(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))


def percepta_target(site_id: str, model_id: str, request: Request,
                    context_id: str = Header(..., alias="X-Installation-Context", min_length=1),
                    credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    settings = request.app.state.settings
    secret = settings.percepta_cycle_token
    if not secret or not credentials or not hmac.compare_digest(
        credentials.credentials.encode(), secret.get_secret_value().encode()
    ):
        raise HTTPException(401, "Percepta cycle credential required")
    if (site_id, model_id) != (settings.installation_site_id, settings.deployment_model_id):
        raise HTTPException(404, "Unknown installation or model")
    service = request.app.state.preparation
    state = service.lifecycle.status()
    if not state["ready"]:
        raise HTTPException(503, "No ready activation for this installation")
    if context_id != state["context_id"]:
        raise HTTPException(409, "Installation context changed")
    return service, state


@router.get("/sites/{site_id}/models/{model_id}/observe")
def percepta_readiness(site_id: str, model_id: str, target=Depends(percepta_target)):
    _, state = target
    return {"site_id": site_id, "model_id": model_id, "context_id": state["context_id"], "ready": True}


@router.post("/sites/{site_id}/models/{model_id}/observe", response_model=PerceptaObservation)
def observe_percepta_cycle(site_id: str, model_id: str, body: dict, request: Request,
                           target=Depends(percepta_target)):
    service, state = target
    context_id = state["context_id"]
    try:
        start, end = (period_timestamp(body[key]) for key in ("period_start", "period_end"))
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError):
        raise HTTPException(422, "A cycle requires an ordered period with timezone offsets") from None
    cycle = CycleRequest(features=body, expected_context_id=context_id,
                         expected_activation_id=state["activation_id"],
                         cycle_id=f"{site_id}/{start.astimezone(timezone.utc).isoformat()}/{end.astimezone(timezone.utc).isoformat()}")
    decision = managed_decision(cycle, request, service)
    current = service.lifecycle.status()
    if not current["ready"] or (current["context_id"], current["activation_id"]) != (context_id, cycle.expected_activation_id):
        raise HTTPException(409, "Installation or activation changed during the cycle")
    return {"decision": decision}
