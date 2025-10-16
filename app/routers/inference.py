from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from app.models.requests import InferenceRequest
from app.models.responses import InferenceResponse
from app.logging import get_logger
from app.state import store

router = APIRouter(prefix="/inference", tags=["inference"])


def get_runtime_pipeline():
    try:
        return store.get_pipeline()
    except RuntimeError:
        get_logger().warning("Inference requested before model configuration")
        raise HTTPException(status_code=503, detail="Model not configured") from None


@router.post("", response_model=InferenceResponse)
async def run_inference(payload: InferenceRequest, pipeline = Depends(get_runtime_pipeline)):
    """Run inference for the configured agent using the supplied feature dict."""
    try:
        actions = pipeline.inference(payload.features)
    except KeyError as exc:
        get_logger().warning("Inference payload missing data", missing=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        get_logger().exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference execution failed") from exc
    return InferenceResponse(actions=actions)
