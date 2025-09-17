from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app.models.requests import InferenceRequest
from app.models.responses import InferenceResponse
from app.state import store

router = APIRouter(prefix="/inference", tags=["inference"])


def get_runtime_pipeline():
    try:
        return store.get_pipeline()
    except RuntimeError:
        logger.warning("Inference requested before model configuration")
        raise HTTPException(status_code=503, detail="Model not configured") from None


@router.post("", response_model=InferenceResponse)
async def run_inference(payload: InferenceRequest, pipeline = Depends(get_runtime_pipeline)):
    try:
        actions = pipeline.inference(payload.features)
    except KeyError as exc:
        logger.warning("Inference payload missing data: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Inference execution failed") from exc
    return InferenceResponse(actions=actions)
