from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app.models.requests import RewardRequest
from app.models.responses import RewardResponse
from app.state import store
from app.services.reward_service import RewardCalculator

router = APIRouter(prefix="/reward", tags=["reward"])


def get_reward_calculator():
    if not store.is_configured():
        logger.warning("Reward requested before model configuration")
        raise HTTPException(status_code=503, detail="Model not configured")
    pipeline = store.get_pipeline()
    return RewardCalculator(manifest=pipeline.manifest)


@router.post("", response_model=RewardResponse)
async def compute_reward(payload: RewardRequest, calculator = Depends(get_reward_calculator)):
    try:
        rewards = calculator.calculate(payload.observations)
    except NotImplementedError as exc:
        logger.warning("Requested unsupported reward function: %s", exc)
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Reward calculation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Reward calculation failed") from exc
    return RewardResponse(rewards=rewards)
