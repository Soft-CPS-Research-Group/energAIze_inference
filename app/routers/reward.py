from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from app.models.requests import RewardRequest
from app.models.responses import RewardResponse
from app.state import store
from app.services.reward_service import RewardCalculator
from app.logging import get_logger

router = APIRouter(prefix="/reward", tags=["reward"])


def get_reward_calculator():
    if not store.is_configured():
        get_logger().warning("Reward requested before model configuration")
        raise HTTPException(status_code=503, detail="Model not configured")
    pipeline = store.get_pipeline()
    return RewardCalculator(manifest=pipeline.manifest)


@router.post("", response_model=RewardResponse)
async def compute_reward(payload: RewardRequest, calculator = Depends(get_reward_calculator)):
    """Compute reward values using the configured reward function."""
    try:
        rewards = calculator.calculate(payload.observations)
    except NotImplementedError as exc:
        get_logger().warning("Requested unsupported reward function", error=str(exc))
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        get_logger().exception("Reward calculation failed")
        raise HTTPException(status_code=500, detail="Reward calculation failed") from exc
    return RewardResponse(rewards=rewards)
