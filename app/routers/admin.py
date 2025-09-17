from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body, HTTPException
from loguru import logger

from app.state import store

router = APIRouter(prefix="/admin", tags=["admin"], responses={401: {"description": "Unauthorized"}})


@router.post("/load")
async def admin_load(
    manifest_path: Path = Body(..., embed=True),
    agent_index: int = Body(..., embed=True, ge=0),
    artifacts_dir: Path | None = Body(default=None, embed=True),
):
    try:
        record = store.load(manifest_path, artifacts_dir, agent_index)
    except FileNotFoundError as exc:
        logger.warning("Failed to load pipeline: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load pipeline: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to load model") from exc

    return {
        "status": "loaded",
        "manifest_path": str(record.manifest_path),
        "agent_index": record.agent_index,
        "loaded_at": record.loaded_at.isoformat() + "Z",
    }


@router.post("/unload")
async def admin_unload():
    if not store.is_configured():
        raise HTTPException(status_code=409, detail="No model configured")
    store.unload()
    return {"status": "unloaded"}
