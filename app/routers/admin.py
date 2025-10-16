from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body, HTTPException
from app.state import store
from app.logging import get_logger

router = APIRouter(prefix="/admin", tags=["admin"], responses={401: {"description": "Unauthorized"}})


@router.post("/load")
async def admin_load(
    manifest_path: Path = Body(..., embed=True),
    agent_index: int = Body(..., embed=True, ge=0),
    artifacts_dir: Path | None = Body(default=None, embed=True),
    alias_mapping_path: Path | None = Body(default=None, embed=True),
):
    """Load a model pipeline from disk, replacing any currently active pipeline."""
    try:
        record = store.load(manifest_path, artifacts_dir, agent_index, alias_mapping_path)
    except FileNotFoundError as exc:
        get_logger().warning("Failed to load pipeline", error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        get_logger().exception("Failed to load pipeline")
        raise HTTPException(status_code=500, detail="Failed to load model") from exc

    return {
        "status": "loaded",
        "manifest_path": str(record.manifest_path),
        "agent_index": record.agent_index,
        "loaded_at": record.loaded_at.isoformat() + "Z",
    }


@router.post("/unload")
async def admin_unload():
    """Unload the active pipeline, returning the service to an unconfigured state."""
    if not store.is_configured():
        raise HTTPException(status_code=409, detail="No model configured")
    store.unload()
    get_logger().info("Pipeline successfully unloaded")
    return {"status": "unloaded"}
