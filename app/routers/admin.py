from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body, HTTPException
from app.state import store
from app.logging import get_logger
from app.settings import settings

router = APIRouter(prefix="/admin", tags=["admin"], responses={401: {"description": "Unauthorized"}})


def _validate_path_within_allowed_root(path: Path, *, field_name: str) -> None:
    root = settings.allowed_bundle_root
    if root is None:
        return

    resolved_root = root.expanduser().resolve()
    resolved_path = path.expanduser().resolve()

    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{field_name} path '{resolved_path}' is outside ALLOWED_BUNDLE_ROOT "
                f"'{resolved_root}'"
            ),
        ) from exc


@router.post("/load")
async def admin_load(
    manifest_path: Path = Body(..., embed=True),
    agent_index: int | None = Body(default=None, embed=True, ge=0),
    artifacts_dir: Path | None = Body(default=None, embed=True),
    alias_mapping_path: Path | None = Body(default=None, embed=True),
):
    """Load a model pipeline from disk, replacing any currently active pipeline."""
    _validate_path_within_allowed_root(manifest_path, field_name="manifest_path")
    if artifacts_dir is not None:
        _validate_path_within_allowed_root(artifacts_dir, field_name="artifacts_dir")

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
        "default_agent_index": record.default_agent_index,
        "loaded_agent_indices": record.loaded_agent_indices,
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
