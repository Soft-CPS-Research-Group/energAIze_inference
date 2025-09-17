from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from app.models.responses import InfoResponse
from app.state import store
from app.version import __version__
from app.main import get_startup_time

router = APIRouter(prefix="/info", tags=["info"])


def get_record():
    record = store.get_record()
    if not record:
        raise HTTPException(status_code=503, detail="Model not configured")
    return record


@router.get("", response_model=InfoResponse)
async def get_info(record = Depends(get_record)):
    manifest = record.pipeline.manifest
    uptime_seconds = (datetime.utcnow() - get_startup_time()).total_seconds()
    return InfoResponse(
        algorithm=manifest.algorithm.get("name", "unknown"),
        manifest_version=manifest.manifest_version,
        metadata=manifest.metadata,
        topology=manifest.topology,
        service_version=__version__,
        agent_index=record.agent_index,
        action_names=manifest.environment.action_names,
        uptime_seconds=uptime_seconds,
        loaded_at=record.loaded_at.isoformat() + "Z",
    )
