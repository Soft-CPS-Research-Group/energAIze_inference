from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.routers import admin, inference, info, reward
from app.logging import RequestContextMiddleware, get_logger, init_logging
from app.settings import settings
from app.state import store
from app.version import __version__

init_logging()

app = FastAPI(title="Energy Flexibility Inference API", version=__version__)
app.add_middleware(RequestContextMiddleware)

app.include_router(info.router)
app.include_router(inference.router)
app.include_router(reward.router)
app.include_router(admin.router)


@app.on_event("startup")
async def on_startup() -> None:
    if settings.manifest_path is not None:
        if settings.agent_index is None:
            raise RuntimeError("MODEL_AGENT_INDEX must be set when MODEL_MANIFEST_PATH is provided")
        store.load(
            settings.manifest_path,
            settings.artifacts_dir,
            settings.agent_index,
            settings.alias_mapping_path,
        )
    else:
        get_logger().info("Service started without configured model. Awaiting /admin/load to configure.")


@app.get("/health", tags=["info"])
async def health_check():
    record = store.get_record()
    return {
        "status": "ok",
        "configured": record is not None,
        "agent_index": record.agent_index if record else None,
    }
