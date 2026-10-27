from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable

from fastapi import Depends, FastAPI, Request

from app.adapters.hardware import _cuda_available
from app.application.service import InferenceService
from app.bootstrap import build_service
from app.dependencies import get_service
from app.routers import admin, inference, info
from app.logging import RequestContextMiddleware, get_logger, init_logging
from app.settings import Settings, settings
from app.state import store
from app.version import __version__

init_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown behaviour for the inference service."""

    settings = app.state.settings
    service = app.state.inference_service
    if settings.manifest_path is not None:
        service.load(
            settings.manifest_path,
            settings.artifacts_dir,
            settings.agent_index,
            settings.alias_mapping_path,
        )
    else:
        get_logger().info("Service started without configured model. Awaiting /admin/load to configure.")

    try:
        yield
    finally:
        if service.store.is_configured():
            get_logger().info("Lifespan shutdown unloading pipeline")
            service.unload()


async def health_check(request: Request, service: InferenceService = Depends(get_service)):
    record = service.store.get_record()
    gpu_available = request.app.state.gpu_probe()
    providers = []
    manifest_path = None
    alias_path = None
    if record:
        providers = getattr(record.pipeline.agent, "providers", [])
        manifest_path = str(record.manifest_path)
        alias_path = str(record.alias_mapping_path) if record.alias_mapping_path else None
    return {
        "status": "ok",
        "configured": record is not None,
        "agent_index": record.agent_index if record else None,
        "default_agent_index": record.default_agent_index if record else None,
        "loaded_agent_indices": record.loaded_agent_indices if record else [],
        "providers": providers,
        "manifest_path": manifest_path,
        "alias_mapping_path": alias_path,
        "gpu_available": gpu_available,
    }


def create_app(
    *,
    config: Settings | None = None,
    service: InferenceService | None = None,
    gpu_probe: Callable[[], bool] | None = None,
) -> FastAPI:
    """Create an isolated API instance with explicitly replaceable dependencies."""
    config = config if config is not None else Settings()
    application = FastAPI(
        title="Energy Flexibility Inference API", version=__version__, lifespan=lifespan
    )
    application.state.settings = config
    application.state.inference_service = service if service is not None else build_service(config)
    application.state.gpu_probe = gpu_probe if gpu_probe is not None else _cuda_available
    application.add_middleware(RequestContextMiddleware)
    application.include_router(info.router)
    application.include_router(inference.router)
    application.include_router(admin.router)
    application.get("/health", tags=["info"])(health_check)
    return application


# Preserve the ASGI entrypoint and legacy scripts that use app.state.store.
app = create_app(config=settings, service=build_service(settings, store))
