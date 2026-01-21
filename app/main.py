from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache

import onnxruntime as ort
from fastapi import FastAPI, HTTPException

from app.routers import admin, inference, info
from app.logging import RequestContextMiddleware, get_logger, init_logging
from app.settings import settings
from app.state import store
from app.version import __version__

init_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown behaviour for the inference service."""

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

    try:
        yield
    finally:
        if store.is_configured():
            get_logger().info("Lifespan shutdown unloading pipeline")
            store.unload()


app = FastAPI(title="Energy Flexibility Inference API", version=__version__, lifespan=lifespan)
app.add_middleware(RequestContextMiddleware)

app.include_router(info.router)
app.include_router(inference.router)
app.include_router(admin.router)


@lru_cache(maxsize=1)
def _cuda_available() -> bool:
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False
    try:
        import onnx
        from onnx import TensorProto, helper
    except Exception:
        get_logger().warning("onnx.unavailable_for_gpu_check")
        return False

    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "gpu_check",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, producer_name="gpu_check")
    try:
        session = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
        )
        return "CUDAExecutionProvider" in session.get_providers()
    except Exception:
        return False


@app.get("/health", tags=["info"])
async def health_check():
    record = store.get_record()
    gpu_available = _cuda_available()
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
        "providers": providers,
        "manifest_path": manifest_path,
        "alias_mapping_path": alias_path,
        "gpu_available": gpu_available,
    }
