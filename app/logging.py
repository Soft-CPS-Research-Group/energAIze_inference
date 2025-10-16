from __future__ import annotations

import contextvars
import sys
import time
import uuid
from typing import Callable

from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from app.settings import settings

_logger_ctx: contextvars.ContextVar = contextvars.ContextVar("_logger_ctx", default=logger)


def init_logging() -> None:
    """Configure Loguru with structured output and request-aware extras."""

    logger.remove()
    logger.configure(
        extra={
            "request_id": "-",
            "path": "-",
            "method": "-",
            "status_code": 0,
            "duration_ms": 0.0,
            "agent_index": None,
            "manifest_path": None,
        }
    )

    if settings.log_json:
        logger.add(
            sys.stdout,
            level=settings.log_level.upper(),
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )
    else:
        fmt = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
            "request_id={extra[request_id]} | status={extra[status_code]} | "
            "{extra[method]} {extra[path]} | {message}"
        )
        logger.add(
            sys.stdout,
            level=settings.log_level.upper(),
            format=fmt,
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )


def get_logger():  # noqa: ANN001
    """Return the request-scoped logger if available, otherwise the base logger."""

    return _logger_ctx.get()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach request-scoped metadata and structured logging to each request."""

    async def dispatch(self, request: Request, call_next: Callable):  # type: ignore[override]
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        start = time.perf_counter()

        request_logger = logger.bind(
            request_id=request_id,
            path=str(request.url.path),
            method=request.method,
        )
        token = _logger_ctx.set(request_logger)

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000
            request_logger.bind(status_code=response.status_code, duration_ms=duration_ms).info(
                "request.completed"
            )
            response.headers["x-request-id"] = request_id
            return response
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            request_logger.bind(status_code=500, duration_ms=duration_ms).exception("request.failed")
            raise
        finally:
            _logger_ctx.reset(token)
