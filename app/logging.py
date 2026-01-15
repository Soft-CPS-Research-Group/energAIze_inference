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
_ACTION_MESSAGES = {"rbc.actions", "inference.actions"}


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value) -> str:
    numeric = _as_float(value)
    if numeric is None:
        if value is None:
            return "-"
        return str(value)
    return f"{numeric:.1f}"


def _connected_ids(connected) -> set[str]:
    if not isinstance(connected, dict):
        return set()
    ids = set()
    for cid, value in connected.items():
        if isinstance(value, bool):
            if value:
                ids.add(str(cid))
            continue
        if value is None:
            continue
        if str(value).strip():
            ids.add(str(cid))
    return ids


def _summarize_actions(actions, connected: set[str], max_items: int = 32) -> str:
    if not isinstance(actions, dict) or not actions:
        return "-"
    items = (
        {cid: actions.get(cid) for cid in connected if cid in actions}
        if connected
        else dict(actions)
    )
    parts = [f"{cid}={_format_value(items[cid])}" for cid in sorted(items)]
    if len(parts) > max_items:
        parts = parts[:max_items] + [f"...+{len(parts) - max_items}"]
    return ", ".join(parts) if parts else "-"


def _summarize_flex(flex: dict, max_items: int = 6) -> str:
    if not isinstance(flex, dict) or not flex:
        return "-"
    parts: list[str] = []
    for cid, details in sorted(flex.items()):
        if not isinstance(details, dict):
            parts.append(f"{cid}")
            continue
        ev = details.get("ev")
        req = details.get("required_kw")
        prio = details.get("priority")
        parts.append(
            f"{cid}(ev={ev},req={_format_value(req)},prio={_format_value(prio)})"
        )
    if len(parts) > max_items:
        parts = parts[:max_items] + [f"...+{len(parts) - max_items}"]
    return "; ".join(parts) if parts else "-"


def _summarize_phase_totals(phase_totals: dict) -> str:
    if not isinstance(phase_totals, dict) or not phase_totals:
        return "-"
    parts = [f"{phase}={_format_value(value)}" for phase, value in sorted(phase_totals.items())]
    return " ".join(parts) if parts else "-"


def _patch_record(record: dict) -> None:
    extra = record["extra"]
    actions = extra.get("actions")
    connected_ids = _connected_ids(extra.get("connected"))
    extra["connected_count"] = len(connected_ids)
    if isinstance(actions, dict):
        extra["actions_summary"] = _summarize_actions(actions, connected_ids)
    phase_totals = extra.get("phase_totals")
    if isinstance(phase_totals, dict):
        extra["phase_summary"] = _summarize_phase_totals(phase_totals)
    flex = extra.get("flex")
    if isinstance(flex, dict):
        extra["flex_summary"] = _summarize_flex(flex)
    board_total = extra.get("board_total")
    if isinstance(board_total, (int, float)):
        extra["board_total_kw"] = _format_value(board_total)


def _is_action_record(record: dict) -> bool:
    return record.get("message") in _ACTION_MESSAGES


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
            "strategy": "-",
            "actions_summary": "-",
            "phase_summary": "-",
            "flex_summary": "-",
            "board_total_kw": "-",
            "connected_count": "-",
        },
        patcher=_patch_record,
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
        default_fmt = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
            "request_id={extra[request_id]} | status={extra[status_code]} | "
            "{extra[method]} {extra[path]} | {message}"
        )
        action_fmt = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message} | "
            "request_id={extra[request_id]} | {extra[method]} {extra[path]} | "
            "strategy={extra[strategy]} | board={extra[board_total_kw]} | "
            "phases={extra[phase_summary]} | connected={extra[connected_count]} | "
            "flex={extra[flex_summary]} | actions={extra[actions_summary]}"
        )
        logger.add(
            sys.stdout,
            level=settings.log_level.upper(),
            format=default_fmt,
            filter=lambda record: not _is_action_record(record),
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )
        logger.add(
            sys.stdout,
            level=settings.log_level.upper(),
            format=action_fmt,
            filter=_is_action_record,
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
        # propagate request_id on the request for downstream handlers
        request.state.request_id = request_id
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
