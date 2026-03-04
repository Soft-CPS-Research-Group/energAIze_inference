from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from app.models.requests import InferenceRequest
from app.models.responses import InferenceResponse
from app.logging import get_logger
from app.services.rbc.community_optimizer import CommunityOptimizerRuntime
from app.state import store
from app.utils.flatten import flatten_payload

router = APIRouter(prefix="/inference", tags=["inference"])
community_optimizer = CommunityOptimizerRuntime()


def _ensure_configured():
    if not store.is_configured():
        get_logger().warning("Inference requested before model configuration")
        raise HTTPException(status_code=503, detail="Model not configured")


def _select_agent_features(features: Dict[str, Any], pipeline) -> Dict[str, Any]:
    artifact_cfg = pipeline.manifest.get_artifact(pipeline.agent_index).config or {}
    site_key = str(artifact_cfg.get("input_site_key") or "").strip()
    if not site_key:
        return dict(features)

    sites = features.get("sites")
    if not isinstance(sites, dict):
        raise KeyError(f"Missing required object 'features.sites' for agent site '{site_key}'")

    site_payload = sites.get(site_key)
    if not isinstance(site_payload, dict):
        raise KeyError(f"Missing required object 'features.sites.{site_key}' for agent_index {pipeline.agent_index}")

    selected = dict(site_payload)
    top_community = features.get("community")
    if isinstance(top_community, dict) and "community" not in selected:
        selected["community"] = dict(top_community)
    top_timestamp = features.get("timestamp")
    if "timestamp" not in selected and top_timestamp is not None:
        selected["timestamp"] = top_timestamp
    top_timestamp_date = features.get("timestamp.$date")
    if (
        "timestamp" not in selected
        and "timestamp.$date" not in selected
        and top_timestamp_date is not None
    ):
        selected["timestamp.$date"] = top_timestamp_date
    return selected


def _normalize_agent_input(selected_features: Dict[str, Any], pipeline) -> Dict[str, Any]:
    artifact_cfg = pipeline.manifest.get_artifact(pipeline.agent_index).config or {}
    require_observations = bool(artifact_cfg.get("require_observations_envelope", False))
    if not require_observations:
        return dict(selected_features)

    observations = selected_features.get("observations")
    if not isinstance(observations, dict):
        raise KeyError("Missing required object 'observations' for this bundle")

    normalized = dict(observations)
    top_timestamp = selected_features.get("timestamp")
    if "timestamp" not in normalized and top_timestamp is not None:
        normalized["timestamp"] = top_timestamp
    top_timestamp_date = selected_features.get("timestamp.$date")
    if (
        "timestamp" not in normalized
        and "timestamp.$date" not in normalized
        and top_timestamp_date is not None
    ):
        normalized["timestamp.$date"] = top_timestamp_date
    return normalized


def _is_community_mode(features: Dict[str, Any], record) -> bool:
    if record is None or len(record.loaded_agent_indices) <= 1:
        return False
    sites = features.get("sites")
    if not isinstance(sites, dict):
        return False

    manifest = record.pipelines[record.loaded_agent_indices[0]].manifest
    has_site_mapping = True
    optimization_enabled = False
    for agent_index in record.loaded_agent_indices:
        cfg = manifest.get_artifact(agent_index).config or {}
        site_key = str(cfg.get("input_site_key") or "").strip()
        if not site_key:
            has_site_mapping = False
            break
        if bool(cfg.get("community_optimization_enabled", False)):
            optimization_enabled = True

    return has_site_mapping and optimization_enabled


def _log_single_agent_actions(pipeline, flattened: Dict[str, Any], actions: Dict[str, Dict[str, float]]) -> None:
    manifest = pipeline.manifest
    agent_cfg = manifest.get_artifact(pipeline.agent_index).config or {}
    chargers_cfg = agent_cfg.get("chargers", {})
    actions_for_agent = actions.get(str(pipeline.agent_index), actions.get(pipeline.agent_index, {}))
    connected = {
        cid: bool(str(flattened.get(f"charging_sessions.{cid}.electric_vehicle", "")).strip())
        for cid in chargers_cfg
        if not str(cid).startswith("b_")
    }
    line_totals: dict[str, float] = {}
    for cid, value in actions_for_agent.items():
        if str(cid).startswith("b_") or cid not in chargers_cfg:
            continue
        if not connected.get(cid):
            continue
        meta = chargers_cfg.get(cid, {})
        phases = meta.get("phases") or ([meta.get("line")] if meta.get("line") else [])
        phases = [p for p in phases if p]
        n_phases = max(len(phases), 1)
        per_phase = value / n_phases
        for phase in phases or ["unknown"]:
            line_totals[phase] = line_totals.get(phase, 0.0) + per_phase
    board_total = sum(v for k, v in actions_for_agent.items() if connected.get(k))
    line_totals = dict(sorted(line_totals.items()))
    get_logger().debug(
        "inference.actions",
        actions=actions_for_agent,
        connected=connected,
        phase_totals=line_totals,
        board_total=board_total,
    )


@router.post("", response_model=InferenceResponse)
async def run_inference(payload: InferenceRequest, request: Request, _ = Depends(_ensure_configured)):
    """Run inference for the configured agent using the supplied feature dict."""
    log = get_logger()
    request_id = None
    if request is not None:
        request_id = getattr(request.state, "request_id", None) or request.headers.get("x-request-id")
    try:
        record = store.get_record()
        if _is_community_mode(payload.features, record):
            actions = community_optimizer.allocate(payload.features, record)
        else:
            pipeline = store.get_pipeline(payload.agent_index)
            selected_features = _select_agent_features(payload.features, pipeline)
            runtime_features = _normalize_agent_input(selected_features, pipeline)
            flattened = flatten_payload(runtime_features)
            actions = pipeline.inference(flattened)
            try:
                _log_single_agent_actions(pipeline, flattened, actions)
            except Exception:
                log.exception("inference.action_logging_failed")
    except KeyError as exc:
        log.warning("Inference payload missing data", missing=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc)
        status_code = 503 if "not configured" in detail else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference execution failed") from exc

    return InferenceResponse(actions=actions, request_id=request_id)
