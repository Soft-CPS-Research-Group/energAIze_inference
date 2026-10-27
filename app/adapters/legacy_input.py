"""Current Percepta envelope/flat payload behavior, preserved for compatibility.

This adapter is not the future versioned installation/message contract.
"""

from typing import Any, Dict

from app.core.contracts import Features, Pipeline
from app.utils.flatten import flatten_payload


def _select_agent_features(features: Dict[str, Any], pipeline) -> Dict[str, Any]:
    _ = pipeline
    return dict(features)


def _normalize_agent_input(selected_features: Dict[str, Any], pipeline) -> Dict[str, Any]:
    artifact_cfg = pipeline.manifest.get_artifact(pipeline.agent_index).config or {}
    require_observations = bool(artifact_cfg.get("require_observations_envelope", False))
    if not require_observations:
        return dict(selected_features)

    observations = selected_features.get("observations")
    if not isinstance(observations, dict):
        raise KeyError("Missing required object 'observations' for this bundle")

    normalized = dict(observations)
    top_forecasts = selected_features.get("forecasts")
    if isinstance(top_forecasts, dict) and "forecasts" not in normalized:
        normalized["forecasts"] = dict(top_forecasts)
    top_community = selected_features.get("community")
    if isinstance(top_community, dict) and "community" not in normalized:
        normalized["community"] = dict(top_community)
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


def _validate_required_community_fields(flattened: Dict[str, Any], pipeline) -> None:
    artifact_cfg = pipeline.manifest.get_artifact(pipeline.agent_index).config or {}
    if not bool(artifact_cfg.get("community_participation_enabled", False)):
        return

    in_key = str(artifact_cfg.get("community_energy_in_key", "community.energy_in_total")).strip()
    out_key = str(artifact_cfg.get("community_energy_out_key", "community.energy_out_total")).strip()
    if not in_key:
        in_key = "community.energy_in_total"
    if not out_key:
        out_key = "community.energy_out_total"

    missing: list[str] = []
    for key in (in_key, out_key):
        value = flattened.get(key)
        if value is None:
            missing.append(key)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(key)

    if missing:
        quoted = ", ".join(repr(key) for key in missing)
        raise KeyError(f"Missing required community field(s): {quoted}")


def prepare_legacy_input(features: Features, pipeline: Pipeline) -> Features:
    selected = _select_agent_features(features, pipeline)
    normalized = _normalize_agent_input(selected, pipeline)
    flattened = flatten_payload(normalized)
    _validate_required_community_fields(flattened, pipeline)
    return flattened
