"""Legacy iCharging diagnostics; never part of generic HTTP handling."""

from typing import Any, Dict

from app.logging import get_logger


def log_single_agent_actions(pipeline, flattened: Dict[str, Any], actions: Dict[str, Dict[str, float]]) -> None:
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


