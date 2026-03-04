from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EPS = 1e-6
BASE_BOARD_LIMIT_KW = 55.0
TRI_PHASE = {"BB000018_1": 3}
LINE_GROUPS = {
    "L1": [
        "ACEXT003_1",
        "ACEXT002_1",
        "AC000014_1",
        "AC000011_1",
        "AC000008_1",
        "AC000005_1",
        "AC000002_1",
        "ACEXT001_1",
        "BB000018_1",
    ],
    "L2": ["ACEXT004_1", "AC000012_1", "AC000009_1", "AC000006_1", "AC000003_1", "BB000018_1"],
    "L3": ["AC000013_1", "AC000010_1", "AC000007_1", "AC000004_1", "AC000001_1", "BB000018_1"],
}


def post_inference(client: Any, payload: dict[str, Any]) -> dict[str, float]:
    response = client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    return response.json()["actions"]["0"]


def connected_chargers(payload: dict[str, Any]) -> set[str]:
    sessions = payload.get("observations", {}).get("charging_sessions", {})
    return {
        charger_id
        for charger_id, data in sessions.items()
        if str(data.get("electric_vehicle") or "").strip()
    }


def assert_board_and_phase_limits(
    actions: dict[str, float],
    payload: dict[str, Any],
    *,
    board_limit_kw: float,
) -> None:
    connected = connected_chargers(payload)
    board_total = sum(actions.get(charger_id, 0.0) for charger_id in connected)
    assert board_total <= board_limit_kw + EPS

    per_phase_limit = board_limit_kw / 3.0
    for chargers in LINE_GROUPS.values():
        phase_total = sum(
            actions.get(charger_id, 0.0) / TRI_PHASE.get(charger_id, 1)
            for charger_id in chargers
            if charger_id in connected
        )
        assert phase_total <= per_phase_limit + EPS


def assert_connected_charger_action_bounds(actions: dict[str, float], payload: dict[str, Any]) -> None:
    connected = connected_chargers(payload)
    for charger_id in connected:
        action = actions.get(charger_id, 0.0)
        assert action >= 0.0
        max_kw = 10.0 if charger_id == "BB000018_1" else 4.6
        assert action <= max_kw + EPS
        if action > EPS:
            assert action >= 1.6 - EPS


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": record.get("timestamp") or record.get("timestamp.$date"),
        "observations": dict(record.get("observations", {})),
        "forecasts": dict(record.get("forecasts", {})),
    }
