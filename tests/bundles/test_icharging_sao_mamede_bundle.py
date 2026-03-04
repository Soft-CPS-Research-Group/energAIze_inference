from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store


BUNDLE_DIR = Path("examples/icharging_sao_mamede_without_virtual_battery")
MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
ALIAS_PATH = BUNDLE_DIR / "aliases.json"
MESSAGE_PATH = BUNDLE_DIR / "exemplos_mensagem_SaoMamede.json"


MIN_TECHNICAL_KW = 8.0
MAX_BB_KW = 50.0


def _run(client: TestClient, payload: dict) -> dict[str, float]:
    response = client.post("/inference", json={"features": payload})
    assert response.status_code == 200
    return response.json()["actions"]["0"]


@pytest.fixture
def sao_mamede_client():
    if store.is_configured():
        store.unload()
    store.load(MANIFEST_PATH, BUNDLE_DIR, 0, ALIAS_PATH)
    client = TestClient(app)
    try:
        yield client
    finally:
        if store.is_configured():
            store.unload()


def _base_payload() -> dict:
    payload = json.loads(MESSAGE_PATH.read_text(encoding="utf-8"))[0]
    payload = copy.deepcopy(payload)
    payload["timestamp"] = "2026-03-01T10:00:00Z"
    payload["observations"]["charging_sessions"] = {
        "BB000SMI_1": {"power": 0.0, "electric_vehicle": ""},
        "BB000SMI_2": {"power": 0.0, "electric_vehicle": ""},
        "M1123089-5_1": {"power": 0.0, "electric_vehicle": ""},
        "M1123089-5_2": {"power": 0.0, "electric_vehicle": ""},
        "M1123089-6_1": {"power": 0.0, "electric_vehicle": ""},
        "M1123089-6_2": {"power": 0.0, "electric_vehicle": ""},
    }
    payload["observations"]["electric_vehicles"] = {}
    payload["forecasts"] = {}
    return payload


def test_bundle_loads_sao_mamede(sao_mamede_client):
    pipeline = store.get_pipeline()
    assert pipeline.agent.strategy == "icharging_breaker"
    assert pipeline.agent._icharging_runtime is not None  # noqa: SLF001


def test_actions_contract_per_real_plug(sao_mamede_client):
    payload = _base_payload()
    actions = _run(sao_mamede_client, payload)
    assert set(actions.keys()) == {"BB000SMI_1", "BB000SMI_2"}


def test_idle_plug_outputs_min_technical_kw(sao_mamede_client):
    payload = _base_payload()
    payload["observations"]["charging_sessions"]["BB000SMI_1"] = {
        "power": 20.0,
        "electric_vehicle": "SM_EV_01",
    }
    payload["observations"]["electric_vehicles"] = {
        "SM_EV_01": {
            "SoC": 0.40,
            "flexibility": {
                "estimated_soc_at_departure": 0.80,
                "estimated_time_at_departure": "2026-03-01T14:00:00Z",
            },
        }
    }

    actions = _run(sao_mamede_client, payload)
    assert actions["BB000SMI_2"] == pytest.approx(MIN_TECHNICAL_KW, rel=1e-6)


def test_both_idle_output_minimum(sao_mamede_client):
    payload = _base_payload()
    actions = _run(sao_mamede_client, payload)
    assert actions["BB000SMI_1"] == pytest.approx(MIN_TECHNICAL_KW, rel=1e-6)
    assert actions["BB000SMI_2"] == pytest.approx(MIN_TECHNICAL_KW, rel=1e-6)


def test_active_plug_is_optimized(sao_mamede_client):
    payload = _base_payload()
    payload["observations"]["charging_sessions"]["BB000SMI_2"] = {
        "power": 18.0,
        "electric_vehicle": "SM_EV_02",
    }
    payload["observations"]["electric_vehicles"] = {
        "SM_EV_02": {
            "SoC": 0.30,
            "flexibility": {
                "estimated_soc_at_departure": 0.90,
                "estimated_time_at_departure": "2026-03-01T13:00:00Z",
            },
        }
    }

    actions = _run(sao_mamede_client, payload)
    assert MIN_TECHNICAL_KW <= actions["BB000SMI_2"] <= MAX_BB_KW
    assert actions["BB000SMI_1"] == pytest.approx(MIN_TECHNICAL_KW, rel=1e-6)


def test_ambiguous_dual_ev_prioritizes_highest_power_with_warning(
    sao_mamede_client,
    monkeypatch,
):
    warnings: list[tuple[str, dict]] = []

    class _DummyLogger:
        def warning(self, message, *args, **kwargs):  # noqa: ANN001
            warnings.append((str(message), dict(kwargs)))

        def debug(self, *args, **kwargs):  # noqa: ANN001
            return None

        def info(self, *args, **kwargs):  # noqa: ANN001
            return None

        def exception(self, *args, **kwargs):  # noqa: ANN001
            return None

    monkeypatch.setattr("app.services.rbc.icharging.get_logger", lambda: _DummyLogger())

    payload = _base_payload()
    payload["observations"]["charging_sessions"]["BB000SMI_1"] = {
        "power": 35.0,
        "electric_vehicle": "SM_EV_11",
    }
    payload["observations"]["charging_sessions"]["BB000SMI_2"] = {
        "power": 10.0,
        "electric_vehicle": "SM_EV_22",
    }
    payload["observations"]["electric_vehicles"] = {
        "SM_EV_11": {
            "SoC": 0.20,
            "flexibility": {
                "estimated_soc_at_departure": 0.90,
                "estimated_time_at_departure": "2026-03-01T12:00:00Z",
            },
        },
        "SM_EV_22": {
            "SoC": 0.55,
            "flexibility": {
                "estimated_soc_at_departure": 0.70,
                "estimated_time_at_departure": "2026-03-01T16:00:00Z",
            },
        },
    }

    actions = _run(sao_mamede_client, payload)
    assert MIN_TECHNICAL_KW <= actions["BB000SMI_1"] <= MAX_BB_KW
    assert actions["BB000SMI_2"] == pytest.approx(MIN_TECHNICAL_KW, rel=1e-6)
    assert any(msg == "rbc.exclusive_group_conflict" for msg, _ in warnings)


def test_price_forecast_do_not_change_dispatch(sao_mamede_client):
    payload_a = _base_payload()
    payload_a["observations"]["charging_sessions"]["BB000SMI_1"] = {
        "power": 25.0,
        "electric_vehicle": "SM_EV_03",
    }
    payload_a["observations"]["electric_vehicles"] = {
        "SM_EV_03": {
            "SoC": 0.35,
            "flexibility": {
                "estimated_soc_at_departure": 0.85,
                "estimated_time_at_departure": "2026-03-01T15:00:00Z",
            },
        }
    }

    payload_b = copy.deepcopy(payload_a)
    payload_b["observations"]["energy_price"]["values"] = [9.0] * 96
    payload_b["forecasts"] = {"dummy": {"values": [123.0, 456.0]}}

    actions_a = _run(sao_mamede_client, payload_a)
    actions_b = _run(sao_mamede_client, payload_b)
    assert actions_a == actions_b
