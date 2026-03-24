from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import store

pytestmark = pytest.mark.skip(reason="Community em standby")


COMMUNITY_BUNDLE_DIR = Path("examples/icharging_community_boavista_sao_mamede_with_virtual_battery")
COMMUNITY_MANIFEST_PATH = COMMUNITY_BUNDLE_DIR / "artifact_manifest.json"
SINGLE_AGENT_BUNDLE_DIR = Path("examples/icharging_boavista_with_flex")
SINGLE_AGENT_MANIFEST_PATH = SINGLE_AGENT_BUNDLE_DIR / "artifact_manifest.json"
SINGLE_AGENT_ALIAS_PATH = SINGLE_AGENT_BUNDLE_DIR / "aliases.json"


def _community_features() -> dict:
    return {
        "timestamp": "2026-02-22T12:00:00Z",
        "sites": {
            "boavista": {
                "timestamp": "2026-02-22T12:00:00Z",
                "observations": {
                    "non_shiftable_load": 7.0,
                    "solar_generation": 2.0,
                    "energy_price": {
                        "values": [0.11] * 96,
                        "measurement_unit": "€/kWh",
                        "frequency_seconds": 900,
                        "horizon_seconds": 86400,
                    },
                    "charging_sessions": {
                        "AC000004_1": {"power": 2.0, "electric_vehicle": "11824"},
                        "AC000007_1": {"power": 2.0, "electric_vehicle": "11823"},
                    },
                    "electric_vehicles": {
                        "11824": {
                            "SoC": 0.3,
                            "flexibility": {
                                "estimated_soc_at_departure": 0.8,
                                "estimated_time_at_departure": "2026-02-22T15:00:00Z",
                            },
                        },
                        "11823": {
                            "SoC": 0.6,
                            "flexibility": {
                                "estimated_soc_at_departure": -1,
                                "estimated_time_at_departure": "",
                            },
                        },
                    },
                },
                "forecasts": {},
            },
            "sao_mamede": {
                "timestamp": "2026-02-22T12:00:00Z",
                "observations": {
                    "solar_generation": 12.0,
                    "energy_price": {
                        "values": [0.12] * 96,
                        "measurement_unit": "€/kWh",
                        "frequency_seconds": 900,
                        "horizon_seconds": 86400,
                    },
                    "charging_sessions": {
                        "BB000SMI_1": {"power": 0.0, "electric_vehicle": ""},
                        "BB000SMI_2": {"power": 0.0, "electric_vehicle": "SM_EV_01"},
                    },
                    "electric_vehicles": {
                        "SM_EV_01": {
                            "SoC": 0.35,
                            "flexibility": {
                                "estimated_soc_at_departure": 0.8,
                                "estimated_time_at_departure": "2026-02-22T16:00:00Z",
                            },
                        },
                    },
                    "virtual_battery": {"soc": 0.5},
                },
                "forecasts": {},
            },
        },
        "community": {
            "target_net_import_kw": 100.0,
            "current_net_import_kw": 140.0,
            "price_signal": {
                "values": [0.2] * 96,
                "measurement_unit": "€/kWh",
                "frequency_seconds": 900,
                "horizon_seconds": 86400,
            },
        },
    }


@pytest.fixture
def api_client():
    if store.is_configured():
        store.unload()
    client = TestClient(app)
    try:
        yield client
    finally:
        if store.is_configured():
            store.unload()


def test_admin_load_loads_all_agents_from_manifest(api_client):
    response = api_client.post(
        "/admin/load",
        json={
            "manifest_path": str(COMMUNITY_MANIFEST_PATH),
            "artifacts_dir": str(COMMUNITY_BUNDLE_DIR),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["agent_index"] == 0
    assert body["default_agent_index"] == 0
    assert set(body["loaded_agent_indices"]) == {0, 1}

    record = store.get_record()
    assert record is not None
    assert record.default_agent_index == 0
    assert set(record.loaded_agent_indices) == {0, 1}


def test_inference_selects_agent_by_body_index(api_client):
    load_resp = api_client.post(
        "/admin/load",
        json={
            "manifest_path": str(COMMUNITY_MANIFEST_PATH),
            "artifacts_dir": str(COMMUNITY_BUNDLE_DIR),
            "agent_index": 0,
        },
    )
    assert load_resp.status_code == 200

    features = _community_features()

    sm_resp = api_client.post(
        "/inference",
        json={
            "agent_index": 1,
            "features": features,
        },
    )
    assert sm_resp.status_code == 200
    sm_actions = sm_resp.json()["actions"]
    assert set(sm_actions.keys()) == {"0", "1"}
    assert set(sm_actions["1"].keys()) == {"BB000SMI_1", "BB000SMI_2", "virtual_battery_kw"}
    assert "AC000004_1" in sm_actions["0"]

    bv_resp = api_client.post(
        "/inference",
        json={
            "agent_index": 0,
            "features": features,
        },
    )
    assert bv_resp.status_code == 200
    bv_actions = bv_resp.json()["actions"]
    assert set(bv_actions.keys()) == {"0", "1"}
    assert "AC000004_1" in bv_actions["0"]
    assert set(bv_actions["1"].keys()) == {"BB000SMI_1", "BB000SMI_2", "virtual_battery_kw"}


def test_inference_defaults_to_default_agent_when_missing_agent_index(api_client):
    load_resp = api_client.post(
        "/admin/load",
        json={
            "manifest_path": str(COMMUNITY_MANIFEST_PATH),
            "artifacts_dir": str(COMMUNITY_BUNDLE_DIR),
            "agent_index": 1,
        },
    )
    assert load_resp.status_code == 200

    response = api_client.post(
        "/inference",
        json={
            "features": _community_features(),
        },
    )
    assert response.status_code == 200
    actions = response.json()["actions"]
    assert set(actions.keys()) == {"0", "1"}


def test_single_agent_bundle_still_works_without_agent_index(api_client):
    load_resp = api_client.post(
        "/admin/load",
        json={
            "manifest_path": str(SINGLE_AGENT_MANIFEST_PATH),
            "artifacts_dir": str(SINGLE_AGENT_BUNDLE_DIR),
            "alias_mapping_path": str(SINGLE_AGENT_ALIAS_PATH),
        },
    )
    assert load_resp.status_code == 200

    response = api_client.post(
        "/inference",
        json={
            "features": {
                "timestamp": "2026-02-22T12:00:00Z",
                "observations": {
                    "non_shiftable_load": 4.0,
                    "solar_generation": 1.5,
                    "energy_price": {
                        "values": [0.1] * 96,
                        "measurement_unit": "€/kWh",
                        "frequency_seconds": 900,
                        "horizon_seconds": 86400,
                    },
                    "charging_sessions": {
                        "AC000004_1": {"power": 1.8, "electric_vehicle": "11824"}
                    },
                    "electric_vehicles": {
                        "11824": {
                            "SoC": 0.3,
                            "flexibility": {
                                "estimated_soc_at_departure": 0.8,
                                "estimated_time_at_departure": "2026-02-22T14:00:00Z",
                            },
                        },
                    },
                },
                "forecasts": {},
            }
        },
    )

    assert response.status_code == 200
    actions = response.json()["actions"]
    assert set(actions.keys()) == {"0"}
    assert "AC000004_1" in actions["0"]


def test_info_and_health_expose_multi_agent_fields(api_client):
    load_resp = api_client.post(
        "/admin/load",
        json={
            "manifest_path": str(COMMUNITY_MANIFEST_PATH),
            "artifacts_dir": str(COMMUNITY_BUNDLE_DIR),
            "agent_index": 1,
        },
    )
    assert load_resp.status_code == 200

    info_resp = api_client.get("/info")
    assert info_resp.status_code == 200
    info = info_resp.json()
    assert info["agent_index"] == 1
    assert info["default_agent_index"] == 1
    assert set(info["loaded_agent_indices"]) == {0, 1}

    health_resp = api_client.get("/health")
    assert health_resp.status_code == 200
    health = health_resp.json()
    assert health["configured"] is True
    assert health["agent_index"] == 1
    assert health["default_agent_index"] == 1
    assert set(health["loaded_agent_indices"]) == {0, 1}
