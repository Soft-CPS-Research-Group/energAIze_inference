from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.state import store


COMMUNITY_BUNDLE_DIR = Path("examples/icharging_community_boavista_sao_mamede_rh1_with_virtual_battery")
COMMUNITY_MANIFEST_PATH = COMMUNITY_BUNDLE_DIR / "artifact_manifest.json"
SINGLE_AGENT_BUNDLE_DIR = Path("examples/icharging_boavista_with_flex")
SINGLE_AGENT_MANIFEST_PATH = SINGLE_AGENT_BUNDLE_DIR / "artifact_manifest.json"
SINGLE_AGENT_ALIAS_PATH = SINGLE_AGENT_BUNDLE_DIR / "aliases.json"


def _community_features() -> dict:
    return {
        "timestamp": "2026-03-03T12:00:00Z",
        "sites": {
            "boavista": {
                "timestamp": "2026-03-03T12:00:00Z",
                "observations": {
                    "non_shiftable_load": 8.0,
                    "solar_generation": 2.0,
                    "energy_price": {
                        "values": [0.12] * 96,
                        "measurement_unit": "€/kWh",
                        "frequency_seconds": 900,
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
                                "estimated_time_at_departure": "2026-03-03T15:00:00Z",
                            },
                        },
                        "11823": {"SoC": 0.6, "flexibility": {}},
                    },
                },
                "forecasts": {},
            },
            "sao_mamede": {
                "timestamp": "2026-03-03T12:00:00Z",
                "observations": {
                    "non_shiftable_load": 2.0,
                    "solar_generation": 15.0,
                    "energy_price": {
                        "values": [0.12] * 96,
                        "measurement_unit": "€/kWh",
                        "frequency_seconds": 900,
                    },
                    "charging_sessions": {
                        "BB000SMI_1": {"power": 0.0, "electric_vehicle": ""},
                        "BB000SMI_2": {"power": 10.0, "electric_vehicle": "SM_EV_1"},
                    },
                    "electric_vehicles": {
                        "SM_EV_1": {
                            "SoC": 0.4,
                            "flexibility": {
                                "estimated_soc_at_departure": 0.8,
                                "estimated_time_at_departure": "2026-03-03T16:00:00Z",
                            },
                        }
                    },
                    "virtual_battery": {"soc": 0.5},
                },
                "forecasts": {},
            },
            "rh1": {
                "timestamp": "2026-03-03T12:00:00Z",
                "observations": {
                    "non_shiftable_load": 2.5,
                    "solar_generation": 0.4,
                    "grid": {"import_limit_kw": 20.0, "export_limit_kw": 20.0},
                    "energy_price": {
                        "values": [0.18] + [0.10] * 95,
                        "measurement_unit": "€/kWh",
                        "frequency_seconds": 900,
                    },
                    "charging_sessions": {"EVC01": {"power": 2.0, "electric_vehicle": "EV01"}},
                    "electric_vehicles": {
                        "EV01": {
                            "SoC": 0.35,
                            "flexibility": {
                                "estimated_soc_at_departure": 0.8,
                                "estimated_time_at_departure": "2026-03-03T17:00:00Z",
                            },
                        }
                    },
                    "batteries": {"B01": {"SoC": 50}},
                },
                "forecasts": {},
            },
        },
        "community": {
            "price_signal": {
                "values": [0.19] + [0.12] * 95,
                "measurement_unit": "€/kWh",
                "frequency_seconds": 900,
            }
        },
    }


def _single_agent_features() -> dict:
    return {
        "timestamp": "2026-03-03T12:00:00Z",
        "observations": {
            "solar_generation": 1.0,
            "energy_price": {"values": [0.1] * 96, "measurement_unit": "€/kWh", "frequency_seconds": 900},
            "charging_sessions": {"AC000004_1": {"power": 1.6, "electric_vehicle": "11824"}},
            "electric_vehicles": {
                "11824": {
                    "SoC": 0.3,
                    "flexibility": {
                        "estimated_soc_at_departure": 0.8,
                        "estimated_time_at_departure": "2026-03-03T14:00:00Z",
                    },
                }
            },
        },
        "forecasts": {},
    }


def test_inference_community_returns_all_agents():
    if store.is_configured():
        store.unload()
    store.load(COMMUNITY_MANIFEST_PATH, COMMUNITY_BUNDLE_DIR, 0)
    client = TestClient(app)
    try:
        response = client.post("/inference", json={"agent_index": 2, "features": _community_features()})
        assert response.status_code == 200
        actions = response.json()["actions"]
        assert set(actions.keys()) == {"0", "1", "2"}
        assert "AC000004_1" in actions["0"]
        assert "BB000SMI_1" in actions["1"]
        assert "ev_charge_kw" in actions["2"]
    finally:
        if store.is_configured():
            store.unload()


def test_inference_community_missing_site_returns_400():
    if store.is_configured():
        store.unload()
    store.load(COMMUNITY_MANIFEST_PATH, COMMUNITY_BUNDLE_DIR, 0)
    client = TestClient(app)
    try:
        features = _community_features()
        del features["sites"]["rh1"]
        response = client.post("/inference", json={"features": features})
        assert response.status_code == 400
    finally:
        if store.is_configured():
            store.unload()


def test_inference_single_agent_bundle_unchanged():
    if store.is_configured():
        store.unload()
    store.load(
        SINGLE_AGENT_MANIFEST_PATH,
        SINGLE_AGENT_BUNDLE_DIR,
        0,
        SINGLE_AGENT_ALIAS_PATH,
    )
    client = TestClient(app)
    try:
        response = client.post("/inference", json={"features": _single_agent_features()})
        assert response.status_code == 200
        actions = response.json()["actions"]
        assert set(actions.keys()) == {"0"}
        assert "AC000004_1" in actions["0"]
    finally:
        if store.is_configured():
            store.unload()
