"""Bundle device requirements are immutable and fail before activation."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.test_preparation import setup, store_and_prepare, codes
from tests.test_execution import bound_setup, onnx_identity, run


@pytest.mark.parametrize("device,expected", [
    ("cpu", None), ("gpu", "provider_unavailable"),
    ("auto", "execution_device_unsupported"), (None, "execution_device_unsupported"),
])
def test_bundle_device_is_enforced_during_preparation(bound_setup, device, expected):
    settings, config, request, manifest, _, write = bound_setup
    onnx_identity(bound_setup)
    manifest["agent"]["artifacts"][0]["config"]["execution_device"] = device
    write()
    # An explicit CPU bundle takes precedence over deployment defaults.
    settings.onnx_execution_providers = ["UnavailableDefault"]
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        if expected:
            assert candidate["state"] == "rejected"
            assert expected in codes(candidate), candidate
        else:
            assert candidate["state"] == "ready", candidate
            assert candidate["requested_onnx_providers"] == ["CPUExecutionProvider"]
            assert candidate["validated_providers"] == ["CPUExecutionProvider"]
            # Changing the source bundle cannot change the prepared snapshot.
            manifest["agent"]["artifacts"][0]["config"]["execution_device"] = "gpu"
            write()
            result = run(client, candidate, {"observations": {"non_shiftable_load": 2}})
            assert result.status_code == 200, result.text
            assert result.json()["providers"] == ["CPUExecutionProvider"]


def test_gpu_flag_does_not_claim_missing_native_runtime(bound_setup, monkeypatch):
    from app.adapters.tensorrt_session import TensorRTSession
    settings, config, request, manifest, _, write = bound_setup
    onnx_identity(bound_setup)
    settings.tensorrt_enabled = True
    manifest["agent"]["artifacts"][0]["config"]["execution_device"] = "gpu"
    write()
    def missing():
        raise RuntimeError("CUDA device unavailable")
    monkeypatch.setattr(TensorRTSession, "probe", missing)
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        assert candidate["state"] == "rejected"
        assert "provider_unavailable" in codes(candidate)
        assert candidate["validated_providers"] == []


def test_device_cannot_be_hidden_in_external_config(bound_setup):
    settings, config, request, manifest, bundle, write = bound_setup
    onnx_identity(bound_setup)
    manifest["agent"]["artifacts"][0]["config"]["config_path"] = "runtime.json"
    (bundle / "runtime.json").write_text(json.dumps({"execution_device": "gpu"}))
    write()
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        assert "device_override_unsupported" in codes(candidate)
