"""HTTP and filesystem boundaries around the replaceable application core."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.adapters.bundles import FilesystemBundleLoader
from app.main import create_app
from app.settings import Settings
from tests.test_application import make_record, make_service
from tests.test_rule_based_pipeline import make_rbc_manifest


def test_two_apps_have_independent_configuration_load_and_unload():
    config = Settings(_env_file=None, manifest_path=None, allowed_bundle_root=None)
    first_service = make_service(make_record(value=1.0))
    second_service = make_service(make_record(value=7.0))
    first = TestClient(create_app(config=config, service=first_service, gpu_probe=lambda: False))
    second = TestClient(create_app(config=config, service=second_service, gpu_probe=lambda: False))
    assert first.post("/inference", json={"features": {}}).status_code == 503
    assert first.get("/info").status_code == 503
    assert first.post("/admin/unload").status_code == 409
    for client in (first, second):
        assert client.post("/admin/load", json={"manifest_path": "/unused"}).status_code == 200
    for client, value in ((first, 1.0), (second, 7.0)):
        response = client.post("/inference", json={"features": {}}, headers={"x-request-id": "test-id"})
        assert response.json() == {"actions": {"0": {"power": value}}, "request_id": "test-id"}
        assert client.get("/health").json()["providers"] == ["test"]
        assert client.get("/info").json()["algorithm"] == "test"
        assert client.post("/inference", json={"features": {}, "agent_index": 1}).status_code == 503
    assert first.post("/admin/unload").status_code == 200
    assert not first.get("/health").json()["configured"]
    assert second.get("/health").json()["configured"]
    assert second.post("/inference", json={"features": {}}).status_code == 200


def test_factory_defaults_do_not_share_store():
    config = Settings(_env_file=None, manifest_path=None)
    first = create_app(config=config)
    second = create_app(config=config)
    first.state.inference_service.store.activate(make_record())
    assert not second.state.inference_service.store.is_configured()


def test_startup_loads_through_injected_loader_and_shutdown_unloads():
    service = make_service()
    config = Settings(_env_file=None, manifest_path=Path("/unused"))
    with TestClient(create_app(config=config, service=service, gpu_probe=lambda: False)) as client:
        assert client.get("/health").json()["configured"]
        assert client.post("/inference", json={"features": {}}).status_code == 200
    assert not service.store.is_configured()


def test_each_app_uses_its_own_allowed_bundle_root():
    blocked = TestClient(create_app(
        config=Settings(_env_file=None, manifest_path=None, allowed_bundle_root=Path("/allowed")),
        service=make_service(),
    ))
    allowed = TestClient(create_app(
        config=Settings(_env_file=None, manifest_path=None, allowed_bundle_root=None),
        service=make_service(),
    ))
    assert blocked.post("/admin/load", json={"manifest_path": "/unused"}).status_code == 400
    assert allowed.post("/admin/load", json={"manifest_path": "/unused"}).status_code == 200


def test_failure_in_second_agent_does_not_publish_partial_bundle(tmp_path):
    path = make_rbc_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["agent"]["artifacts"].append({**manifest["agent"]["artifacts"][0], "agent_index": 1})
    path.write_text(json.dumps(manifest))
    service = make_service()
    active = service.load(Path("/unused"))
    built = []

    def build(**kwargs):
        built.append(kwargs["agent_index"])
        if kwargs["agent_index"] == 1:
            raise ValueError("second agent failed")
        return make_record().pipeline

    service.loader = FilesystemBundleLoader(build)
    with pytest.raises(ValueError, match="second agent failed"):
        service.load(path)
    assert built == [0, 1]
    assert service.store.get_record() is active
    assert service.infer({}) == {"0": {"power": 2.0}}
