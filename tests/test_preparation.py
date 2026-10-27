import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.contracts.v1.execution import ConfigurationDraft
from app.main import create_app
from app.preparation.models import PrepareCandidate
from app.settings import Settings
from tests.test_application import make_record, make_service

TOKEN = "test-only-token-for-candidate-api-12345"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "bundles"
    bundle = root / "one"
    bundle.mkdir(parents=True)
    config = json.loads(Path("contracts/v1/examples/configuration.json").read_text())
    config["safety"] = dict(schema_version="safety-policy.v1-draft", revision="off-1", mode="off", profile_id="off.v1", rule_ids=[])
    config["bundle"]["inputs"][0]["name"] = "non_shiftable_load"
    config["bindings"]["inputs"][0]["signal_name"] = "non_shiftable_load"
    config["installation"]["equipment"][0]["equipment_id"] = "EVC01"
    for channel in config["installation"]["channels"]:
        if channel.get("equipment_id"):
            channel["equipment_id"] = "EVC01"
    manifest = dict(manifest_version=1, metadata={"site_id": "example-house"}, simulator={}, training={},
                    topology={"num_agents": 1}, algorithm={"name": "RuleBasedPolicy"},
                    environment=dict(observation_names=[["non_shiftable_load"]], encoders=[[{"type": "NoNormalization", "params": {}}]],
                                     action_names=["EVC01"], action_bounds=[{"low": [0], "high": [7.4]}], reward_function={"name": "test", "params": {}}),
                    agent={"artifacts": [{"agent_index": 0, "format": "rule_based", "path": "policy.json", "config": {"require_observations_envelope": True}}]})
    (bundle / "policy.json").write_text(json.dumps({"default_actions": {"EVC01": 2}, "rules": []}))

    def write_manifest():
        data = json.dumps(manifest).encode()
        (bundle / "artifact_manifest.json").write_bytes(data)
        config["bundle"]["manifest_sha256"] = hashlib.sha256(data).hexdigest()

    write_manifest()
    request = dict(schema_version="preparation.v2-draft", configuration_revision="configuration-1", bundle={"directory": "one"})
    settings = Settings(_env_file=None, manifest_path=None, allowed_bundle_root=root, configuration_store_dir=tmp_path / "state",
                        installation_site_id="example-house", admin_api_token=TOKEN)
    return settings, config, request, manifest, bundle, write_manifest


def store_and_prepare(client, config, request, key="test-1"):
    response = client.put(f'/admin/v1/configurations/{config["revision"]}', json=config, headers=AUTH)
    assert response.status_code in (200, 201), response.text
    response = client.post("/admin/v1/candidates", json=request, headers={**AUTH, "Idempotency-Key": key})
    assert response.status_code == 202, response.text
    op = response.json()["operation_id"]
    receipt = client.get(f"/admin/v1/operations/{op}", headers=AUTH).json()
    assert receipt["status"] == "succeeded", receipt
    return client.get(f"/admin/v1/candidates/{op}", headers=AUTH).json()


def codes(candidate):
    return {i["code"] for i in candidate["compatibility"]["issues"]}


def test_prepare_isolated_snapshot_preserves_active_and_is_idempotent(setup):
    settings, config, request, _, bundle, _ = setup
    active_service = make_service(make_record(value=3))
    active = active_service.load(Path("unused"))
    app = create_app(config=settings, service=active_service, gpu_probe=lambda: False)
    with TestClient(app) as client:
        result = store_and_prepare(client, config, request)
        assert result["state"] == "ready", result
        assert active_service.store.get_record() is active
        assert active_service.infer({}) == {"0": {"power": 3}}
        assert result["requires_revalidation_before_activation"]
        (bundle / "policy.json").write_text('{"modified": true}')
        again = store_and_prepare(client, config, request)
        assert again == result
        frozen = settings.configuration_store_dir / "candidates" / result["candidate_id"] / "files/policy.json"
        assert json.loads(frozen.read_text())["default_actions"] == {"EVC01": 2}
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == result["file_hashes"]["policy.json"]


def test_authentication_and_legacy_bypass_are_enforced(setup):
    settings, config, request, *_ = setup
    with TestClient(create_app(config=settings)) as client:
        assert client.put("/admin/v1/configurations/configuration-1", json=config).status_code == 401
        assert client.get("/admin/v1/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.post("/admin/load", json={"manifest_path": "/irrelevant"}).status_code == 401
        assert client.post("/admin/load", json={"manifest_path": "/irrelevant"}, headers=AUTH).status_code == 409
        assert client.post("/admin/unload", headers=AUTH).status_code == 409
        assert not client.get("/health").json()["configured"]
        assert client.get("/admin/v1/status", headers=AUTH).json()["activation_supported"] is False


def test_revision_conflict_and_site_scope_are_immutable(setup):
    settings, config, *_ = setup
    with TestClient(create_app(config=settings)) as client:
        url = "/admin/v1/configurations/configuration-1"
        assert client.put(url, json=config, headers=AUTH).status_code == 201
        assert client.put(url, json=config, headers=AUTH).status_code == 200
        changed = copy.deepcopy(config)
        changed["installation"]["equipment"][0]["label"] = "changed"
        assert client.put(url, json=changed, headers=AUTH).status_code == 409
        changed["revision"] = "other"
        changed["installation"]["site_id"] = changed["bundle"]["target_site_id"] = "another-house"
        assert client.put("/admin/v1/configurations/other", json=changed, headers=AUTH).status_code == 422
        assert client.get(url, headers=AUTH).json()["installation"]["equipment"][0]["label"] != "changed"


@pytest.mark.parametrize("field,code", [("artifact", "invalid_bundle_path"), ("alias", "invalid_bundle_path"), ("config", "invalid_bundle_path"), ("symlink", "invalid_bundle_path"), ("hash", "hash_mismatch")])
def test_files_cannot_escape_or_bypass_available_hashes(setup, field, code):
    settings, config, request, manifest, bundle, write = setup
    if field == "artifact": manifest["agent"]["artifacts"][0]["path"] = "../outside.json"
    if field == "alias": manifest["metadata"]["alias_mapping_path"] = "/tmp/outside.json"
    if field == "config": manifest["agent"]["artifacts"][0]["config"]["config_path"] = "../outside.json"
    if field == "symlink":
        original = bundle / "policy.json"
        original.unlink()
        original.symlink_to(bundle / "other.json")
        (bundle / "other.json").write_text("{}")
    if field == "hash": request["bundle"]["file_hashes"] = {"policy.json": "0" * 64}
    write()
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        assert candidate["state"] == "rejected"
        assert code in codes(candidate), candidate
        assert not client.get("/health").json()["configured"]


@pytest.mark.parametrize("change,code", [
    ("hash", "manifest_hash_mismatch"), ("version", "manifest_version_unsupported"),
    ("site", "manifest_site_mismatch"), ("agent", "agent_missing"),
    ("dimension", "dimension_mismatch"), ("encoder", "encoder_count_mismatch"),
    ("order", "input_order_mismatch"), ("strategy", "strategy_contract_unsupported"),
    ("policy", "policy_outputs_invalid"), ("provider", "runtime_mismatch"),
    ("topology", "topology_mismatch"), ("dynamic", "topology_unsupported"),
    ("mapping", "input_mapping_unsupported"), ("output_type", "output_type_unsupported"),
])
def test_manifest_and_runtime_contract_rejections(setup, change, code):
    settings, config, request, manifest, bundle, write = setup
    if change == "version": manifest["manifest_version"] = 2
    if change == "site": manifest["metadata"]["site_id"] = "different"
    if change == "agent": config["bundle"]["agent_index"] = 9
    if change == "dimension": manifest["agent"]["artifacts"][0]["observation_dimension"] = 2
    if change == "encoder": manifest["environment"]["encoders"] = [[]]
    if change == "order": manifest["environment"]["observation_names"] = [["other"]]
    if change == "strategy": manifest["agent"]["artifacts"][0]["config"]["strategy"] = "rh1_house_rbc_v1"
    if change == "policy": (bundle / "policy.json").write_text('{"default_actions": {}, "rules": []}')
    if change == "provider": config["bundle"]["runtime"] = "missing"
    if change == "topology": manifest["topology"]["num_agents"] = 2
    if change == "dynamic": manifest["environment"]["topology_mode"] = "dynamic"
    if change == "mapping": config["installation"]["channels"][0]["message_path"] = ["forecasts", "non_shiftable_load"]
    if change == "output_type":
        config["bundle"]["outputs"][0]["meaning"]["value_type"] = "number_vector"
        config["installation"]["channels"][2]["meaning"]["value_type"] = "number_vector"
        config["installation"]["aggregate_limits"] = []
    write()
    if change == "hash": config["bundle"]["manifest_sha256"] = "0" * 64
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        assert code in codes(candidate), candidate


@pytest.mark.parametrize("change,code", [("access", "access_unknown"), ("hash", "manifest_hash_unknown"), ("safety", "safety_profile_unsupported"), ("adapter", "adapter_unsupported"), ("charger", "uncovered_action")])
def test_unknown_knowledge_and_unsupported_capabilities_are_not_compatible(setup, change, code):
    settings, config, request, *_ = setup
    if change == "access": config["installation"]["channels"][2]["access"] = "unknown"
    if change == "hash": config["bundle"].pop("manifest_sha256")
    if change == "safety": config["safety"].update(mode="intervene", profile_id="limits", rule_ids=["clamp"])
    if change == "adapter": config["message"]["adapter_revision"] = "future-2"
    if change == "charger":
        new = copy.deepcopy(config["installation"]["channels"][2])
        new["channel_id"] = "second-setpoint"
        config["installation"]["channels"].append(new)
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        assert code in codes(candidate), candidate
        assert candidate["compatibility"]["status"] == ("insufficient_information" if change in ("access", "hash") else "incompatible")


def test_restart_keeps_revisions_and_marks_interrupted_work_uncertain(setup):
    settings, config, request, *_ = setup
    with TestClient(create_app(config=settings)) as client:
        service = client.app.state.preparation
        service.put(config["revision"], ConfigurationDraft.parse_obj(config))
        pending = service.submit("interrupted", PrepareCandidate.parse_obj(request))
    with TestClient(create_app(config=settings)) as client:
        assert client.get("/admin/v1/configurations/configuration-1", headers=AUTH).status_code == 200
        assert client.get(f"/admin/v1/operations/{pending}", headers=AUTH).json()["status"] == "uncertain"
        ready = store_and_prepare(client, config, request, "retry-new-key")
        assert ready["state"] == "ready"
    with TestClient(create_app(config=settings)) as client:
        restored = client.get(f'/admin/v1/candidates/{ready["candidate_id"]}', headers=AUTH).json()
        assert restored == ready
        assert not client.get("/health").json()["configured"]


def test_startup_files_use_same_put_prepare_and_do_not_activate(setup, tmp_path):
    settings, config, request, *_ = setup
    settings.configuration_file = tmp_path / "configuration.json"
    settings.preparation_request_file = tmp_path / "prepare.json"
    settings.configuration_file.write_text(json.dumps(config))
    settings.preparation_request_file.write_text(json.dumps(request))
    with TestClient(create_app(config=settings)) as client:
        status = client.get("/admin/v1/status", headers=AUTH).json()
        assert status["configurations"] == status["completed_candidates"] == 1
        assert not client.get("/health").json()["configured"]


def test_parallel_idempotency_and_single_owner(setup):
    settings, config, request, *_ = setup
    with TestClient(create_app(config=settings)) as client:
        service = client.app.state.preparation
        service.put(config["revision"], ConfigurationDraft.parse_obj(config))
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(lambda _: service.submit("same-key", PrepareCandidate.parse_obj(request)), range(8)))
            list(pool.map(service.prepare, ids))
        assert len(set(ids)) == 1
        assert service.candidate(ids[0])["state"] == "ready"
        with pytest.raises(BlockingIOError):
            with TestClient(create_app(config=settings)):
                pass


def test_composition_impact_uses_explicit_baseline(setup):
    settings, config, request, *_ = setup
    with TestClient(create_app(config=settings)) as client:
        store_and_prepare(client, config, request)
        new = copy.deepcopy(config)
        new["revision"] = "configuration-2"
        new["installation"]["revision"] = "installation-2"
        new["installation"]["equipment"][0]["label"] = "New display name"
        next_request = {**request, "configuration_revision": "configuration-2", "baseline_configuration_revision": "configuration-1"}
        result = store_and_prepare(client, new, next_request, "change-label")
        assert result["state"] == "ready"
        assert result["impact"]["required_change"] == "configuration"
        assert result["impact"]["changed_semantics"] == []


@pytest.mark.parametrize("width,provider,code", [(1, "CPUExecutionProvider", None), (2, "CPUExecutionProvider", "onnx_shape_mismatch"), (1, "MissingProvider", "provider_unavailable")])
def test_onnx_real_session_shapes_and_providers(setup, width, provider, code):
    import onnx
    from onnx import TensorProto, helper
    settings, config, request, manifest, bundle, write = setup
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "test",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, width])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, width])])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)]), bundle / "agent.onnx")
    manifest["agent"]["artifacts"][0].update(format="onnx", path="agent.onnx")
    config["bundle"]["runtime"] = "onnx"
    settings.onnx_execution_providers = [provider]
    write()
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        if code:
            assert code in codes(candidate), candidate
        else:
            assert candidate["state"] == "ready", candidate


@pytest.mark.parametrize("variant", ["valid", "escape", "duplicate", "basepath"])
def test_onnx_external_tensor_files_are_snapshotted_or_rejected(setup, variant):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    settings, config, request, manifest, bundle, write = setup
    graph = helper.make_graph([helper.make_node("MatMul", ["x", "weight"], ["y"])], "external",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
                              [numpy_helper.from_array(np.array([[1]], dtype=np.float32), "weight")])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.save_model(model, bundle / "agent.onnx", save_as_external_data=True,
                    all_tensors_to_one_file=True, location="weights.bin", size_threshold=0)
    if variant != "valid":
        model = onnx.load(bundle / "agent.onnx", load_external_data=False)
        if variant == "escape":
            for entry in model.graph.initializer[0].external_data:
                if entry.key == "location": entry.value = "../outside.bin"
        else:
            entry = model.graph.initializer[0].external_data.add()
            entry.key = "location" if variant == "duplicate" else "basepath"
            entry.value = "weights.bin" if variant == "duplicate" else "/tmp"
        (bundle / "agent.onnx").write_bytes(model.SerializeToString())
    manifest["agent"]["artifacts"][0].update(format="onnx", path="agent.onnx")
    config["bundle"]["runtime"] = "onnx"
    write()
    with TestClient(create_app(config=settings)) as client:
        result = store_and_prepare(client, config, request)
        if variant != "valid":
            assert "invalid_external_data" in codes(result)
        else:
            assert result["state"] == "ready", result
            assert "weights.bin" in result["file_hashes"]


def test_failed_preparation_keeps_active_pipeline_and_idempotency_conflicts(setup):
    settings, config, request, *_ = setup
    active_service = make_service()
    active = active_service.load(Path("unused"))
    config["bundle"]["manifest_sha256"] = "0" * 64
    with TestClient(create_app(config=settings, service=active_service)) as client:
        result = store_and_prepare(client, config, request)
        assert result["state"] == "rejected"
        assert active_service.store.get_record() is active
        changed = {**request, "bundle": {"directory": "another"}}
        response = client.post("/admin/v1/candidates", json=changed, headers={**AUTH, "Idempotency-Key": "test-1"})
        assert response.status_code == 409
        assert active_service.infer({}) == {"0": {"power": 2.0}}


def test_changed_semantics_with_same_bundle_is_explicitly_incompatible(setup):
    settings, config, request, *_ = setup
    with TestClient(create_app(config=settings)) as client:
        store_and_prepare(client, config, request)
        new = copy.deepcopy(config)
        new["revision"] = "configuration-2"
        new["installation"]["revision"] = "installation-2"
        new["installation"]["channels"][0]["meaning"]["positive_direction"] = "generation"
        new["bundle"]["inputs"][0]["meaning"]["positive_direction"] = "generation"
        next_request = {**request, "configuration_revision": "configuration-2", "baseline_configuration_revision": "configuration-1"}
        result = store_and_prepare(client, new, next_request, "changed-semantics")
        assert "interface_changed_without_bundle_evidence" in codes(result)
        assert result["impact"]["changed_semantics"] == ["base-load"]


def test_new_physical_limit_detects_policy_mismatch(setup):
    settings, config, request, *_ = setup
    config["installation"]["channels"][2]["physical_bounds"]["upper"] = 1
    with TestClient(create_app(config=settings)) as client:
        result = store_and_prepare(client, config, request)
        assert "physical_output_mismatch" in codes(result)


def test_startup_rejects_missing_auth_and_wrong_store_site(setup):
    settings, config, request, *_ = setup
    with TestClient(create_app(config=settings)) as client:
        store_and_prepare(client, config, request)
    settings.installation_site_id = "another-house"
    with pytest.raises(ValueError, match="different installation"):
        with TestClient(create_app(config=settings)):
            pass
    settings.admin_api_token = None
    with pytest.raises(ValueError, match="ADMIN_API_TOKEN"):
        with TestClient(create_app(config=settings)):
            pass


def test_documented_demo_and_schema_are_executable(tmp_path):
    from scripts.export_contract_schemas import schema_for
    assert json.loads(Path("contracts/v2/preparation.schema.json").read_text()) == schema_for(PrepareCandidate)
    root = Path("examples/preparation_demo").resolve()
    settings = Settings(_env_file=None, manifest_path=None, allowed_bundle_root=root,
                        configuration_store_dir=tmp_path / "state", installation_site_id="example-house",
                        admin_api_token=TOKEN, configuration_file=root / "configuration.json",
                        preparation_request_file=root / "prepare.json")
    with TestClient(create_app(config=settings)) as client:
        status = client.get("/admin/v1/status", headers=AUTH).json()
        assert status["latest_operation"]["status"] == "succeeded"
        candidate_id = status["latest_operation"]["operation_id"]
        result = client.get(f"/admin/v1/candidates/{candidate_id}", headers=AUTH).json()
        assert result["state"] == "ready"
        assert not client.get("/health").json()["configured"]


@pytest.mark.parametrize("path", ["../one", "/absolute", "one/../one", "one//nested", "one\\nested"])
def test_local_reference_rejects_noncanonical_or_escaping_paths(path):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PrepareCandidate.parse_obj({"schema_version": "preparation.v2-draft", "configuration_revision": "revision", "bundle": {"directory": path}})
