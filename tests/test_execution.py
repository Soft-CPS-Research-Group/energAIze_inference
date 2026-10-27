"""Iteration 4: real prepared candidates, bound inputs, proposals and engine failures."""

import copy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.adapters.channel_message import ChannelMessageAdapter
from app.adapters.execution_context import compile_context
from app.application.evaluation import CandidateExecution
from app.contracts.v1.execution import ConfigurationDraft
from app.core.errors import ExecutionError, InputError
from app.main import create_app
from app.services.preprocessor import AgentPreprocessor, NoNormalization
from app.services.runtimes.onnx import OnnxAgentRuntime
from app.utils.manifest import Manifest
from tests.test_application import make_record, make_service
from tests.test_preparation import AUTH, setup, store_and_prepare


@pytest.fixture
def bound_setup(setup):
    settings, config, request, manifest, bundle, write = setup
    config["message"]["adapter_revision"] = "channels-1"
    config["bundle"]["inputs"][0]["name"] = "load_kw"
    config["bindings"]["inputs"][0]["signal_name"] = "load_kw"
    config["bundle"]["outputs"][0]["name"] = "dispatch"
    config["bindings"]["outputs"][0]["signal_name"] = "dispatch"
    manifest["environment"]["observation_names"] = [["load_kw"]]
    manifest["environment"]["action_names"] = ["dispatch"]
    (bundle / "policy.json").write_text(json.dumps({"default_actions": {"dispatch": 2},
        "rules": [{"if": {"load_kw": 1}, "actions": {"dispatch": 3}}]}))
    write()
    return setup


def run(client, candidate, features):
    return client.post(f'/admin/v1/candidates/{candidate["candidate_id"]}/evaluations',
                       json={"features": features}, headers={**AUTH, "x-request-id": "evaluation-test"})


def onnx_identity(fixture, encoder=None):
    import onnx
    from onnx import TensorProto, helper
    settings, config, request, manifest, bundle, write = fixture
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "identity",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)]), bundle / "agent.onnx")
    manifest["agent"]["artifacts"][0].update(format="onnx", path="agent.onnx")
    if encoder:
        manifest["environment"]["encoders"] = [[encoder]]
    config["bundle"]["runtime"] = "onnx"
    write()


def test_real_percepta_record_evaluates_with_channel_identity_and_preserves_active(bound_setup):
    settings, config, request, manifest, bundle, write = bound_setup
    # The stored sample's load values are energy-like. Use its explicit battery
    # SoC, rather than claiming those scalars are power in the synthetic context.
    config["installation"]["aggregate_limits"] = []
    channel = config["installation"]["channels"][0]
    channel["message_path"] = ["observations", "batteries", "B01", "SoC"]
    channel["meaning"].update(quantity="state_of_charge", unit="1", value_semantics="state", positive_direction="stored_charge")
    config["bundle"]["inputs"][0].update(name="battery_soc", meaning=copy.deepcopy(channel["meaning"]))
    config["bindings"]["inputs"][0]["signal_name"] = "battery_soc"
    manifest["environment"]["observation_names"] = [["battery_soc"]]
    (bundle / "policy.json").write_text('{"default_actions":{"dispatch":2},"rules":[]}')
    write()
    active = make_service()
    record = active.load(Path("unused"))
    message = json.loads(Path("contracts/v1/examples/legacy-request.json").read_text())["features"]
    original = copy.deepcopy(message)
    with TestClient(create_app(config=settings, service=active)) as client:
        candidate = store_and_prepare(client, config, request)
        response = run(client, candidate, message)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["schema_version"] == "evaluation.v1-draft"
        assert result["proposal_only"] and "actions" not in result
        assert result["request_id"] == "evaluation-test"
        assert result["proposed_actions"] == [{"action_name": "dispatch", "channel_id": "charger-1-setpoint",
            "equipment_id": "EVC01", "unit": "kW", "proposed_value": 2}]
        assert result["context"]["configuration_revision"] == "configuration-1"
        assert result["context"]["adapter_revision"] == "channels-1"
        assert result["context"]["producer_revision"] is None
        assert result["input_evidence"]["freshness"] == "not_assessed"
        assert active.store.get_record() is record
        assert active.infer({}) == {"0": {"power": 2}}
    assert message == original


def test_only_bound_fields_are_used_and_rules_get_canonical_names(bound_setup):
    settings, config, request, *_ = bound_setup
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        features = {"observations": {"non_shiftable_load": 1, "load_kw": 99, "new-charger": 100},
                    "load_kw": 200, "decisions": {"dispatch": 999}, "site_id": "another-site", "_id": {"$oid": "x"}}
        response = run(client, candidate, features)
        assert response.status_code == 200, response.text
        assert response.json()["proposed_actions"][0]["proposed_value"] == 3
        assert response.json()["context"]["site_id"] == "example-house"
        assert response.json()["input_evidence"]["timestamp"] is None
        assert len(response.json()["input_evidence"]["inputs"]) == 1


@pytest.mark.parametrize("value", [None, "1.0", True, {}, [], [1]])
def test_missing_and_invalid_required_values_never_default_to_zero(bound_setup, value):
    settings, config, request, *_ = bound_setup
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        response = run(client, candidate, {"observations": {"non_shiftable_load": value}})
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "input_type_invalid"
        assert response.json()["detail"]["path"] == "/features/observations/non_shiftable_load"
        missing = run(client, candidate, {"observations": {}})
        assert missing.status_code == 400
        assert missing.json()["detail"]["code"] == "input_missing"


def forecast_binding(fixture):
    _, config, _, _, _, write = fixture
    channel = config["installation"]["channels"][0]
    channel["role"] = "forecast"
    channel["message_path"] = ["forecasts", "ConsumptionForecastService", "consumption_total", "values", "0"]
    channel["meaning"].update(value_semantics="forecast", interval_seconds=900)
    config["bundle"]["inputs"][0]["meaning"] = copy.deepcopy(channel["meaning"])
    write()


def forecast_features(**overrides):
    descriptor = {"values": [5500, 6500], "measurement_unit": "W", "frequency_seconds": 900, "horizon_seconds": 1800}
    descriptor.update(overrides)
    return {"forecasts": {"ConsumptionForecastService": {"consumption_total": descriptor}}}


def test_conversion_and_encoder_run_once_and_raw_proposal_is_retained(bound_setup):
    settings, config, request, *_ = bound_setup
    forecast_binding(bound_setup)
    onnx_identity(bound_setup, {"type": "Normalize", "params": {"x_min": 0, "x_max": 10}})
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        assert candidate["state"] == "ready", candidate
        response = run(client, candidate, forecast_features())
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["proposed_actions"][0]["proposed_value"] == pytest.approx(0.55)
        evidence = result["input_evidence"]["inputs"][0]
        assert evidence["converted"] and evidence["source_unit"] == "W" and evidence["unit"] == "kW"
        assert evidence["interval_seconds"] == 900 and evidence["interval_source"] == "message"


@pytest.mark.parametrize("change,code", [
    ({"measurement_unit": "kWh"}, "unit_mismatch"), ({"measurement_unit": None}, "unit_invalid"),
    ({"frequency_seconds": 60}, "interval_mismatch"), ({"frequency_seconds": 0}, "interval_invalid"),
    ({"horizon_seconds": 900}, "horizon_mismatch"), ({"horizon_seconds": -1}, "horizon_invalid"),
    ({"values": []}, "input_missing"), ({"values": 1}, "forecast_invalid"),
])
def test_forecast_evidence_rejects_wrong_units_intervals_and_structure(bound_setup, change, code):
    settings, config, request, *_ = bound_setup
    forecast_binding(bound_setup)
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        response = run(client, candidate, forecast_features(**change))
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["code"] == code


@pytest.mark.parametrize("timestamp", ["2025-09-30T09:20:00+01:00", {"$date": "2025-09-30T08:20:00Z"},
    {"$date": {"$numberLong": "1759220400000"}}])
def test_timestamp_is_metadata_and_mongo_dates_are_normalized(bound_setup, timestamp):
    _, config, _, manifest, *_ = bound_setup
    context = compile_context(ConfigurationDraft.parse_obj(config), Manifest.parse_obj(manifest), "candidate")
    message = {"timestamp": timestamp, "observations": {"non_shiftable_load": 1}}
    prepared, evidence = ChannelMessageAdapter(context).prepare(message)
    assert prepared == {"load_kw": 1}
    assert evidence["timestamp"] == "2025-09-30T08:20:00+00:00"
    assert evidence["freshness"] == "not_assessed"


@pytest.mark.parametrize("timestamp", [None, "yesterday", "2026-01-01T12:00:00", {"$date": {}}, False])
def test_malformed_timestamp_is_explicit(bound_setup, timestamp):
    _, config, _, manifest, *_ = bound_setup
    context = compile_context(ConfigurationDraft.parse_obj(config), Manifest.parse_obj(manifest), "candidate")
    with pytest.raises(InputError, match="timestamp"):
        ChannelMessageAdapter(context).prepare({"timestamp": timestamp, "observations": {"non_shiftable_load": 1}})


def test_literal_dots_and_array_indices_do_not_alias_equipment(bound_setup):
    settings, config, request, _, _, write = bound_setup
    config["installation"]["channels"][0]["message_path"] = ["observations", "charger.1", "samples", "0"]
    write()
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        response = run(client, candidate, {"observations": {"charger.1": {"samples": [1]}, "charger": {"1": {"samples": [9]}}}})
        assert response.status_code == 200, response.text
        assert response.json()["proposed_actions"][0]["proposed_value"] == 3


def test_new_adapter_rejects_alias_sidecars_instead_of_applying_twice(bound_setup):
    settings, config, request, manifest, bundle, write = bound_setup
    (bundle / "aliases.json").write_text('{"non_shiftable_load": "load_kw"}')
    manifest["metadata"]["alias_mapping_path"] = "aliases.json"
    write()
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        assert candidate["state"] == "rejected"
        assert any(x["code"] == "alias_binding_conflict" for x in candidate["compatibility"]["issues"])


def test_candidate_evaluation_auth_and_no_silent_legacy_adapter_migration(setup):
    settings, config, request, *_ = setup
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        url = f'/admin/v1/candidates/{candidate["candidate_id"]}/evaluations'
        assert client.post(url, json={"features": {}}).status_code == 401
        response = client.post(url, json={"features": {}}, headers=AUTH)
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "evaluation_adapter_unsupported"


def test_source_changes_are_ignored_and_snapshot_tampering_rejected(bound_setup):
    settings, config, request, _, bundle, _ = bound_setup
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        (bundle / "policy.json").write_text('{"default_actions":{"dispatch":7}}')
        features = {"observations": {"non_shiftable_load": 1}}
        assert run(client, candidate, features).json()["proposed_actions"][0]["proposed_value"] == 3
        path = settings.configuration_store_dir / "candidates" / candidate["candidate_id"] / "files/policy.json"
        path.chmod(0o600)
        path.write_text('{"default_actions":{"dispatch":7}}')
        response = run(client, candidate, features)
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "hash_mismatch"
        assert not list(settings.configuration_store_dir.glob("evaluation-*"))


def test_onnx_proposals_outside_bounds_are_reported_without_safety_veto(bound_setup):
    settings, config, request, *_ = bound_setup
    onnx_identity(bound_setup)
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        response = run(client, candidate, {"observations": {"non_shiftable_load": 100}})
        assert response.status_code == 200, response.text
        assert response.json()["proposed_actions"][0]["proposed_value"] == 100
        assert response.json()["proposal_only"]


def fake_onnx(output=None, fail=False, bounds=None):
    def execute(*args):
        if fail:
            raise RuntimeError("native engine detail")
        return [output]
    session = SimpleNamespace(get_providers=lambda: ["test"], get_inputs=lambda: [SimpleNamespace(name="x")], run=execute)
    return OnnxAgentRuntime(0, session, AgentPreprocessor(["x"], [NoNormalization()]), ["action"], action_bounds=bounds)


def test_native_failure_is_a_technical_error_not_zero_success():
    runtime = fake_onnx(fail=True)
    with pytest.raises(ExecutionError) as error:
        runtime.infer({"x": 1})
    assert error.value.code == "onnx_execution_failed"
    assert "native engine detail" not in error.value.detail


@pytest.mark.parametrize("output", [np.array([[np.nan]], dtype=np.float32), np.array([[np.inf]], dtype=np.float32),
    np.array([[1, 2]], dtype=np.float32), np.array([1], dtype=np.float32), np.array([["a"]])])
def test_invalid_model_outputs_never_become_successful_decisions(output):
    with pytest.raises(ExecutionError):
        fake_onnx(output).infer({"x": 1})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "not-a-number"])
def test_encoded_invalid_input_does_not_reach_native_engine(value):
    with pytest.raises(InputError):
        fake_onnx(fail=True).infer({"x": value})


@pytest.mark.parametrize("low,high,value", [(0.16, 0.19, 0.17), (-0.16, 0.2, -0.155), (1.26, 1.26, 1.26)])
def test_rounding_cannot_cross_bounds(low, high, value):
    runtime = fake_onnx(np.array([[value]], dtype=np.float32), bounds={"low": [low], "high": [high]})
    raw = runtime.propose({"x": 1})["action"]
    result = runtime.infer({"x": 1})["action"]
    assert raw == pytest.approx(value)
    assert low <= result <= high


def test_active_record_is_captured_once_even_if_changed_during_preparation():
    first, second = make_record(value=1), make_record(value=9)
    service = make_service(first)
    service.load(Path("unused"))
    selected = []
    def prepare(features, pipeline):
        selected.append(pipeline)
        service.store.activate(second)
        return features
    service.prepare_input = prepare
    assert service.infer({}) == {"0": {"power": 1}}
    assert selected == [first.pipeline]
    assert service.infer({}) == {"0": {"power": 9}}


def test_compiled_context_is_detached_and_immutable(bound_setup):
    _, config, _, manifest, *_ = bound_setup
    parsed = ConfigurationDraft.parse_obj(config)
    context = compile_context(parsed, Manifest.parse_obj(manifest), "candidate")
    parsed.installation.channels[0].message_path.append("mutated")
    assert context.inputs[0].path == ("observations", "non_shiftable_load")
    with pytest.raises(FrozenInstanceError):
        context.identity.site_id = "another"


@pytest.mark.parametrize("proposal", [{}, {"dispatch": float("nan")}, {"dispatch": True}, {"other": 1}])
def test_common_output_contract_applies_to_rule_engines_too(bound_setup, proposal):
    _, config, _, manifest, *_ = bound_setup
    context = compile_context(ConfigurationDraft.parse_obj(config), Manifest.parse_obj(manifest), "candidate")
    execution = CandidateExecution(context, ChannelMessageAdapter(context), SimpleNamespace(propose=lambda features: proposal))
    with pytest.raises(ExecutionError):
        execution.evaluate({"observations": {"non_shiftable_load": 1}})


def test_unknown_categories_are_explicit_not_an_all_zero_onehot(bound_setup):
    from dataclasses import replace
    _, config, _, manifest, *_ = bound_setup
    context = compile_context(ConfigurationDraft.parse_obj(config), Manifest.parse_obj(manifest), "candidate")
    context = replace(context, inputs=(replace(context.inputs[0], value_type="string", classes=("eco", "fast")),))
    adapter = ChannelMessageAdapter(context)
    assert adapter.prepare({"observations": {"non_shiftable_load": "eco"}})[0] == {"load_kw": "eco"}
    with pytest.raises(InputError) as error:
        adapter.prepare({"observations": {"non_shiftable_load": "unknown"}})
    assert error.value.code == "category_unknown"


def test_nonfinite_required_scalar_is_rejected_by_bound_adapter(bound_setup):
    _, config, _, manifest, *_ = bound_setup
    context = compile_context(ConfigurationDraft.parse_obj(config), Manifest.parse_obj(manifest), "candidate")
    for value in (float("inf"), float("nan"), 10 ** 400):
        with pytest.raises(InputError):
            ChannelMessageAdapter(context).prepare({"observations": {"non_shiftable_load": value}})


def test_legacy_http_reports_native_failure_as_500_without_actions(bound_setup):
    settings, *_ = bound_setup
    service = make_service()
    record = service.load(Path("unused"))
    record.pipeline._agent = fake_onnx(fail=True)
    with TestClient(create_app(config=settings, service=service)) as client:
        response = client.post("/inference", json={"features": {"x": 1}})
        assert response.status_code == 500
        assert response.json()["detail"]["code"] == "onnx_execution_failed"
        assert "actions" not in response.json()
        response = client.post("/inference", json={"features": {}})
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "encoding_failed"


def test_prepared_execution_survives_restart_without_delivery_directory(bound_setup):
    import shutil
    settings, config, request, _, bundle, _ = bound_setup
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
    shutil.rmtree(bundle)
    with TestClient(create_app(config=settings)) as client:
        response = run(client, candidate, {"observations": {"non_shiftable_load": 1}})
        assert response.status_code == 200, response.text
        assert response.json()["proposed_actions"][0]["proposed_value"] == 3
        assert not client.get("/health").json()["configured"]


def test_non_decimal_onnx_bounds_can_now_be_prepared(bound_setup):
    settings, config, request, manifest, _, write = bound_setup
    onnx_identity(bound_setup)
    manifest["environment"]["action_bounds"] = [{"low": [0.16], "high": [0.19]}]
    write()
    with TestClient(create_app(config=settings)) as client:
        assert store_and_prepare(client, config, request)["state"] == "ready"


def test_documented_evaluation_demo_uses_actual_repository_message(tmp_path):
    from app.settings import Settings
    from tests.test_preparation import TOKEN
    root = Path("examples/evaluation_demo").resolve()
    settings = Settings(_env_file=None, manifest_path=None, allowed_bundle_root=root,
                        configuration_store_dir=tmp_path / "state", installation_site_id="example-house",
                        admin_api_token=TOKEN, configuration_file=root / "configuration.json",
                        preparation_request_file=root / "prepare.json")
    with TestClient(create_app(config=settings)) as client:
        status = client.get("/admin/v1/status", headers=AUTH).json()
        candidate_id = status["latest_operation"]["operation_id"]
        request = json.loads(Path("contracts/v1/examples/legacy-request.json").read_text())
        response = client.post(f"/admin/v1/candidates/{candidate_id}/evaluations", json=request, headers=AUTH)
        assert response.status_code == 200, response.text
        assert response.json()["proposed_actions"][0]["proposed_value"] == 2.5
        assert response.json()["context"]["candidate_id"] == candidate_id
        assert response.json()["input_evidence"]["inputs"][0]["unit"] == "1"
        assert not client.get("/health").json()["configured"]


@pytest.mark.parametrize("classes", ["ab", [1, True], [1, 1.0]])
def test_ambiguous_encoder_categories_are_rejected_during_preparation(bound_setup, classes):
    settings, config, request, *_ = bound_setup
    onnx_identity(bound_setup, {"type": "OnehotEncoding", "params": {"classes": classes}})
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        assert candidate["state"] == "rejected"
        assert any(issue["code"] == "encoder_classes_invalid" for issue in candidate["compatibility"]["issues"])


def test_fixed_tensor_order_survives_reordered_bindings_and_extra_channels(bound_setup):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    settings, config, request, manifest, bundle, write = bound_setup
    second = config["installation"]["channels"][1]
    config["bundle"]["inputs"].append({"name": "charger_kw", "meaning": copy.deepcopy(second["meaning"])})
    config["bindings"]["inputs"].insert(0, {"signal_name": "charger_kw", "channel_id": second["channel_id"]})
    unused = copy.deepcopy(second)
    unused.update(channel_id="alphabetically-first-unused", message_path=["observations", "unused"])
    config["installation"]["channels"].insert(0, unused)
    manifest["environment"]["observation_names"] = [["load_kw", "charger_kw"]]
    manifest["environment"]["encoders"] = [[{"type": "NoNormalization", "params": {}}] * 2]
    manifest["agent"]["artifacts"][0].update(format="onnx", path="agent.onnx")
    config["bundle"]["runtime"] = "onnx"
    graph = helper.make_graph([helper.make_node("MatMul", ["x", "weights"], ["y"])], "ordered",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(np.array([[1], [10]], dtype=np.float32), "weights")])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)]), bundle / "agent.onnx")
    write()
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        response = run(client, candidate, {"observations": {"unused": 999,
            "charging_sessions": {"EVC01": {"power": 2}}, "non_shiftable_load": 1}})
        assert response.status_code == 200, response.text
        assert response.json()["proposed_actions"][0]["proposed_value"] == 21


def test_evaluation_request_requires_message_object():
    from pydantic import ValidationError
    from app.contracts.v1.evaluation import EvaluationRequest
    with pytest.raises(ValidationError):
        EvaluationRequest(features=[["observations", {}]])


def test_encoder_categories_must_match_channel_type(bound_setup):
    settings, config, request, *_ = bound_setup
    onnx_identity(bound_setup, {"type": "OnehotEncoding", "params": {"classes": ["eco"]}})
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        assert candidate["state"] == "rejected"
        assert any(issue["code"] == "encoder_classes_type" for issue in candidate["compatibility"]["issues"])
