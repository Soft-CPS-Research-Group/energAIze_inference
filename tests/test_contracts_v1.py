"""Draft contract structure and semantic invariants; no runtime activation."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.contracts.v1.decision import DecisionResult
from app.contracts.v1.execution import CompatibilityReport, ConfigurationDraft
from app.contracts.v1.installation import InstallationContext
from app.contracts.v1.lifecycle import ActivationRequest, OperationReceipt
from app.models.requests import InferenceRequest
from scripts.export_contract_schemas import MODELS, schema_for


ROOT = Path(__file__).resolve().parents[1] / "contracts/v1"


def example(name):
    return json.loads((ROOT / "examples" / name).read_text())


@pytest.mark.parametrize("name,model", [
    ("installation-one-charger.json", InstallationContext),
    ("installation-two-chargers.json", InstallationContext),
    ("configuration.json", ConfigurationDraft),
    ("impact-add-charger.json", CompatibilityReport),
    ("decision-off.json", DecisionResult),
    ("decision-observe.json", DecisionResult),
    ("decision-intervene.json", DecisionResult),
    ("activation-request.json", ActivationRequest),
])
def test_valid_examples_round_trip(name, model):
    parsed = model.parse_obj(example(name))
    assert model.parse_raw(parsed.json()) == parsed


@pytest.mark.parametrize("name,model,reason", [
    ("wrong-site.json", ConfigurationDraft, "target_site_id"),
    ("unknown-equipment.json", InstallationContext, "Unknown equipment_id"),
    ("opposite-direction.json", ConfigurationDraft, "meaning mismatch"),
    ("unreported-correction.json", DecisionResult, "reported applied corrections"),
])
def test_invalid_examples_have_explicit_errors(name, model, reason):
    with pytest.raises(ValidationError, match=reason):
        model.parse_obj(example("invalid/" + name))


@pytest.mark.parametrize("name,model", list(MODELS.items()))
def test_exported_schemas_are_current(name, model):
    assert json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text()) == schema_for(model)


@pytest.mark.parametrize("value", ["installation.v2", None, 1])
def test_unknown_versions_are_not_silently_treated_as_v1(value):
    data = example("installation-one-charger.json")
    data["schema_version"] = value
    with pytest.raises(ValidationError):
        InstallationContext.parse_obj(data)


def test_version_is_required_and_unknown_fields_are_rejected():
    data = example("configuration.json")
    del data["schema_version"]
    with pytest.raises(ValidationError):
        ConfigurationDraft.parse_obj(data)
    data = example("configuration.json")
    data["installation"]["credentials"] = {"token": "example-only"}
    with pytest.raises(ValidationError, match="extra fields not permitted"):
        ConfigurationDraft.parse_obj(data)


def test_paths_preserve_literal_keys_and_do_not_silently_trim_identifiers():
    data = example("installation-one-charger.json")
    data["channels"][1]["message_path"][2] = "EVC.01"
    assert InstallationContext.parse_obj(data).channels[1].message_path[2] == "EVC.01"
    data["channels"][1]["message_path"][2] = " EVC.01 "
    with pytest.raises(ValidationError):
        InstallationContext.parse_obj(data)


@pytest.mark.parametrize("field", ["equipment", "channels", "aggregate_limits"])
def test_duplicate_installation_identifiers_are_rejected(field):
    data = example("installation-one-charger.json")
    data[field].append(copy.deepcopy(data[field][0]))
    with pytest.raises(ValidationError, match="Duplicate"):
        InstallationContext.parse_obj(data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "7.4"])
def test_physical_limits_require_finite_json_numbers(value):
    data = example("installation-one-charger.json")
    data["channels"][2]["physical_bounds"]["upper"] = value
    with pytest.raises(ValidationError):
        InstallationContext.parse_obj(data)


def test_reversed_bounds_and_unknown_constraint_channels_are_rejected():
    data = example("installation-one-charger.json")
    data["channels"][2]["physical_bounds"] = {"lower": 8, "upper": 7}
    with pytest.raises(ValidationError, match="lower exceeds upper"):
        InstallationContext.parse_obj(data)
    data = example("installation-one-charger.json")
    data["aggregate_limits"][0]["terms"][0]["channel_id"] = "missing"
    with pytest.raises(ValidationError, match="Unknown constraint channel"):
        InstallationContext.parse_obj(data)


@pytest.mark.parametrize("field,value", [
    ("unit", "W"), ("quantity", "reactive_power"),
    ("positive_direction", "generation"), ("value_semantics", "cumulative"),
    ("interval_seconds", 30), ("value_type", "number_vector"),
])
def test_same_dimension_does_not_hide_a_semantic_change(field, value):
    data = example("configuration.json")
    data["installation"]["channels"][0]["meaning"][field] = value
    # The aggregate constraint is not relevant to this binding check.
    data["installation"]["aggregate_limits"] = []
    with pytest.raises(ValidationError, match="meaning mismatch"):
        ConfigurationDraft.parse_obj(data)


def test_missing_or_duplicate_action_bindings_are_rejected():
    data = example("configuration.json")
    data["bindings"]["outputs"] = []
    with pytest.raises(ValidationError, match="cover exactly"):
        ConfigurationDraft.parse_obj(data)
    data = example("configuration.json")
    data["bindings"]["outputs"].append(data["bindings"]["outputs"][0])
    with pytest.raises(ValidationError, match="Duplicate"):
        ConfigurationDraft.parse_obj(data)


def test_unknown_access_and_missing_hash_are_explicit_not_proven_compatible():
    data = example("configuration.json")
    data["installation"]["channels"][2]["access"] = "unknown"
    parsed = ConfigurationDraft.parse_obj(data)
    assert parsed.installation.channels[2].access == "unknown"
    assert parsed.bundle.manifest_sha256 is None
    assert not hasattr(parsed, "compatible")


def test_new_charger_does_not_extend_a_fixed_algorithm_implicitly():
    data = example("configuration.json")
    data["installation"] = example("installation-two-chargers.json")
    parsed = ConfigurationDraft.parse_obj(data)
    assert len(parsed.installation.equipment) == 2
    assert [item.name for item in parsed.bundle.outputs] == ["EVC01"]
    # This is a contract example, not a runtime evaluator. Preparation must detect coverage.
    report = CompatibilityReport.parse_obj(example("impact-add-charger.json"))
    assert report.status == "incompatible"
    assert report.issues[0].required_change == "bundle"


def test_message_profile_is_selected_by_configuration_not_claimed_by_producer():
    config = ConfigurationDraft.parse_obj(example("configuration.json"))
    assert config.message.producer_revision is None
    request = InferenceRequest.parse_obj(example("legacy-request.json"))
    source = json.loads(Path("examples/rh1_bundle/exemplos_mensagem_R-H-01_2303.json").read_text())[0]
    assert request.features == source
    assert "observations" in request.features
    assert "schema_version" not in request.features


@pytest.mark.parametrize("mode,final,applied", [("off", 9, None), ("observe", 9, False), ("intervene", 7.4, True)])
def test_safety_mode_distinguishes_proposal_suggestion_and_final_action(mode, final, applied):
    result = DecisionResult.parse_obj(example(f"decision-{mode}.json"))
    assert result.proposed_actions == {"EVC01": 9}
    assert result.legacy_response()["actions"] == {"0": {"EVC01": final}}
    if applied is None:
        assert result.safety.changes == []
    else:
        assert result.safety.changes[0].applied is applied


@pytest.mark.parametrize("mutation", ["wrong_proposal", "wrong_mode", "unknown_action", "nonfinite", "extra_final", "missing_change"])
def test_inconsistent_safety_audit_cannot_be_serialized_as_valid(mutation):
    data = example("decision-intervene.json")
    if mutation == "wrong_proposal": data["safety"]["changes"][0]["proposed_value"] = 10
    if mutation == "wrong_mode": data["safety"]["mode"] = "observe"
    if mutation == "unknown_action": data["safety"]["changes"][0]["action_name"] = "EVC02"
    if mutation == "nonfinite": data["actions"]["EVC01"] = float("nan")
    if mutation == "extra_final": data["actions"]["EVC02"] = 2
    if mutation == "missing_change": data["safety"]["changes"] = []
    with pytest.raises(ValidationError):
        DecisionResult.parse_obj(data)


def test_activation_requires_an_explicit_expected_state_including_null():
    data = example("activation-request.json")
    data["expected_activation_id"] = None
    assert ActivationRequest.parse_obj(data).expected_activation_id is None
    del data["expected_activation_id"]
    with pytest.raises(ValidationError, match="field required"):
        ActivationRequest.parse_obj(data)


def test_schema_exports_null_and_required_precondition_consistently():
    schema = schema_for(ActivationRequest)
    assert "expected_activation_id" in schema["required"]
    assert {"type": "null"} in schema["properties"]["expected_activation_id"]["anyOf"]


@pytest.mark.parametrize("data", [
    {"operation_id": "op", "status": "failed"},
    {"operation_id": "op", "status": "succeeded"},
    {"operation_id": "op", "status": "succeeded", "resource_id": "candidate", "error": {"code": "failed", "path": "/", "detail": "failure"}},
])
def test_operation_outcome_cannot_hide_failure_or_claim_missing_result(data):
    with pytest.raises(ValidationError):
        OperationReceipt.parse_obj(data)
