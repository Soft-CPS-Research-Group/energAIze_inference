"""Iteration 5: correction semantics, bounded joint constraints and public audit."""

import copy
from dataclasses import replace
from fractions import Fraction as F
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.adapters.execution_context import compile_context
from app.adapters.channel_message import ChannelMessageAdapter
from app.application.evaluation import CandidateExecution
from app.contracts.v2.safety import ConfigurationV2, EvaluationResultV2
from app.core.errors import ExecutionError
from app.core.safety import ActionLimit, AggregateLimit, SafetyPlan, correct, decimal
from app.main import create_app
from app.utils.manifest import Manifest
from tests.test_preparation import AUTH, setup, store_and_prepare
from tests.test_execution import bound_setup, run, onnx_identity


def plan(mode="intervene", aggregate=True):
    actions = tuple(ActionLimit(name, name + "-channel", name + "-bounds", F(1), F("7.4"), 1) for name in ("a", "b"))
    joint = AggregateLimit("site-budget", "site-import", "kW", F(11), (("load", F(1)),),
                           (("a", F(1)), ("b", F(1))), ("b", "a")) if aggregate else None
    return SafetyPlan("safety-2", mode, "physical-limits.v1", actions, joint)


def upgrade(config, mode="intervene"):
    config["schema_version"] = "configuration.v2-draft"
    config["installation"]["channels"][0]["physical_bounds"] = {"lower": 0, "upper": 30}
    config["safety"] = dict(schema_version="safety-policy.v2-draft", revision="safety-2", mode=mode,
        profile_id="physical-limits.v1", action_rules=[dict(rule_id="charger-bounds", channel_id="charger-1-setpoint", decimal_places=1)],
        aggregate_rule=dict(rule_id="site-budget", constraint_id="site-import", curtailment_order=["charger-1-setpoint"],
                            infeasible_behavior="minimum_actions_with_report"))
    return config


@pytest.mark.parametrize("mode", ["off", "observe", "intervene"])
def test_modes_retain_original_and_separate_suggested_feasibility(mode):
    proposal = {"a": 6, "b": 6}
    final, audit = correct(plan(mode), proposal, {"load": 3})
    assert proposal == {"a": 6, "b": 6}
    assert final == ({"a": 6, "b": 2} if mode == "intervene" else proposal)
    if mode == "off":
        assert not audit["checked"] and audit["changes"] == []
        assert audit["final_feasible"] is None
    else:
        assert audit["proposed_feasible"] is False
        assert audit["suggested_feasible"] is True
        assert audit["final_feasible"] == (mode == "intervene")
        assert audit["changes"] == [dict(action_name="b", proposed_value=6, corrected_value=2,
            applied=mode == "intervene", rule_ids=["site-budget"], reason="aggregate upper limit; explicit curtailment priority")]


def test_priority_and_explicit_nonzero_minima_fallback():
    final, audit = correct(plan(), {"a": 6, "b": 6}, {"load": 12})
    assert final == {"a": 1, "b": 1}
    assert not audit["final_feasible"]
    residual = audit["unresolved_constraints"][0]
    assert (residual["value"], residual["upper"], residual["excess"]) == (14, 11, 3)
    assert all(c["applied"] for c in audit["changes"])
    swapped = replace(plan(), aggregate=replace(plan().aggregate, order=("a", "b")))
    assert correct(swapped, {"a": 6, "b": 6}, {"load": 3})[0] == {"a": 2, "b": 6}


@pytest.mark.parametrize("value,expected", [(-99, -1.2), (-1.25, -1.2), (-0.15, -0.2), (0.15, 0.2), (1.25, 1.2), (99, 1.2)])
def test_negative_bounds_ties_and_rounding_revalidated(value, expected):
    profile = SafetyPlan("r", "intervene", "physical-limits.v1", (ActionLimit("a", "c", "r", F("-1.25"), F("1.25"), 1),))
    result, audit = correct(profile, {"a": value}, {})
    assert result == {"a": expected}
    assert audit["final_feasible"]


def test_unequal_coefficients_quantized_budget_and_determinism():
    profile = replace(plan(), aggregate=replace(plan().aggregate, upper=F("9.93"), actions=(("a", F("1.5")), ("b", F("0.7")))))
    # Exhaustively exercise negative/oversized proposals, fractional budgets and
    # impossible observations. Independent invariants, not a duplicate algorithm.
    for a, b, load in itertools.product([-10, 1, 2.05, 6.17, 100], [-4, 1.01, 5, 8], [0, 3.02, 12]):
        proposal, observations = {"a": a, "b": b}, {"load": load}
        result, audit = correct(profile, proposal, observations)
        assert (result, audit) == correct(profile, proposal, observations)
        assert all(1 <= value <= 7.4 and (decimal(value) * 10).denominator == 1 for value in result.values())
        total = decimal(load) + F("1.5") * decimal(result["a"]) + F("0.7") * decimal(result["b"])
        assert audit["final_feasible"] == (total <= F("9.93"))
        if not audit["final_feasible"]:
            assert result == {"a": 1, "b": 1}
        expected = dict(proposal)
        for change in audit["changes"]:
            assert change["proposed_value"] == proposal[change["action_name"]]
            expected[change["action_name"]] = change["corrected_value"]
        assert expected == result


def test_unchanged_actions_have_no_audit_noise():
    _, audit = correct(plan(), {"a": 2, "b": 3}, {"load": 1})
    assert audit["changes"] == [] and audit["proposed_feasible"]


@pytest.mark.parametrize("mode", ["off", "observe", "intervene"])
def test_api_v2_all_modes_and_restart_preserve_audit(bound_setup, mode):
    settings, config, request, manifest, bundle, write = bound_setup
    upgrade(config, mode)
    (bundle / "policy.json").write_text('{"default_actions":{"dispatch":20},"rules":[]}')
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        assert candidate["state"] == "ready", candidate
        response = run(client, candidate, {"observations": {"non_shiftable_load": 5}})
        assert response.status_code == 200, response.text
        result = response.json()
        EvaluationResultV2.parse_obj(result)
        assert result["evaluation_only"] and result["proposed_actions"][0]["proposed_value"] == 20
        assert result["actions"][0]["value"] == (6 if mode == "intervene" else 20)
        if mode != "off":
            change = result["safety"]["changes"][0]
            assert change["channel_id"] == "charger-1-setpoint" and change["equipment_id"] == "EVC01"
            assert change["rule_ids"] == ["charger-bounds", "site-budget"]
    with TestClient(create_app(config=settings)) as client:
        saved = client.get('/admin/v1/configurations/configuration-1', headers=AUTH)
        assert saved.status_code == 200 and saved.json()["schema_version"] == "configuration.v2-draft"
        assert run(client, candidate, {"observations": {"non_shiftable_load": 5}}).json() == result


def test_real_onnx_raw_output_outside_limits_is_corrected(bound_setup):
    settings, config, request, *_ = bound_setup
    upgrade(config)
    onnx_identity(bound_setup)
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        assert candidate["state"] == "ready", candidate
        result = run(client, candidate, {"observations": {"non_shiftable_load": 9}}).json()
        assert result["proposed_actions"][0]["proposed_value"] == 9
        assert result["actions"][0]["value"] == 2
        assert result["safety"]["final_feasible"]


def test_api_uncontrolled_excess_returns_decision_and_residual(bound_setup):
    settings, config, request, *_ = bound_setup
    upgrade(config)
    config["installation"]["channels"][-1]["physical_bounds"]["lower"] = 1
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        response = run(client, candidate, {"observations": {"non_shiftable_load": 12}})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["actions"][0]["value"] == 1
        assert result["safety"]["unresolved_constraints"][0]["excess"] == 2
        assert result["safety"]["final_feasible"] is False


@pytest.mark.parametrize("case,code", [
    ("missing_bounds", "safety_bounds_incomplete"), ("empty_grid", "safety_grid_empty"),
    ("large_bounds", "safety_bounds_unsupported"), ("missing_coverage", "safety_coverage_invalid"),
    ("missing_joint", "safety_aggregate_coverage"), ("lower_joint", "safety_aggregate_unsupported"),
    ("multiple_joint", "safety_aggregate_unsupported"), ("negative_control", "safety_coefficient_unsupported"),
    ("priority", "safety_priority_invalid"), ("missing_input_bounds", "safety_observation_bounds"),
    ("static_impossible", "safety_statically_infeasible"), ("legacy_adapter", "safety_adapter_unsupported"),
    ("noninstantaneous", "safety_observation_unsupported"),
])
def test_unsupported_profiles_rejected_before_loading(bound_setup, case, code):
    settings, config, request, *_ = bound_setup
    upgrade(config)
    channels, limits = config["installation"]["channels"], config["installation"]["aggregate_limits"]
    if case == "missing_bounds": channels[-1]["physical_bounds"] = {"upper": 7}
    if case == "empty_grid": channels[-1]["physical_bounds"] = {"lower": 0.01, "upper": 0.02}
    if case == "large_bounds": channels[-1]["physical_bounds"]["upper"] = 1e10
    if case == "missing_coverage": config["safety"]["action_rules"][0]["channel_id"] = "other"
    if case == "missing_joint": config["safety"]["aggregate_rule"] = None
    if case == "lower_joint": limits[0]["bounds"]["lower"] = 0
    if case == "multiple_joint": limits.append({**copy.deepcopy(limits[0]), "constraint_id": "another"})
    if case == "negative_control": limits[0]["terms"][-1]["coefficient"] = -1
    if case == "priority": config["safety"]["aggregate_rule"]["curtailment_order"] = ["base-load"]
    if case == "missing_input_bounds": del channels[0]["physical_bounds"]
    if case == "static_impossible": channels[0]["physical_bounds"]["lower"] = 15
    if case == "legacy_adapter": config["message"]["adapter_revision"] = "legacy-1"
    if case == "noninstantaneous":
        channels[0]["meaning"]["value_semantics"] = "state"
        config["bundle"]["inputs"][0]["meaning"]["value_semantics"] = "state"
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        assert candidate["state"] == "rejected", candidate
        assert code in {i["code"] for i in candidate["compatibility"]["issues"]}


def test_safety_snapshot_precedes_runtime_mutation_and_preserves_message(bound_setup):
    _, config, _, _, bundle, _ = bound_setup
    upgrade(config)
    context = compile_context(ConfigurationV2.parse_obj(config), Manifest.parse_obj(json.loads((bundle / "artifact_manifest.json").read_text())), "candidate")
    def mutate(prepared):
        prepared["load_kw"] = 999
        return {"dispatch": 20}
    execution = CandidateExecution(context, ChannelMessageAdapter(context), SimpleNamespace(propose=mutate))
    message = {"observations": {"non_shiftable_load": 5}}
    result = execution.evaluate(message)
    assert message == {"observations": {"non_shiftable_load": 5}}
    assert result["actions"][0]["value"] == 6


def test_audit_contract_rejects_unaudited_action_and_wrong_revision(bound_setup):
    settings, config, request, *_ = bound_setup
    upgrade(config)
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        result = run(client, candidate, {"observations": {"non_shiftable_load": 10}}).json()
    EvaluationResultV2.parse_obj(result)
    for field in ("value", "revision", "applied", "feasible", "identity"):
        broken = copy.deepcopy(result)
        if field == "value": broken["actions"][0]["value"] = 999
        if field == "revision": broken["safety"]["policy_revision"] = "different"
        if field == "applied": broken["safety"]["changes"][0]["applied"] = False
        if field == "feasible": broken["safety"]["suggested_feasible"] = False
        if field == "identity": broken["safety"]["changes"][0]["channel_id"] = "other"
        with pytest.raises(ValidationError): EvaluationResultV2.parse_obj(broken)


def test_canonical_units_used_before_onnx_normalization(bound_setup):
    settings, config, request, *_ = bound_setup
    upgrade(config)
    config["installation"]["channels"][0]["message_path"] = ["observations", "load", "values", "0"]
    onnx_identity(bound_setup, {"type": "Normalize", "params": {"x_min": 0, "x_max": 10}})
    message = {"observations": {"load": {"values": [10950], "measurement_unit": "W"}}}
    original = copy.deepcopy(message)
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        assert candidate["state"] == "ready", candidate
        result = run(client, candidate, message).json()
        assert result["proposed_actions"][0]["proposed_value"] == pytest.approx(1.095)
        assert result["actions"][0]["value"] == 0
        assert result["safety"]["final_feasible"]
        assert result["input_evidence"]["inputs"][0]["converted"]
    assert message == original


@pytest.mark.parametrize("case", ["exception", "nan", "names"])
def test_technical_failures_are_not_disguised_as_safety_corrections(bound_setup, monkeypatch, case):
    from app.services.runtimes.rule_based import RuleBasedRuntime
    settings, config, request, *_ = bound_setup
    upgrade(config)
    with TestClient(create_app(config=settings)) as client:
        candidate = store_and_prepare(client, config, request)
        def broken(self, features):
            if case == "exception": raise RuntimeError("private native detail")
            return {"dispatch": float("nan")} if case == "nan" else {"unknown": 3}
        monkeypatch.setattr(RuleBasedRuntime, "propose", broken)
        response = run(client, candidate, {"observations": {"non_shiftable_load": 3}})
        assert response.status_code == 500
        assert "actions" not in response.json() and "safety" not in response.json()
        assert "private native detail" not in response.text


@pytest.mark.parametrize("case", ["duplicate_rule", "duplicate_priority", "missing_fallback", "invalid_precision"])
def test_ambiguous_or_incomplete_policy_rejected_at_boundary(bound_setup, case):
    _, config, *_ = bound_setup
    upgrade(config)
    if case == "duplicate_rule": config["safety"]["aggregate_rule"]["rule_id"] = "charger-bounds"
    if case == "duplicate_priority": config["safety"]["aggregate_rule"]["curtailment_order"] *= 2
    if case == "missing_fallback": del config["safety"]["aggregate_rule"]["infeasible_behavior"]
    if case == "invalid_precision": config["safety"]["action_rules"][0]["decimal_places"] = True
    with pytest.raises(ValidationError): ConfigurationV2.parse_obj(config)


def test_negative_uncontrolled_coefficient_and_no_joint_constraint():
    profile = replace(plan(), aggregate=replace(plan().aggregate, observations=(("load", F(-1)),)))
    assert correct(profile, {"a": 6, "b": 6}, {"load": 3})[0] == {"a": 6, "b": 6}
    assert correct(plan(aggregate=False), {"a": 8, "b": 1.25}, {})[0] == {"a": 7.4, "b": 1.3}


def test_numeric_overflow_is_explicit_not_nonfinite_audit():
    profile = replace(plan(), aggregate=replace(plan().aggregate, observations=(("load", F("1e308")),)))
    with pytest.raises(ExecutionError, match="JSON numeric range"):
        correct(profile, {"a": 6, "b": 6}, {"load": 1e308})


def test_committed_demo_boots_from_files_and_evaluates_two_chargers(tmp_path):
    from app.settings import Settings
    from tests.test_preparation import TOKEN
    root = Path("examples/safety_demo").resolve()
    settings = Settings(_env_file=None, manifest_path=None, allowed_bundle_root=root, configuration_store_dir=tmp_path / "state",
        installation_site_id="example-house", admin_api_token=TOKEN, configuration_file=root / "configuration.json",
        preparation_request_file=root / "prepare.json")
    with TestClient(create_app(config=settings)) as client:
        request = json.loads((root / "prepare.json").read_text())
        candidate = store_and_prepare(client, json.loads((root / "configuration.json").read_text()), request)
        assert candidate["state"] == "ready", candidate
        for filename, expected, feasible in [("request.json", [6, 2], True), ("request-residual.json", [1, 1], False)]:
            result = run(client, candidate, json.loads((root / filename).read_text())["features"]).json()
            assert [a["value"] for a in result["actions"]] == expected
            assert result["safety"]["final_feasible"] == feasible
            EvaluationResultV2.parse_obj(result)
            if not feasible:
                assert result["safety"]["unresolved_constraints"][0]["excess"] == 3


def test_v2_schemas_match_exports():
    from scripts.export_safety_schemas import MODELS
    from scripts.export_contract_schemas import schema_for
    for name, model in MODELS.items():
        assert json.loads(Path(f"contracts/v2/{name}.schema.json").read_text()) == schema_for(model)
