"""Use-case tests with no HTTP server, bundle files or execution engine."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.application.service import InferenceService
from app.core.contracts import PipelineRecord
from app.core.state import InMemoryPipelineStore, PipelineNotConfigured
from app.services.pipeline import InferencePipeline


def make_record(*, agent_index=0, value=2.0):
    runtime = SimpleNamespace(
        index=agent_index, providers=["test"], infer=lambda features: {"power": value},
    )
    manifest = SimpleNamespace(
        manifest_version=1, metadata={}, topology={"num_agents": 1},
        algorithm={"name": "test"},
        get_artifact=lambda index: SimpleNamespace(config={}),
        get_action_names=lambda index: ["power"],
    )
    pipeline = InferencePipeline(
        manifest, Path("/unused"), agent_index,
        runtime_factory=SimpleNamespace(build=lambda *args: runtime),
    )
    return PipelineRecord(
        pipelines={agent_index: pipeline}, manifest_path=Path("/unused/manifest.json"),
        artifacts_dir=Path("/unused"), agent_index=agent_index,
        loaded_agent_indices=[agent_index], alias_mapping_path=None,
    )


def make_service(record=None, **kwargs):
    record = record if record is not None else make_record()
    return InferenceService(
        store=InMemoryPipelineStore(),
        loader=SimpleNamespace(load=lambda *args: record),
        prepare_input=lambda features, pipeline: dict(features),
        **kwargs,
    )


def test_load_publish_execute_and_unload_without_external_dependencies():
    record = make_record(agent_index=3)
    service = make_service(record)
    assert not service.store.is_configured()
    assert service.load(Path("/unused")) is record
    assert service.infer({}) == {"3": {"power": 2.0}}
    assert service.infer({}, 3) == {"3": {"power": 2.0}}
    with pytest.raises(PipelineNotConfigured, match="Agent index 0 not configured"):
        service.infer({}, 0)
    service.unload()
    with pytest.raises(PipelineNotConfigured, match="Model pipeline not configured"):
        service.infer({})


def test_failed_candidate_keeps_previous_record_and_decisions():
    service = make_service()
    active = service.load(Path("/unused"))

    def fail(*args):
        raise ValueError("invalid candidate")

    service.loader = SimpleNamespace(load=fail)
    with pytest.raises(ValueError, match="invalid candidate"):
        service.load(Path("/new"))
    assert service.store.get_record() is active
    assert service.infer({}) == {"0": {"power": 2.0}}


def test_preparation_and_diagnostics_receive_the_selected_pipeline():
    observed = []
    service = make_service(observe_actions=lambda *args: observed.append(args))
    record = service.load(Path("/unused"))
    service.prepare_input = lambda features, pipeline: {"power": features["raw"] * 2}
    record.pipeline.agent.infer = lambda features: {"power": features["power"]}
    features = {"raw": 3.0}
    assert service.infer(features) == {"0": {"power": 6.0}}
    assert features == {"raw": 3.0}
    assert observed == [(record.pipeline, {"power": 6.0}, {"0": {"power": 6.0}})]


def test_diagnostic_failure_does_not_discard_decision(caplog):
    def fail(*args):
        raise ValueError("diagnostic failure")

    service = make_service(observe_actions=fail)
    service.load(Path("/unused"))
    assert service.infer({}) == {"0": {"power": 2.0}}
    assert "inference.action_logging_failed" in caplog.text


@pytest.mark.parametrize("stage", ["prepare", "runtime"])
def test_execution_errors_propagate_without_inventing_a_decision(stage):
    observed = []
    service = make_service(observe_actions=lambda *args: observed.append(args))
    record = service.load(Path("/unused"))

    def fail(*args):
        raise KeyError("missing measurement")

    if stage == "prepare":
        service.prepare_input = fail
    else:
        record.pipeline.agent.infer = fail
    with pytest.raises(KeyError, match="missing measurement"):
        service.infer({})
    assert observed == []


def test_core_and_injected_pipeline_do_not_import_external_frameworks():
    # A fresh interpreter proves this even if the full suite already imported HTTP/ONNX.
    script = '''
import sys
from pathlib import Path
from types import SimpleNamespace
from app.application.service import InferenceService
from app.core.state import InMemoryPipelineStore
from app.services.pipeline import InferencePipeline
runtime = SimpleNamespace(index=0, infer=lambda payload: {"action": 1.0})
pipeline = InferencePipeline(None, Path("unused"), 0,
    runtime_factory=SimpleNamespace(build=lambda *args: runtime))
assert pipeline.inference({}) == {"0": {"action": 1.0}}
for module in ("fastapi", "pydantic", "numpy", "onnx", "onnxruntime", "app.settings", "app.logging"):
    assert module not in sys.modules, module
'''
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
