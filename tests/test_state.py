import json
from pathlib import Path

import pytest

from app.state import store


def build_manifest(tmp_path: Path, metadata_alias_path: str | None = None) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    manifest = {
        "manifest_version": 1,
        "metadata": {},
        "simulator": {},
        "training": {},
        "topology": {"num_agents": 1},
        "algorithm": {"name": "MADDPG", "hyperparameters": {}},
        "environment": {
            "observation_names": [["feat"]],
            "encoders": [[{"type": "NoNormalization", "params": {}}]],
            "action_bounds": [[{"low": [0], "high": [1]}]],
            "action_names": ["action"],
            "reward_function": {"name": "RewardFunction", "params": {}},
        },
        "agent": {
            "format": "onnx",
            "artifacts": [
                {
                    "agent_index": 0,
                    "path": "onnx_models/agent_0.onnx",
                }
            ],
        },
    }
    if metadata_alias_path is not None:
        manifest["metadata"]["alias_mapping_path"] = metadata_alias_path

    import onnx
    from onnx import helper, TensorProto

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "IdentityGraph", [input_tensor], [output_tensor])
    model = helper.make_model(
        graph,
        producer_name="unit-test",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )

    (bundle_dir / "onnx_models").mkdir()
    onnx.save(model, bundle_dir / "onnx_models" / "agent_0.onnx")

    manifest_path = bundle_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_store_load_unload(tmp_path):
    if store.is_configured():
        store.unload()
    manifest_path = build_manifest(tmp_path)
    alias_file = tmp_path / "aliases.json"
    alias_file.write_text('{"alias_feat": "feat"}', encoding="utf-8")
    record = store.load(manifest_path, None, 0, alias_file)
    assert record.pipeline is not None
    assert store.is_configured()
    store.unload()
    assert not store.is_configured()
    with pytest.raises(RuntimeError):
        store.get_pipeline()


def test_store_load_uses_manifest_alias_mapping_path_fallback(tmp_path):
    if store.is_configured():
        store.unload()
    manifest_path = build_manifest(tmp_path, metadata_alias_path="aliases.json")
    alias_file = manifest_path.parent / "aliases.json"
    alias_file.write_text('{"alias_feat": "feat"}', encoding="utf-8")

    record = store.load(manifest_path, manifest_path.parent, 0)
    assert record.alias_mapping_path == alias_file.resolve()
    assert store.get_pipeline(0).agent.feature_aliases["alias_feat"] == "feat"

    store.unload()


def test_store_load_raises_when_manifest_alias_mapping_missing(tmp_path):
    if store.is_configured():
        store.unload()
    manifest_path = build_manifest(tmp_path, metadata_alias_path="missing_aliases.json")

    with pytest.raises(FileNotFoundError, match="Alias mapping file not found"):
        store.load(manifest_path, manifest_path.parent, 0)
