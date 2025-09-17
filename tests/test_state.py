import json
from pathlib import Path

import pytest

from app.state import store


def build_manifest(tmp_path: Path) -> Path:
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
