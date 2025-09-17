import json
from pathlib import Path

from app.services.pipeline import InferencePipeline
from app.utils.manifest import Manifest


def test_feature_alias_overrides(tmp_path: Path) -> None:
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
            "observation_names": [["feat1"]],
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
    model = helper.make_model(graph)

    (bundle_dir / "onnx_models").mkdir()
    onnx.save(model, bundle_dir / "onnx_models" / "agent_0.onnx")

    manifest_path = bundle_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    manifest_model = Manifest.parse_file(manifest_path)
    pipeline = InferencePipeline(
        manifest=manifest_model,
        artifacts_root=manifest_path.parent,
        agent_index=0,
        alias_overrides={"alias_feat1": "feat1"},
    )
    result = pipeline.inference({"alias_feat1": 0.8})
    assert result["0"]["action"] == 0.8
