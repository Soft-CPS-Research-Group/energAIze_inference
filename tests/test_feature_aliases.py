import json
from pathlib import Path

from app.services.pipeline import InferencePipeline
from app.utils.manifest import Manifest


def make_manifest_with_alias(tmp_path: Path) -> Path:
    manifest = {
        "manifest_version": 1,
        "metadata": {"experiment_name": "alias", "run_name": "alias"},
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
                    "config": {
                        "feature_aliases": {"alias_feat1": "feat1"}
                    }
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

    bundle_dir = tmp_path / "alias_bundle"
    (bundle_dir / "onnx_models").mkdir(parents=True)
    onnx.save(model, bundle_dir / "onnx_models" / "agent_0.onnx")

    manifest_path = bundle_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_feature_aliases(tmp_path):
    manifest_path = make_manifest_with_alias(tmp_path)
    manifest = Manifest.parse_file(manifest_path)
    pipeline = InferencePipeline(manifest=manifest, artifacts_root=manifest_path.parent, agent_index=0)
    result = pipeline.inference({"alias_feat1": 0.3})
    assert result["0"]["action"] == 0.3
