import json
from pathlib import Path

import numpy as np

from app.services.pipeline import InferencePipeline
from app.utils.manifest import Manifest


def make_dummy_manifest(tmp_path: Path) -> Path:
    manifest = {
        "manifest_version": 1,
        "metadata": {"experiment_name": "demo", "run_name": "run"},
        "simulator": {},
        "training": {},
        "topology": {"num_agents": 1},
        "algorithm": {"name": "MADDPG", "hyperparameters": {}},
        "environment": {
            "observation_names": [["feat1"]],
            "encoders": [[{"type": "Normalize", "params": {"x_min": 0, "x_max": 1}}]],
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
                    "observation_dimension": 1,
                    "action_dimension": 1,
                }
            ],
        },
    }

    # Create dummy ONNX model (1 input -> 1 output identity)
    import onnx
    from onnx import helper, TensorProto

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "IdentityGraph", [input_tensor], [output_tensor])
    model = helper.make_model(graph)

    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "onnx_models").mkdir(parents=True)
    onnx.save(model, bundle_dir / "onnx_models" / "agent_0.onnx")

    manifest_path = bundle_dir / "artifact_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return manifest_path


def test_pipeline_infers(tmp_path):
    manifest_path = make_dummy_manifest(tmp_path)
    manifest = Manifest.parse_file(manifest_path)
    pipeline = InferencePipeline(manifest=manifest, artifacts_root=manifest_path.parent, agent_index=0)
    payload = {"feat1": 0.5}
    result = pipeline.inference(payload)
    assert "0" in result
    assert "action" in result["0"]
    assert np.isclose(result["0"]["action"], 0.5)
