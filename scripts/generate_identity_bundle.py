"""Generate a minimal identity ONNX bundle and manifest for manual testing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper

MANIFEST_TEMPLATE = {
    "manifest_version": 1,
    "metadata": {"experiment_name": "onnx_demo", "run_name": "identity"},
    "simulator": {},
    "training": {},
    "topology": {"num_agents": 1},
    "algorithm": {"name": "DummyONNX", "hyperparameters": {}},
    "environment": {
        "observation_names": [["feat"]],
        "encoders": [[{"type": "NoNormalization", "params": {}}]],
        "action_bounds": [[{"low": [0], "high": [1]}]],
        "action_names": ["action"],
        "reward_function": {"name": "RewardFunction", "params": {}}
    },
    "agent": {
        "format": "onnx",
        "artifacts": [
            {
                "agent_index": 0,
                "path": "onnx_models/agent_0.onnx",
                "observation_dimension": 1,
                "action_dimension": 1
            }
        ]
    }
}


def build_identity_model() -> onnx.ModelProto:
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "IdentityGraph", [input_tensor], [output_tensor])
    model = helper.make_model(
        graph,
        producer_name="energAIze-test",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    return model


def write_bundle(target_dir: Path) -> Path:
    bundle_dir = target_dir / "identity_bundle"
    onnx_dir = bundle_dir / "onnx_models"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    model = build_identity_model()
    onnx.save(model, onnx_dir / "agent_0.onnx")

    manifest_path = bundle_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(MANIFEST_TEMPLATE, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("examples"),
        help="Directory where the bundle directory will be created (default: examples)"
    )
    args = parser.parse_args()

    manifest_path = write_bundle(args.output.resolve())
    print(f"Generated bundle at {manifest_path}")


if __name__ == "__main__":
    main()
