from app.services.reward_service import RewardCalculator
from app.utils.manifest import Manifest


def build_manifest() -> Manifest:
    return Manifest.parse_obj(
        {
            "manifest_version": 1,
            "metadata": {},
            "simulator": {},
            "training": {},
            "topology": {"num_agents": 1},
            "algorithm": {"name": "MADDPG", "hyperparameters": {}},
            "environment": {
                "observation_names": [["net_electricity_consumption"]],
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
                        "observation_dimension": 1,
                        "action_dimension": 1,
                    }
                ],
            },
        }
    )


def test_reward_calculator_basic():
    calculator = RewardCalculator(build_manifest())
    rewards = calculator.calculate({"net_electricity_consumption": 5.0})
    assert rewards["0"] == -5.0
