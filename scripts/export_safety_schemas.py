"""Export/check v2 safety and lifecycle boundaries without changing v1 schemas."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_contract_schemas import schema_for
from app.contracts.v2.safety import ConfigurationV2, SafetyPolicyV2, EvaluationResultV2
from app.contracts.v2.lifecycle import Activate, InstallationUpdate, CycleRequest, Decision

MODELS = {"configuration": ConfigurationV2, "safety-policy": SafetyPolicyV2, "evaluation": EvaluationResultV2,
          "activation": Activate, "installation-update": InstallationUpdate,
          "inference-request": CycleRequest, "decision": Decision}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1] / "contracts/v2"
    for name, model in MODELS.items():
        path = target / f"{name}.schema.json"
        rendered = json.dumps(schema_for(model), indent=2, ensure_ascii=False) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != rendered:
                raise SystemExit(f"Stale schema: {path}")
        else:
            path.write_text(rendered)


if __name__ == "__main__":
    main()
