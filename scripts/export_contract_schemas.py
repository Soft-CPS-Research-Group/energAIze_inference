"""Export JSON Schemas from the draft boundary models; run from repository root."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts.v1.decision import DecisionResult
from app.contracts.v1.execution import BundleInterface, CompatibilityReport, ConfigurationDraft, SafetyPolicy
from app.contracts.v1.installation import InstallationContext
from app.contracts.v1.lifecycle import ActivationRequest, ContractError, OperationReceipt, PreparationRequest


MODELS = {
    "installation": InstallationContext,
    "bundle-interface": BundleInterface,
    "configuration": ConfigurationDraft,
    "safety-policy": SafetyPolicy,
    "decision": DecisionResult,
    "compatibility": CompatibilityReport,
    "preparation": PreparationRequest,
    "activation": ActivationRequest,
    "operation": OperationReceipt,
    "error": ContractError,
}


def schema_for(model):
    schema = deepcopy(model.schema())
    seen = set()

    def visit(cls, node):
        if cls in seen:
            return
        seen.add(cls)
        for name, field in cls.__fields__.items():
            # Pydantic 1 accepts null but omits it from its JSON Schema output.
            if field.allow_none:
                prop = node["properties"][name]
                node["properties"][name] = {"anyOf": [prop, {"type": "null"}]}
            if isinstance(field.type_, type) and issubclass(field.type_, BaseModel):
                visit(field.type_, schema["definitions"][field.type_.__name__])

    visit(model, schema)
    return {"$schema": "http://json-schema.org/draft-07/schema#", **schema}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if committed schemas are stale")
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1] / "contracts/v1/schemas"
    if not args.check:
        target.mkdir(parents=True, exist_ok=True)
    for name, model in MODELS.items():
        path = target / f"{name}.schema.json"
        rendered = json.dumps(schema_for(model), indent=2, ensure_ascii=False) + "\n"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != rendered:
                raise SystemExit(f"Stale schema: {path}")
        else:
            path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
