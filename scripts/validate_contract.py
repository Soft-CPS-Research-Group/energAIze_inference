"""Validate a draft document. Success does not authorize activation or actuation."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError
from scripts.export_contract_schemas import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=MODELS)
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.file.read_text(encoding="utf-8"))
        MODELS[args.kind].parse_obj(data)
    except ValidationError as exc:
        print(exc.json())
        return 1
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    print(json.dumps({"contract_valid": True, "activation_checked": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
