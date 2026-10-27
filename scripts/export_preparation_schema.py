"""Export the v2 local-preparation request without changing the v1 draft."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.preparation.models import PrepareCandidate
from scripts.export_contract_schemas import schema_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "contracts/v2/preparation.schema.json"
    text = json.dumps(schema_for(PrepareCandidate), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not path.exists() or path.read_text() != text:
            raise SystemExit("Local preparation schema is stale")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
