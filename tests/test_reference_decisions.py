"""Reference responses captured from commit 8b1b2ac before the refactor."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


REFERENCE = json.loads((Path(__file__).parent / "fixtures/inference_reference.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: (
    Path(case["manifest"]).parent.name + ("-valid-community" if "features_patch" in case else "")
))
def test_recorded_sequence_preserves_decisions_and_http_errors(case):
    application = create_app(
        config=Settings(_env_file=None, manifest_path=None, allowed_bundle_root=None),
        gpu_probe=lambda: False,
    )
    with TestClient(application) as client:
        response = client.post("/admin/load", json={"manifest_path": case["manifest"]})
        assert response.status_code == 200
        records = json.loads(Path(case["messages"]).read_text())
        for step in case["sequence"]:
            features = {**records[step["index"]], **case.get("features_patch", {})}
            response = client.post(
                "/inference", json={"features": features},
                headers={"x-request-id": "iteration1-reference"},
            )
            assert response.status_code == step["status"]
            expected = step["body"]
            actual = response.json()
            if step["status"] != 200:
                assert actual == expected
                continue
            assert actual.keys() == expected.keys()
            assert actual["request_id"] == expected["request_id"]
            assert actual["actions"].keys() == expected["actions"].keys()
            for agent, actions in expected["actions"].items():
                assert actual["actions"][agent] == pytest.approx(
                    actions, abs=REFERENCE["absolute_tolerance"], rel=0,
                )
