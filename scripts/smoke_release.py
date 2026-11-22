"""Exercise a published container's real HTTP server with ONNX and native Percepta.

Run inside the target image, mounted read-only at /smoke_release.py. All model,
state and server logs are temporary. A nonzero exit prevents release promotion.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import math
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import onnx
from onnx import TensorProto, helper
import onnxruntime


def main():
    devices = os.environ.get("SMOKE_DEVICES", "default").split(",")
    assert all(d in ("default", "cpu", "gpu") for d in devices)
    device = devices[0]
    expected_arch = os.environ["EXPECTED_ARCH"]
    assert platform.machine() == expected_arch, platform.machine()
    assert "CPUExecutionProvider" in onnxruntime.get_available_providers()
    assert ZoneInfo("Europe/Lisbon").key == "Europe/Lisbon"
    examples = Path("/app/examples/safety_demo")
    config = json.loads((examples / "configuration.json").read_text())
    manifest = json.loads((examples / "bundle/artifact_manifest.json").read_text())
    config["bundle"]["runtime"] = "onnx"
    config["installation"]["channels"][0]["message_path"] = [
        "entities", "meter", "parameters", "load", "final_values", "0", "value"]
    manifest["algorithm"]["name"] = "ReleaseSmokeONNX"
    manifest["agent"]["artifacts"] = [
        {"agent_index": 0, "format": "onnx", "path": "model.onnx"}]
    if device != "default":
        manifest["agent"]["artifacts"][0]["config"] = {"execution_device": device}
    token, cycle_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    base = "http://127.0.0.1:8000"
    proc = None

    with tempfile.TemporaryDirectory(prefix="inference-release-") as tmp:
        root = Path(tmp)
        bundle = root / "bundles" / "bundle"
        bundle.mkdir(parents=True)
        graph = helper.make_graph(
            [helper.make_node("Add", ["x", "x"], ["double"]),
             helper.make_node("Concat", ["double", "double"], ["y"], axis=1)],
            "two_chargers",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
        onnx.checker.check_model(model)
        onnx.save(model, bundle / "model.onnx")
        (bundle / "artifact_manifest.json").write_text(json.dumps(manifest))
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}
        config["bundle"]["manifest_sha256"] = hashes["artifact_manifest.json"]
        env = {k: v for k, v in os.environ.items() if k not in (
            "MODEL_MANIFEST_PATH", "CONFIGURATION_FILE", "PREPARATION_REQUEST_FILE",
            "FEATURE_ALIAS_PATH")}
        env.update(INSTALLATION_SITE_ID="example-house", DEPLOYMENT_MODEL_ID="energy",
                   ALLOWED_BUNDLE_ROOT=str(bundle.parent), CONFIGURATION_STORE_DIR=str(root / "state"),
                   ADMIN_API_TOKEN=token, PERCEPTA_CYCLE_TOKEN=cycle_token,
                   ONNX_EXECUTION_PROVIDERS="CPUExecutionProvider", LOG_LEVEL="WARNING")
        log = root / "server.log"

        def start():
            with log.open("a") as output:
                return subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "app.main:app",
                     "--host", "127.0.0.1", "--port", "8000"],
                    env=env, cwd="/app", stdout=output, stderr=subprocess.STDOUT)

        def stop():
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        def api(path, body=None, method=None, key=None, credential=token, extra=None, status=200):
            headers = {"Authorization": "Bearer " + credential, "Content-Type": "application/json"}
            if key:
                headers["Idempotency-Key"] = key
            headers.update(extra or {})
            req = Request(base + path, json.dumps(body).encode() if body is not None else None,
                          headers, method=method)
            try:
                with urlopen(req, timeout=90) as response:
                    code, content = response.status, response.read()
            except HTTPError as error:
                code, content = error.code, error.read()
            assert code == status, (path, code, content.decode())
            return json.loads(content)

        def wait_server():
            for _ in range(180):
                assert proc.poll() is None, log.read_text()
                try:
                    api("/health")
                    return
                except (URLError, TimeoutError):
                    time.sleep(1)
            raise AssertionError("HTTP startup timed out")

        def resource(receipt):
            for _ in range(180):
                receipt = api("/admin/v1/operations/" + receipt["operation_id"])
                if receipt["status"] == "succeeded":
                    return receipt["resource_id"]
                assert receipt["status"] in ("pending", "running"), receipt
                time.sleep(1)
            raise AssertionError("Operation timed out")

        try:
            proc = start()
            wait_server()
            api("/ready", status=503)
            api("/admin/v1/configurations/" + config["revision"], config, method="PUT", status=201)
            candidate = resource(api("/admin/v1/candidates", {
                "schema_version": "preparation.v2-draft",
                "configuration_revision": config["revision"],
                "bundle": {"directory": "bundle", "file_hashes": hashes}},
                key="prepare", status=202))
            initial = api("/admin/v1/candidates/" + candidate)
            assert initial["state"] == "ready", initial
            context = api("/admin/v1/installation", {
                "schema_version": "installation-update.v1-draft",
                "expected_context_id": None, "expected_activation_id": None,
                "installation": config["installation"]}, method="PUT", key="context")["context_id"]
            activation = resource(api("/admin/v1/activations", {
                "schema_version": "activation.v2-draft", "candidate_id": candidate,
                "expected_context_id": context, "expected_activation_id": None,
                "timeout_seconds": 180}, key="activate", status=202))
            path = "/inference/sites/example-house/models/energy/observe"
            message = {
                "period_start": "2026-11-22 03:45:00 +0000",
                "period_end": "2026-11-22 04:00:00 +0000",
                "entities": {"meter": {"parameters": {"load": {
                    "measurement_unit": "W", "final_values": [{"value": 3000}]}}}}}
            headers = {"X-Installation-Context": context}

            def observe():
                api("/ready")
                assert api(path, credential=cycle_token, extra=headers)["ready"]
                result = api(path, message, credential=cycle_token, extra=headers)
                assert set(result) == {"decision"}
                decision = result["decision"]
                assert decision["activation"]["activation_id"] == activation
                assert decision["activation"]["context_id"] == context
                assert decision["providers"] == ["TensorRTNative" if device == "gpu" else "CPUExecutionProvider"]
                assert [a["proposed_value"] for a in decision["proposed_actions"]] == [6, 6]
                assert [a["value"] for a in decision["actions"]] == [6, 2]
                assert decision["safety"]["final_feasible"] is True
                assert decision["safety"]["changes"][0]["applied"] is True
                assert decision["input_evidence"]["inputs"][0]["converted"] is True
                api(path, message, credential=cycle_token,
                    extra={"X-Installation-Context": "stale"}, status=409)
                api("/admin/v1/status", credential=cycle_token, status=401)

            observe()
            def prepare_device(next_device, index):
                next_bundle = bundle.parent / ("bundle-" + str(index))
                next_bundle.mkdir()
                (next_bundle / "model.onnx").write_bytes((bundle / "model.onnx").read_bytes())
                manifest["agent"]["artifacts"][0]["config"] = {"execution_device": next_device}
                (next_bundle / "artifact_manifest.json").write_text(json.dumps(manifest))
                next_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in next_bundle.iterdir()}
                next_config = copy.deepcopy(config)
                next_config["revision"] = "device-" + str(index)
                next_config["bundle"]["revision"] = "bundle-" + str(index)
                next_config["bundle"]["manifest_sha256"] = next_hashes["artifact_manifest.json"]
                api("/admin/v1/configurations/" + next_config["revision"], next_config, method="PUT", status=201)
                next_candidate = resource(api("/admin/v1/candidates", {
                    "schema_version": "preparation.v2-draft",
                    "configuration_revision": next_config["revision"],
                    "bundle": {"directory": next_bundle.name, "file_hashes": next_hashes}},
                    key="prepare-" + str(index), status=202))
                return api("/admin/v1/candidates/" + next_candidate)

            for index, next_device in enumerate(devices[1:], 1):
                candidate = prepare_device(next_device, index)
                assert candidate["state"] == "ready", candidate
                # Candidate preparation must leave the previous engine serving.
                observe()
                previous = activation
                activation = resource(api("/admin/v1/activations", {
                    "schema_version": "activation.v2-draft", "candidate_id": candidate["candidate_id"],
                    "expected_context_id": context, "expected_activation_id": previous,
                    "timeout_seconds": 180}, key="activate-" + str(index), status=202))
                assert activation != previous
                device = next_device
                observe()

            if os.environ.get("EXPECT_GPU_UNAVAILABLE") == "1":
                rejected = prepare_device("gpu", len(devices))
                assert rejected["state"] == "rejected", rejected
                assert "provider_unavailable" in {i["code"] for i in rejected["compatibility"]["issues"]}
                observe()

            # Vary simultaneous inputs to detect shared GPU-buffer races.
            def concurrent_observe(index):
                sample = copy.deepcopy(message)
                value = 1000 + index * 100
                sample["entities"]["meter"]["parameters"]["load"]["final_values"][0]["value"] = value
                decision = api(path, sample, credential=cycle_token, extra=headers)["decision"]
                assert decision["providers"] == ["TensorRTNative" if device == "gpu" else "CPUExecutionProvider"]
                assert decision["activation"]["activation_id"] == activation
                assert all(math.isclose(a["proposed_value"], value / 500, abs_tol=1e-5)
                           for a in decision["proposed_actions"])
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(concurrent_observe, range(16)))
            if os.environ.get("PERCEPTA_SMOKE_HANDOFF"):
                handoff = Path(os.environ["PERCEPTA_SMOKE_HANDOFF"])
                spec = dict(base=base, path=path, credential=cycle_token, context=context,
                            activation=activation, message=message,
                            provider="TensorRTNative" if device == "gpu" else "CPUExecutionProvider")
                (handoff / "request.tmp").write_text(json.dumps(spec))
                (handoff / "request.tmp").chmod(0o600)
                (handoff / "request.tmp").replace(handoff / "request.json")
                for _ in range(180):
                    if (handoff / "result.json").exists():
                        result = json.loads((handoff / "result.json").read_text())
                        assert result["percepta_service"] == "passed"
                        assert result["providers"] == [spec["provider"]]
                        break
                    time.sleep(1)
                else:
                    raise AssertionError("Native Percepta client did not complete")
            stop()
            proc = start()
            wait_server()
            observe()
            print(json.dumps({
                "architecture": platform.machine(), "onnxruntime": onnxruntime.__version__,
                "provider": "TensorRTNative" if device == "gpu" else "CPUExecutionProvider",
                "devices": devices, "runtime_switches": len(devices) - 1,
                "concurrent_observations": 16,
                "gpu_unavailable_rejected": os.environ.get("EXPECT_GPU_UNAVAILABLE") == "1",
                "native_percepta_client": bool(os.environ.get("PERCEPTA_SMOKE_HANDOFF")),
                "startup": "passed",
                "activation": "passed", "native_percepta_observation": "passed",
                "proposed_kw": [6, 6], "final_kw": [6, 2], "restart_recovery": "passed",
                "timezone": "Europe/Lisbon", "input_conversion": "3000 W -> 3 kW"
            }))
        except BaseException:
            if log.exists():
                print(log.read_text()[-12000:], file=sys.stderr)
            raise
        finally:
            stop()


if __name__ == "__main__":
    main()
