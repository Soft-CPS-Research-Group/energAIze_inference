"""Run in the Percepta image, sharing only the smoke server's network and handoff.

Uses Percepta's real inference service, EnergAIze adapter and AsyncHTTP client.
Storage is an in-memory test sink; no broker, database or equipment is contacted.
The temporary handoff contains a generated test credential, never a pilot secret.
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, "/percepta")

from app.core.connectors.async_http_connector import AsyncHTTPConnector
from app.core.inference.inference_service import InferenceService
from app.core.output_handlers.async_http_output_handler import AsyncHTTPOutputHandler
from app.core.output_handlers.output_handler_base import OutputHandlerBase
from app.core.storages.runtime_data_storage.runtime_data_storage_base import RuntimeDataStorageBase
from app.extension.ml_models.energaize.energaize_ml_model_adapter import EnergAIzeMLModelAdapter
from app.extension.ml_models.energaize.energaize_model_execution_error_handling_logic import EnergAIzeModelExecutionErrorHandlingLogic


class MemoryStorage(RuntimeDataStorageBase):
    async def store(self, data):
        self.records.append(data)

    async def stop(self):
        pass


class NoActuation(OutputHandlerBase):
    async def send(self, message):
        self.called = True
        raise AssertionError("Observation smoke must not dispatch equipment commands")

    def stop(self):
        pass


async def main():
    handoff = Path("/handoff")
    spec = json.loads((handoff / "request.json").read_text())
    logger = logging.getLogger("percepta-smoke")
    logger.setLevel(logging.CRITICAL)
    connector = AsyncHTTPConnector({"url": spec["base"]})
    await connector.connect()
    try:
        handler = AsyncHTTPOutputHandler({"endpoints": [{
            "method": "POST", "path": spec["path"], "timeout_ms": 10000,
            "headers": {"Content-Type": "application/json",
                        "Authorization": "Bearer " + spec["credential"],
                        "X-Installation-Context": spec["context"]},
            "rpc_properties": {"enabled": True},
        }]}, connector, logger)
        environment = {"entities": {}}
        storage = MemoryStorage({}, "inference-gpu-smoke", "example-house", logger)
        storage.records = []
        sink = NoActuation(logger)
        sink.called = False
        service = InferenceService(
            EnergAIzeMLModelAdapter("example-house", environment, logger), handler,
            EnergAIzeModelExecutionErrorHandlingLogic(environment, logger), None,
            storage, sink, logger)
        await asyncio.wait_for(service.receive_message({"payload": spec["message"]}), timeout=30)
        assert len(storage.records) == 1, "Percepta did not store the cycle"
        assert not sink.called, "Percepta attempted to dispatch a command"
        record = storage.records[0]
        decision = record["decisions"]["decision"]
        assert decision["providers"] == [spec["provider"]]
        assert decision["activation"]["activation_id"] == spec["activation"]
        assert [a["proposed_value"] for a in decision["proposed_actions"]] == [6, 6]
        assert [a["value"] for a in decision["actions"]] == [6, 2]
        assert decision["safety"]["changes"][0]["applied"] is True
        assert record["model_inputs"] == spec["message"]
        result = {"percepta_service": "passed", "http_client": "AsyncHTTP",
                  "providers": decision["providers"], "storage": "in-memory test sink",
                  "proposed_kw": [6, 6], "final_kw": [6, 2]}
        (handoff / "result.tmp").write_text(json.dumps(result))
        (handoff / "result.tmp").replace(handoff / "result.json")
        print(json.dumps(result))
    finally:
        await connector.close()


if __name__ == "__main__":
    asyncio.run(main())
