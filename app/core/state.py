"""Process-local active state. Loading and persistence are separate concerns."""

from datetime import datetime

from app.core.contracts import Pipeline, PipelineRecord


class PipelineNotConfigured(RuntimeError):
    """No active model, or the requested agent is not loaded."""


class InMemoryPipelineStore:
    def __init__(self) -> None:
        self._record: PipelineRecord | None = None
        self._startup_time = datetime.utcnow()

    def is_configured(self) -> bool:
        return self._record is not None

    def get_pipeline(self, agent_index: int | None = None) -> Pipeline:
        record = self._record
        if record is None:
            raise PipelineNotConfigured("Model pipeline not configured")
        target = record.agent_index if agent_index is None else int(agent_index)
        pipeline = record.pipelines.get(target)
        if pipeline is None:
            raise PipelineNotConfigured(f"Agent index {target} not configured")
        return pipeline

    def get_record(self) -> PipelineRecord | None:
        return self._record

    def activate(self, record: PipelineRecord) -> None:
        self._record = record

    def unload(self) -> None:
        self._record = None

    def get_startup_time(self) -> datetime:
        return self._startup_time
