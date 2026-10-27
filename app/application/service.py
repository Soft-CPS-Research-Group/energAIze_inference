"""Application orchestration, usable without FastAPI or a concrete runtime."""

import logging
from pathlib import Path

from app.core.contracts import (
    Actions, ActivePipelines, BundleLoader, Features, ObserveActions,
    PipelineRecord, PrepareInput,
)
from app.core.state import PipelineNotConfigured


class InferenceService:
    def __init__(
        self,
        store: ActivePipelines,
        loader: BundleLoader,
        prepare_input: PrepareInput,
        observe_actions: ObserveActions | None = None,
    ) -> None:
        self.store = store
        self.loader = loader
        self.prepare_input = prepare_input
        self.observe_actions = observe_actions

    def load(
        self,
        manifest_path: Path,
        artifacts_dir: Path | None = None,
        agent_index: int | None = None,
        alias_mapping_path: Path | None = None,
    ) -> PipelineRecord:
        # A failed candidate must not replace the active record.
        record = self.loader.load(manifest_path, artifacts_dir, agent_index, alias_mapping_path)
        self.store.activate(record)
        return record

    def unload(self) -> None:
        self.store.unload()

    def infer(self, features: Features, agent_index: int | None = None) -> Actions:
        # Capture once: an overlapping activation cannot mix this request's
        # selection, adapter and runtime with a later record.
        record = self.store.get_record()
        if record is None:
            raise PipelineNotConfigured("Model pipeline not configured")
        target = record.agent_index if agent_index is None else agent_index
        pipeline = record.pipelines.get(target)
        if pipeline is None:
            raise PipelineNotConfigured(f"Agent index {target} not configured")
        prepared = self.prepare_input(features, pipeline)
        actions = pipeline.inference(prepared)
        if self.observe_actions is not None:
            try:
                self.observe_actions(pipeline, prepared, actions)
            except Exception:
                # Diagnostics must not discard an otherwise successful decision.
                logging.getLogger(__name__).exception("inference.action_logging_failed")
        return actions
