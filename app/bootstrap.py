"""Composition root for the production file, runtime and message adapters."""

from app.adapters.action_logging import log_single_agent_actions
from app.adapters.bundles import FilesystemBundleLoader
from app.adapters.legacy_input import prepare_legacy_input
from app.adapters.runtime_factory import ArtifactRuntimeFactory
from app.application.service import InferenceService
from app.core.contracts import ActivePipelines
from app.services.pipeline import InferencePipeline
from app.settings import Settings
from app.state import PipelineStore


def build_service(config: Settings, store: ActivePipelines | None = None) -> InferenceService:
    def build_pipeline(**kwargs):
        return InferencePipeline(
            **kwargs,
            runtime_factory=ArtifactRuntimeFactory(config.onnx_execution_providers, config.tensorrt_enabled),
        )

    return InferenceService(
        store=store if store is not None else PipelineStore(),
        loader=FilesystemBundleLoader(build_pipeline),
        prepare_input=prepare_legacy_input,
        observe_actions=log_single_agent_actions,
    )
