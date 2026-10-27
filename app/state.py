"""Compatibility facade for scripts using the original process-wide store.

New application instances receive their own store through create_app().
"""

from pathlib import Path

from app.core.contracts import PipelineRecord
from app.core.state import InMemoryPipelineStore


class PipelineStore(InMemoryPipelineStore):
    def load(
        self,
        manifest_path: Path,
        artifacts_dir: Path | None,
        agent_index: int | None = None,
        alias_mapping_path: Path | None = None,
    ) -> PipelineRecord:
        # Keep legacy callers working without coupling the core store to files.
        from app.adapters.bundles import FilesystemBundleLoader
        from app.services.pipeline import InferencePipeline

        candidate = FilesystemBundleLoader(InferencePipeline).load(
            manifest_path, artifacts_dir, agent_index, alias_mapping_path
        )
        self.activate(candidate)
        return candidate

    def unload(self) -> None:
        from app.logging import get_logger

        record = self.get_record()
        if record:
            get_logger().info(
                "Unloading pipeline",
                manifest_path=str(record.manifest_path),
                agent_index=record.agent_index,
                loaded_agent_indices=record.loaded_agent_indices,
            )
        super().unload()


store = PipelineStore()
