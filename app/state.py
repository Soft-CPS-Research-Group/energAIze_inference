from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from app.services.pipeline import InferencePipeline
from app.utils.manifest import load_manifest


@dataclass
class PipelineRecord:
    pipeline: InferencePipeline
    manifest_path: Path
    artifacts_dir: Path
    agent_index: int
    loaded_at: datetime = field(default_factory=datetime.utcnow)


class PipelineStore:
    def __init__(self) -> None:
        self._record: Optional[PipelineRecord] = None
        self._startup_time = datetime.utcnow()

    def is_configured(self) -> bool:
        return self._record is not None

    def get_pipeline(self) -> InferencePipeline:
        if not self._record:
            raise RuntimeError("Model pipeline not configured")
        return self._record.pipeline

    def get_record(self) -> Optional[PipelineRecord]:
        return self._record

    def load(self, manifest_path: Path, artifacts_dir: Optional[Path], agent_index: int) -> PipelineRecord:
        manifest_path = manifest_path.expanduser().resolve()
        root = artifacts_dir.expanduser().resolve() if artifacts_dir else manifest_path.parent

        logger.info(
            "Loading pipeline (manifest=%s, artifacts=%s, agent_index=%s)",
            manifest_path,
            root,
            agent_index,
        )
        manifest = load_manifest(manifest_path)
        pipeline = InferencePipeline(manifest=manifest, artifacts_root=root, agent_index=agent_index)
        self._record = PipelineRecord(
            pipeline=pipeline,
            manifest_path=manifest_path,
            artifacts_dir=root,
            agent_index=agent_index,
        )
        return self._record

    def unload(self) -> None:
        if self._record:
            logger.info(
                "Unloading pipeline (manifest=%s, agent_index=%s)",
                self._record.manifest_path,
                self._record.agent_index,
            )
        self._record = None

    def get_startup_time(self) -> datetime:
        return self._startup_time


store = PipelineStore()
