"""Runtime state management for loaded inference pipelines."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from app.services.pipeline import InferencePipeline
from app.utils.manifest import load_manifest


@dataclass
class PipelineRecord:
    """Metadata describing the currently loaded pipeline."""

    pipeline: InferencePipeline
    manifest_path: Path
    artifacts_dir: Path
    agent_index: int
    alias_mapping_path: Path | None
    loaded_at: datetime = field(default_factory=datetime.utcnow)


class PipelineStore:
    """Container for the active inference pipeline and associated metadata."""

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

    def load(
        self,
        manifest_path: Path,
        artifacts_dir: Optional[Path],
        agent_index: int,
        alias_mapping_path: Optional[Path] = None,
    ) -> PipelineRecord:
        manifest_path = manifest_path.expanduser().resolve()
        root = artifacts_dir.expanduser().resolve() if artifacts_dir else manifest_path.parent

        logger.info(
            "Loading pipeline (manifest=%s, artifacts=%s, agent_index=%s)",
            manifest_path,
            root,
            agent_index,
        )
        alias_overrides = _load_alias_overrides(alias_mapping_path)
        manifest = load_manifest(manifest_path)
        pipeline = InferencePipeline(
            manifest=manifest,
            artifacts_root=root,
            agent_index=agent_index,
            alias_overrides=alias_overrides,
        )
        self._record = PipelineRecord(
            pipeline=pipeline,
            manifest_path=manifest_path,
            artifacts_dir=root,
            agent_index=agent_index,
            alias_mapping_path=alias_mapping_path,
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


def _load_alias_overrides(path: Optional[Path]) -> Dict[int, Dict[str, str]]:
    """Load per-agent feature alias overrides from a JSON file if provided."""

    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Alias mapping file not found: {path}")

    import json

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    overrides: Dict[int, Dict[str, str]] = {}
    for key, value in raw.items():
        try:
            agent_idx = int(key)
        except ValueError as exc:
            raise ValueError(f"Alias mapping keys must be integers (got {key!r})") from exc

        if isinstance(value, dict) and "feature_aliases" in value:
            alias_map = value.get("feature_aliases", {})
        elif isinstance(value, dict):
            alias_map = value
        else:
            raise ValueError(f"Alias mapping for agent {agent_idx} must be a dictionary")

        overrides[agent_idx] = {str(k): str(v) for k, v in alias_map.items()}

    logger.debug("Loaded alias overrides for agents: %s", list(overrides.keys()))
    return overrides
