"""Runtime state management for loaded inference pipelines."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from app.services.pipeline import InferencePipeline
from app.utils.manifest import load_manifest
from app.logging import get_logger


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

        get_logger().info(
            "Loading pipeline",
            manifest_path=str(manifest_path),
            artifacts_dir=str(root),
            agent_index=agent_index,
        )
        alias_overrides = _load_alias_overrides(alias_mapping_path, agent_index)
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
            get_logger().info(
                "Unloading pipeline",
                manifest_path=str(self._record.manifest_path),
                agent_index=self._record.agent_index,
            )
        self._record = None

    def get_startup_time(self) -> datetime:
        return self._startup_time


store = PipelineStore()


def _load_alias_overrides(path: Optional[Path], agent_index: int) -> Dict[str, str]:
    """Load flat feature alias overrides from JSON sidecar files.

    The file must contain a single dictionary mapping alias keys to canonical
    feature names. Keys prefixed with an underscore are ignored so files can
    hold short comments (e.g. `_comment`).
    """

    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Alias mapping file not found: {path}")

    import json

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Alias mapping must be a JSON object mapping aliases to features")

    aliases: Dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"Alias mapping value for {key!r} must be scalar")
        aliases[str(key)] = str(value)

    get_logger().debug(
        "Loaded alias overrides",
        alias_count=len(aliases),
        alias_path=str(path),
        agent_index=agent_index,
    )
    return aliases
