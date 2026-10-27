"""Read a complete bundle candidate without mutating active state."""

from pathlib import Path
from typing import Any, Dict, Optional, Callable

from app.core.contracts import Pipeline, PipelineRecord
from app.utils.manifest import load_manifest
from app.logging import get_logger


class FilesystemBundleLoader:
    def __init__(self, pipeline_factory: Callable[..., Pipeline]) -> None:
        self.pipeline_factory = pipeline_factory

    def load(
        self,
        manifest_path: Path,
        artifacts_dir: Optional[Path],
        agent_index: int | None = None,
        alias_mapping_path: Optional[Path] = None,
    ) -> PipelineRecord:
        manifest_path = manifest_path.expanduser().resolve()
        root = artifacts_dir.expanduser().resolve() if artifacts_dir else manifest_path.parent
        manifest = load_manifest(manifest_path)
        effective_alias_mapping_path = _resolve_alias_mapping_path(
            alias_mapping_path=alias_mapping_path,
            manifest_metadata=manifest.metadata,
            artifacts_root=root,
        )

        loaded_agent_indices = sorted({int(artifact.agent_index) for artifact in manifest.agent.artifacts})
        if not loaded_agent_indices:
            raise ValueError("Manifest has no agent artifacts")

        default_agent_index = (
            int(agent_index) if agent_index is not None else loaded_agent_indices[0]
        )
        if default_agent_index not in loaded_agent_indices:
            raise ValueError(
                f"Default agent_index={default_agent_index} not found in manifest artifacts "
                f"{loaded_agent_indices}"
            )

        get_logger().info(
            "Loading pipeline",
            manifest_path=str(manifest_path),
            artifacts_dir=str(root),
            agent_index=default_agent_index,
            loaded_agent_indices=loaded_agent_indices,
        )

        alias_overrides = _load_alias_overrides(effective_alias_mapping_path, default_agent_index)
        pipelines: Dict[int, Pipeline] = {}
        for current_agent_index in loaded_agent_indices:
            pipelines[current_agent_index] = self.pipeline_factory(
                manifest=manifest,
                artifacts_root=root,
                agent_index=current_agent_index,
                alias_overrides=dict(alias_overrides),
            )

        record = PipelineRecord(
            pipelines=pipelines,
            manifest_path=manifest_path,
            artifacts_dir=root,
            agent_index=default_agent_index,
            loaded_agent_indices=loaded_agent_indices,
            alias_mapping_path=effective_alias_mapping_path,
        )
        return record


def _resolve_alias_mapping_path(
    alias_mapping_path: Optional[Path],
    manifest_metadata: Dict[str, Any],
    artifacts_root: Path,
) -> Path | None:
    if alias_mapping_path is not None:
        return alias_mapping_path.expanduser().resolve()

    metadata_alias_path = manifest_metadata.get("alias_mapping_path")
    if metadata_alias_path in (None, ""):
        return None
    if not isinstance(metadata_alias_path, str):
        raise ValueError("metadata.alias_mapping_path must be a string when provided")

    candidate = Path(metadata_alias_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (artifacts_root / candidate).resolve()


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
