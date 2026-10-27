"""Local bundle addressing extends the earlier request as an explicit v2 draft."""

from pathlib import PurePosixPath
from typing import Dict, Literal, Optional

from pydantic import Field, constr, validator

from app.contracts.v1.common import Contract, Name

Digest = constr(strict=True, regex=r"^[a-f0-9]{64}$")


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Expected a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("Absolute paths and parent traversal are forbidden")
    return value


class LocalBundle(Contract):
    directory: Name
    manifest_file: Name = "artifact_manifest.json"
    alias_file: Optional[Name] = None
    file_hashes: Dict[Name, Digest] = Field(default_factory=dict)

    _paths = validator("directory", "manifest_file", allow_reuse=True)(relative_path)

    @validator("alias_file")
    def alias_path(cls, value):
        return relative_path(value) if value is not None else None

    @validator("file_hashes")
    def hashes(cls, values):
        for key in values:
            relative_path(key)
        return values


class PrepareCandidate(Contract):
    schema_version: Literal["preparation.v2-draft"]
    configuration_revision: Name
    bundle: LocalBundle
    baseline_configuration_revision: Optional[Name] = None
