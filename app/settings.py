from __future__ import annotations

from pathlib import Path

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    manifest_path: Path | None = Field(
        default=None,
        env="MODEL_MANIFEST_PATH",
        description="Path to artifact_manifest.json. Optional for OTA loading.",
    )
    artifacts_dir: Path | None = Field(
        default=None,
        env="MODEL_ARTIFACTS_DIR",
        description="Directory containing manifest and artefacts. Optional if manifest path is absolute.",
    )
    agent_index: int | None = Field(
        default=None,
        ge=0,
        env="MODEL_AGENT_INDEX",
        description="Agent index within the manifest to serve. Optional for OTA loading.",
    )
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Log level for the inference service",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("manifest_path", pre=True)
    def _expand_manifest_path(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()

    @validator("artifacts_dir", pre=True)
    def _expand_artifacts_dir(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()


settings = Settings()
