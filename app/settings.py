from __future__ import annotations

from pathlib import Path
from typing import List

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
    log_json: bool = Field(
        default=False,
        env="LOG_JSON",
        description="Emit structured JSON logs when true",
    )
    alias_mapping_path: Path | None = Field(
        default=None,
        env="FEATURE_ALIAS_PATH",
        description="Optional path to JSON file with feature alias overrides",
    )
    onnx_execution_providers: List[str] = Field(
        default=["CPUExecutionProvider"],
        env="ONNX_EXECUTION_PROVIDERS",
        description="Comma-separated list of ONNX Runtime execution providers (order matters)",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

        @classmethod
        def parse_env_var(cls, field_name, raw_value):  # noqa: D401, ANN001
            """Custom parsing for select environment variables."""
            if field_name == "onnx_execution_providers":
                if raw_value in (None, ""):
                    return ["CPUExecutionProvider"]
                if isinstance(raw_value, str):
                    providers = [item.strip() for item in raw_value.split(",") if item.strip()]
                    return providers or ["CPUExecutionProvider"]
            return super().parse_env_var(field_name, raw_value)

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

    @validator("alias_mapping_path", pre=True)
    def _expand_alias_path(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()


settings = Settings()
