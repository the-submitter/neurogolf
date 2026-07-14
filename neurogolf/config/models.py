"""Pydantic models for repository, scorer, gate, and orchestration settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class StrictModel(BaseModel):
    """Configuration section that rejects misspelled keys."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PathsConfig(StrictModel):
    repository_root: Path = Path(".")
    arcgen_root: Path = Path("ARC-GEN")
    kaggle_data_root: Path = Path("data")
    task_map: Path = Path("configs/task_map.json")
    official_utils: Path = Path("utils/neurogolf_utils.py")
    task_workspace_root: Path = Path("tasks")
    champion_root: Path = Path("champions")
    sealed_manifest: Path = Path("sealed/seeds.json")
    integrity_manifest: Path = Path("configs/integrity.json")

    def resolved(self, root: Path | None = None) -> PathsConfig:
        """Return a copy with every repository-relative path made absolute."""

        base = (root or self.repository_root).expanduser().resolve()
        values = self.model_dump()
        values["repository_root"] = base
        for name, value in values.items():
            if name == "repository_root":
                continue
            path = Path(value).expanduser()
            values[name] = path if path.is_absolute() else base / path
        return PathsConfig.model_validate(values)


class GenerationConfig(StrictModel):
    development_seed_count: PositiveInt = 64
    regression_seed_count: PositiveInt = 256
    gate_fuzz_count: PositiveInt = 512
    boundary_seed_count: PositiveInt = 64
    sealed_seed_count: PositiveInt = 2048
    max_attempts_per_seed: PositiveInt = 50


class GateConfig(StrictModel):
    timeout_seconds: PositiveInt = 600
    max_failure_examples: PositiveInt = 8
    require_public_pass: bool = True
    require_regression_pass: bool = True
    require_fuzz_pass: bool = True
    require_sealed_pass: bool = True
    auto_promote_eligible: bool = True


class CodexConfig(StrictModel):
    executable: str = "codex"
    model: str = "gpt-5.6-sol"
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max", "ultra"] = "xhigh"
    max_parallel_tasks: PositiveInt = 4
    timeout_seconds: PositiveInt = 7200
    max_epochs: PositiveInt = 4


class ScorerConfig(StrictModel):
    input_name: str = "input"
    output_name: str = "output"
    channels: PositiveInt = 10
    height: PositiveInt = 30
    width: PositiveInt = 30
    file_size_limit_bytes: PositiveInt = 1_509_949
    ir_version: PositiveInt = 10
    # None means that legality is decided by the pinned ONNX checker/runtime,
    # rather than by an artificial repository-level opset ceiling.
    opset: PositiveInt | None = None
    excluded_ops: tuple[str, ...] = (
        "LOOP",
        "SCAN",
        "NONZERO",
        "UNIQUE",
        "SCRIPT",
        "FUNCTION",
        "COMPRESS",
    )


class NeuroGolfSettings(BaseSettings):
    """Complete settings with environment variables above YAML precedence."""

    model_config = SettingsConfigDict(
        env_prefix="NEUROGOLF_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    paths: PathsConfig = Field(default_factory=PathsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    codex: CodexConfig = Field(default_factory=CodexConfig)
    scorer: ScorerConfig = Field(default_factory=ScorerConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del settings_cls
        return env_settings, dotenv_settings, init_settings, file_secret_settings
