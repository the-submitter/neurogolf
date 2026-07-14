"""Deterministic YAML, environment, dotenv, and CLI configuration merging."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from neurogolf.config.models import (
    CodexConfig,
    GateConfig,
    GenerationConfig,
    NeuroGolfSettings,
    PathsConfig,
    ScorerConfig,
)
from neurogolf.errors import ConfigurationError


def _deep_merge(base: dict[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Could not read configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration root must be an object: {path}")
    return value


def find_repository_root(start: Path | None = None) -> Path:
    """Find the nearest directory containing both the task map and ARC-GEN."""

    current = (start or Path.cwd()).expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs" / "task_map.json").is_file() and (candidate / "ARC-GEN").is_dir():
            return candidate
    raise ConfigurationError(f"Could not locate NeuroGolf repository above {current}")


def load_settings(
    *,
    repository_root: Path | None = None,
    config_files: tuple[Path, ...] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> NeuroGolfSettings:
    """Load defaults < YAML (left-to-right) < env/.env < explicit overrides."""

    root = find_repository_root(repository_root)
    files = config_files or (root / "configs/solver.yaml", root / "configs/scorer.yaml")
    merged: dict[str, Any] = {"paths": {"repository_root": root}}
    for path in files:
        merged = _deep_merge(merged, _read_yaml(path if path.is_absolute() else root / path))
    try:
        settings = NeuroGolfSettings(
            _env_file=root / ".env",  # pyright: ignore[reportCallIssue]
            **merged,
        )
        if overrides:
            values = _deep_merge(settings.model_dump(), overrides)
            unknown = set(values) - {"paths", "generation", "gate", "codex", "scorer"}
            if unknown:
                raise ConfigurationError(
                    f"Unknown top-level configuration keys: {', '.join(sorted(unknown))}"
                )
            # Construct only after each section has been validated. Re-entering the
            # BaseSettings constructor here would re-apply environment sources above
            # the explicit overrides.
            settings = NeuroGolfSettings.model_construct(
                paths=PathsConfig.model_validate(values["paths"]),
                generation=GenerationConfig.model_validate(values["generation"]),
                gate=GateConfig.model_validate(values["gate"]),
                codex=CodexConfig.model_validate(values["codex"]),
                scorer=ScorerConfig.model_validate(values["scorer"]),
            )
        settings.paths = settings.paths.resolved(root)
    except ValidationError as error:
        raise ConfigurationError(f"Invalid NeuroGolf configuration: {error}") from error
    return settings
