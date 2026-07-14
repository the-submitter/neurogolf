from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neurogolf.config import load_settings
from neurogolf.config.loader import find_repository_root
from neurogolf.config.models import CodexConfig, GateConfig
from neurogolf.errors import ConfigurationError


def test_default_configuration_is_resolved(settings):
    assert settings.paths.repository_root.is_absolute()
    assert settings.paths.task_map.is_file()
    assert settings.scorer.channels == 10
    assert settings.scorer.opset is None
    assert settings.generation.sealed_seed_count == 2048
    assert settings.gate.timeout_seconds == 600
    assert settings.gate.auto_promote_eligible
    assert settings.codex.reasoning_effort == "xhigh"


def test_gate_timeout_fallback_matches_repository_default() -> None:
    assert GateConfig().timeout_seconds == 600


def test_environment_then_explicit_override_precedence(repository_root: Path, monkeypatch):
    monkeypatch.setenv("NEUROGOLF_GENERATION__GATE_FUZZ_COUNT", "7")
    from_environment = load_settings(repository_root=repository_root)
    assert from_environment.generation.gate_fuzz_count == 7
    explicit = load_settings(
        repository_root=repository_root,
        overrides={"generation": {"gate_fuzz_count": 3}},
    )
    assert explicit.generation.gate_fuzz_count == 3


def test_invalid_yaml_is_actionable(repository_root: Path, tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not-an-object\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="root must be an object"):
        load_settings(repository_root=repository_root, config_files=(bad,))


def test_repository_root_discovery(repository_root: Path):
    assert find_repository_root(repository_root / "neurogolf" / "judge") == repository_root


def test_codex_reasoning_efforts_include_max_and_ultra() -> None:
    assert CodexConfig().reasoning_effort == "xhigh"
    assert CodexConfig(reasoning_effort="max").reasoning_effort == "max"
    assert CodexConfig(reasoning_effort="ultra").reasoning_effort == "ultra"
    with pytest.raises(ValidationError):
        CodexConfig(reasoning_effort="unsupported")  # type: ignore[arg-type]
