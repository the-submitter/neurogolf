from __future__ import annotations

import json
import shutil
from pathlib import Path

import jsonschema
import pytest

from neurogolf.arcgen.sampling import seed_schedule
from neurogolf.config.models import NeuroGolfSettings
from neurogolf.errors import ConfigurationError
from neurogolf.judge.reports import validate_report
from neurogolf.judge.sealed import load_sealed_manifest
from neurogolf.tasks.integrity import (
    PROTECTED_FILES,
    sha256_file,
    verify_integrity,
    write_manifest,
)


def test_repository_integrity_manifest_is_valid(settings: NeuroGolfSettings) -> None:
    result = verify_integrity(settings)
    assert result["valid"], result["failures"]
    assert result["checked_files"] == len(PROTECTED_FILES)
    manifest = json.loads(settings.paths.integrity_manifest.read_text())
    assert manifest["files"]["utils/neurogolf_utils.py"] == sha256_file(
        settings.paths.official_utils
    )


def test_integrity_manifest_detects_a_protected_change(
    settings: NeuroGolfSettings, tmp_path: Path
) -> None:
    root = tmp_path / "repository"
    for relative in PROTECTED_FILES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(settings.paths.repository_root / relative, destination)
    isolated = settings.model_copy(deep=True)
    isolated.paths.repository_root = root
    isolated.paths.integrity_manifest = root / "configs/integrity.json"
    write_manifest(isolated)
    assert verify_integrity(isolated)["valid"]
    with (root / "neurogolf/judge/gate.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# fixture tamper\n")
    result = verify_integrity(isolated)
    assert not result["valid"]
    assert result["failures"][0]["path"] == "neurogolf/judge/gate.py"


def test_sealed_seed_schedule_is_stable_bounded_and_separate(
    settings: NeuroGolfSettings,
) -> None:
    manifest = load_sealed_manifest(settings.paths.sealed_manifest)
    first = manifest.task_seeds("007bbfb7", 32)
    assert first == manifest.task_seeds("007bbfb7", 32)
    assert len(set(first)) == 32
    assert set(first).isdisjoint(seed_schedule("development", 1, 137, 32))
    assert first != manifest.task_seeds("00d62c1b", 32)
    with pytest.raises(ConfigurationError, match="provides"):
        manifest.base_seeds(manifest.count + 1)


def test_codex_result_schema_accepts_complete_output_and_rejects_extra_fields() -> None:
    payload = {
        "schema_version": "1.0",
        "task_num": 1,
        "task_id": "007bbfb7",
        "status": "reviewed",
        "summary": "Infrastructure fixture only.",
        "candidate_path": None,
        "candidate_sha256": None,
        "gate_result_path": None,
        "commands": [],
        "tests": [{"name": "fixture", "passed": True, "details": None}],
        "next_action": "none",
    }
    validate_report(payload, "codex_result.schema.json")
    with pytest.raises(jsonschema.ValidationError):
        validate_report({**payload, "unexpected": True}, "codex_result.schema.json")


def test_codex_result_schema_is_strict_structured_output_compatible(
    settings: NeuroGolfSettings,
) -> None:
    schema = json.loads(
        (settings.paths.repository_root / "neurogolf/schemas/codex_result.schema.json").read_text()
    )

    def check(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                properties = node.get("properties", {})
                assert isinstance(properties, dict)
                assert set(node.get("required", [])) == set(properties)
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(schema)
