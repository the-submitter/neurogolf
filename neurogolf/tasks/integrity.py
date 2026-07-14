"""SHA-256 and Git-revision verification for protected repository artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from neurogolf.config.models import NeuroGolfSettings
from neurogolf.errors import IntegrityError

JUDGE_VERSION = "neurogolf-judge-v1"
PROTECTED_FILES = (
    "utils/neurogolf_utils.py",
    "configs/task_map.json",
    "configs/solver.yaml",
    "configs/scorer.yaml",
    "sealed/seeds.json",
    "neurogolf/config/__init__.py",
    "neurogolf/config/models.py",
    "neurogolf/config/loader.py",
    "neurogolf/tasks/models.py",
    "neurogolf/tasks/loader.py",
    "neurogolf/tasks/mapping.py",
    "neurogolf/tasks/integrity.py",
    "neurogolf/arcgen/importer.py",
    "neurogolf/arcgen/generator.py",
    "neurogolf/arcgen/sampling.py",
    "neurogolf/arcgen/tracing.py",
    "neurogolf/judge/official_adapter.py",
    "neurogolf/judge/candidate_runner.py",
    "neurogolf/judge/legality.py",
    "neurogolf/judge/gate.py",
    "neurogolf/judge/counterexamples.py",
    "neurogolf/judge/sealed.py",
    "neurogolf/judge/promotion.py",
    "neurogolf/judge/registry.py",
    "neurogolf/judge/reports.py",
    "neurogolf/schemas/gate_result.schema.json",
    "neurogolf/schemas/failure_packet.schema.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise IntegrityError(f"Could not hash {path}: {error}") from error
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise IntegrityError(f"Could not read Git revision for {path}: {error}") from error
    return completed.stdout.strip()


def build_manifest(settings: NeuroGolfSettings) -> dict[str, Any]:
    root = settings.paths.repository_root
    missing = [relative for relative in PROTECTED_FILES if not (root / relative).is_file()]
    if missing:
        raise IntegrityError(f"Cannot create integrity manifest; missing protected files: {missing}")
    return {
        "version": 1,
        "algorithm": "sha256",
        "judge_version": JUDGE_VERSION,
        "files": {relative: sha256_file(root / relative) for relative in PROTECTED_FILES},
        "git_revisions": {"ARC-GEN": git_revision(settings.paths.arcgen_root)},
    }


def write_manifest(settings: NeuroGolfSettings) -> dict[str, Any]:
    manifest = build_manifest(settings)
    path = settings.paths.integrity_manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_manifest(settings: NeuroGolfSettings) -> dict[str, Any]:
    path = settings.paths.integrity_manifest
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f"Could not read integrity manifest {path}: {error}") from error
    if not isinstance(value, dict) or value.get("algorithm") != "sha256":
        raise IntegrityError(f"Unsupported integrity manifest: {path}")
    if value.get("version") != 1 or value.get("judge_version") != JUDGE_VERSION:
        raise IntegrityError(f"Stale integrity/judge version in manifest: {path}")
    files = value.get("files")
    if not isinstance(files, dict) or set(files) != set(PROTECTED_FILES):
        raise IntegrityError(
            f"Integrity manifest must list exactly the protected files: {PROTECTED_FILES}"
        )
    revision = value.get("git_revisions", {}).get("ARC-GEN")
    if not isinstance(revision, str) or not revision:
        raise IntegrityError("Integrity manifest lacks the ARC-GEN revision")
    return value


def verify_integrity(settings: NeuroGolfSettings) -> dict[str, Any]:
    manifest = load_manifest(settings)
    root = settings.paths.repository_root
    failures: list[dict[str, str]] = []
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        actual = sha256_file(path) if path.is_file() else "missing"
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    expected_revision = manifest.get("git_revisions", {}).get("ARC-GEN")
    actual_revision = git_revision(settings.paths.arcgen_root)
    if actual_revision != expected_revision:
        failures.append(
            {
                "path": "ARC-GEN",
                "expected": str(expected_revision),
                "actual": actual_revision,
            }
        )
    return {
        "valid": not failures,
        "judge_version": manifest.get("judge_version"),
        "checked_files": len(manifest.get("files", {})),
        "failures": failures,
    }
