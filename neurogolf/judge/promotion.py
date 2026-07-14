"""Explicit hash-bound champion promotion and rollback transactions."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from neurogolf.config.models import NeuroGolfSettings
from neurogolf.errors import PromotionError
from neurogolf.judge.registry import ChampionRecord, ChampionRegistry
from neurogolf.judge.reports import GateResult, utc_now, validate_report
from neurogolf.tasks.integrity import JUDGE_VERSION, git_revision, sha256_file, verify_integrity
from neurogolf.tasks.loader import load_kaggle_dataset
from neurogolf.tasks.mapping import TaskMap


def _load_gate(path: Path) -> GateResult:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        validate_report(payload, "gate_result.schema.json")
        return GateResult.model_validate(payload)
    except Exception as error:
        raise PromotionError(f"Invalid gate result {path}: {error}") from error


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise PromotionError(f"Refusing to overwrite immutable artifact {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=".promote-", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def promote_candidate(
    *,
    task_num: int,
    candidate: Path,
    gate_result_path: Path,
    settings: NeuroGolfSettings,
    builder_path: Path | None = None,
    explanation_path: Path | None = None,
    notes: str | None = None,
    allow_score_regression: bool = False,
) -> ChampionRecord:
    gate = _load_gate(gate_result_path)
    if gate.task_num != task_num or gate.status != "passed":
        raise PromotionError("Gate result does not authorize promotion for this task")
    if not gate.eligible_for_promotion and not allow_score_regression:
        raise PromotionError("Gate result does not authorize a score-regressing promotion")
    if not candidate.is_file():
        raise PromotionError(f"Candidate does not exist: {candidate}")
    candidate_hash = sha256_file(candidate)
    if gate.candidate_sha256 != candidate_hash:
        raise PromotionError("Candidate hash does not match the passing gate result")
    integrity = verify_integrity(settings)
    if not integrity["valid"]:
        raise PromotionError(f"Protected integrity verification failed: {integrity['failures']}")
    if gate.score is None:
        raise PromotionError("Passing gate result has no official score")
    mapping = TaskMap(settings)
    record = mapping.get(task_num)
    if gate.task_id != record.task_id or gate.judge_version != JUDGE_VERSION:
        raise PromotionError("Gate result task/judge identity does not match current sources")
    dataset = load_kaggle_dataset(record.kaggle_json_path)
    expected_counts = {
        "public": len(dataset.train) + len(dataset.test),
        "provided_arc_gen": len(dataset.arc_gen),
        "regression": settings.generation.regression_seed_count,
        "fuzz": settings.generation.gate_fuzz_count,
        "boundary": settings.generation.boundary_seed_count,
        "sealed": settings.generation.sealed_seed_count,
    }
    for name, expected_count in expected_counts.items():
        suite = gate.suites.get(name)
        if suite is None:
            raise PromotionError(f"Gate result is missing required suite {name}")
        if suite.failed or suite.passed != suite.total:
            raise PromotionError(f"Gate suite {name} is not fully passing")
        if suite.total + suite.skipped != expected_count:
            raise PromotionError(
                f"Gate suite {name} has {suite.total + suite.skipped} cases, "
                f"expected {expected_count}"
            )
    score_keys = {"memory_bytes", "parameter_elements", "objective", "points"}
    if not score_keys.issubset(gate.score):
        raise PromotionError(
            f"Gate score is missing fields: {sorted(score_keys - set(gate.score))}"
        )
    registry = ChampionRegistry(settings.paths.champion_root)
    current = registry.current(task_num)
    objective = int(gate.score["objective"])
    if current is not None and objective > current.objective and not allow_score_regression:
        raise PromotionError(
            f"Candidate objective {objective} is worse than champion objective {current.objective}"
        )
    history_path = (
        settings.paths.champion_root / "history" / f"task{task_num:03d}" / f"{candidate_hash}.onnx"
    )
    model_path = (
        settings.paths.champion_root / "models" / f"task{task_num:03d}-{candidate_hash}.onnx"
    )
    public = gate.suites.get("public")
    regression = gate.suites.get("regression")
    fuzz = gate.suites.get("fuzz")
    sealed = gate.suites.get("sealed")
    assert public is not None and regression is not None and fuzz is not None and sealed is not None
    _atomic_copy(candidate, history_path)
    _atomic_copy(candidate, model_path)
    champion = ChampionRecord(
        task_num=task_num,
        task_id=record.task_id,
        candidate_sha256=candidate_hash,
        onnx_path=str(model_path),
        builder_path=str(builder_path) if builder_path else None,
        explanation_path=str(explanation_path) if explanation_path else None,
        judge_version=gate.judge_version,
        generator_revision=git_revision(settings.paths.arcgen_root),
        official_utils_sha256=str(gate.integrity.get("official_utils_sha256", "")),
        public_passed=public.passed,
        public_total=public.total,
        regression_passed=regression.passed,
        regression_total=regression.total,
        fuzz_passed=fuzz.passed,
        fuzz_total=fuzz.total,
        sealed_passed=sealed.passed,
        sealed_total=sealed.total,
        memory_bytes=int(gate.score["memory_bytes"]),
        parameter_elements=int(gate.score["parameter_elements"]),
        objective=objective,
        points=float(gate.score["points"]),
        created_at=utc_now(),
        parent_candidate=current.candidate_sha256 if current else None,
        status="champion",
        notes=notes,
    )
    columns = list(champion.to_dict())
    placeholders = ",".join("?" for _ in columns)
    with registry.transaction() as connection:
        if current:
            connection.execute(
                "UPDATE champions SET status='historical' WHERE task_num=? AND status='champion'",
                (task_num,),
            )
        connection.execute(
            f"INSERT INTO champions ({','.join(columns)}) VALUES ({placeholders}) "
            "ON CONFLICT(task_num,candidate_sha256) DO UPDATE SET "
            "status='champion', notes=excluded.notes",
            tuple(champion.to_dict()[column] for column in columns),
        )
        connection.execute(
            "INSERT INTO promotion_history "
            "(task_num,candidate_sha256,previous_sha256,action,created_at,notes) "
            "VALUES (?,?,?,?,?,?)",
            (
                task_num,
                candidate_hash,
                current.candidate_sha256 if current else None,
                "promote",
                champion.created_at,
                notes,
            ),
        )
    return champion


def rollback_champion(
    *, task_num: int, sha256: str, settings: NeuroGolfSettings, notes: str | None = None
) -> ChampionRecord:
    integrity = verify_integrity(settings)
    if not integrity["valid"]:
        raise PromotionError(f"Protected integrity verification failed: {integrity['failures']}")
    registry = ChampionRegistry(settings.paths.champion_root)
    target = registry.get(task_num, sha256)
    current = registry.current(task_num)
    if target is None:
        raise PromotionError(f"No registered candidate {sha256} for task {task_num}")
    if not Path(target.onnx_path).is_file():
        raise PromotionError(f"Historical artifact is missing: {target.onnx_path}")
    if sha256_file(Path(target.onnx_path)) != target.candidate_sha256:
        raise PromotionError(f"Historical artifact hash mismatch: {target.onnx_path}")
    if current and current.candidate_sha256 == sha256:
        return current
    created = utc_now()
    with registry.transaction() as connection:
        connection.execute(
            "UPDATE champions SET status='historical' WHERE task_num=? AND status='champion'",
            (task_num,),
        )
        connection.execute(
            "UPDATE champions SET status='champion' WHERE task_num=? AND candidate_sha256=?",
            (task_num, sha256),
        )
        connection.execute(
            "INSERT INTO promotion_history "
            "(task_num,candidate_sha256,previous_sha256,action,created_at,notes) "
            "VALUES (?,?,?,?,?,?)",
            (
                task_num,
                sha256,
                current.candidate_sha256 if current else None,
                "rollback",
                created,
                notes,
            ),
        )
    rolled = registry.current(task_num)
    if rolled is None:
        raise PromotionError("Rollback transaction committed without a current champion")
    return rolled
