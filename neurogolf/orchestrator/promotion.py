"""Auto-promotion policy outside the integrity-protected acceptance gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neurogolf.config.models import NeuroGolfSettings
from neurogolf.errors import PromotionError
from neurogolf.judge.promotion import promote_candidate
from neurogolf.judge.registry import ChampionRegistry
from neurogolf.judge.reports import GateResult, validate_report


def _resolved(path: Path, workspace: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (workspace / path).resolve()


def auto_promote_gate_result(
    *,
    task_num: int,
    candidate: Path,
    gate_result_path: Path,
    settings: NeuroGolfSettings,
    workspace: Path,
) -> dict[str, Any]:
    """Promote an eligible exact hash, or return an explicit no-op status."""

    candidate = _resolved(candidate, workspace)
    gate_result_path = _resolved(gate_result_path, workspace)
    try:
        payload: Any = json.loads(gate_result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("gate result is not an object")
        validate_report(payload, "gate_result.schema.json")
        gate = GateResult.model_validate(payload)
    except Exception as error:
        raise PromotionError(f"Cannot auto-promote from {gate_result_path}: {error}") from error
    if gate.task_num != task_num:
        raise PromotionError(
            f"Gate result task {gate.task_num} does not match requested task {task_num}"
        )
    if gate.status != "passed" or not gate.eligible_for_promotion:
        return {
            "status": "not_eligible",
            "candidate_sha256": gate.candidate_sha256,
            "gate_status": gate.status,
        }
    current = ChampionRegistry(settings.paths.champion_root).current(task_num)
    if current is not None and current.candidate_sha256 == gate.candidate_sha256:
        return {
            "status": "already_current",
            "candidate_sha256": current.candidate_sha256,
            "objective": current.objective,
        }
    builder = workspace / "solution/build.py"
    explanation = workspace / "solution/explanation.md"
    champion = promote_candidate(
        task_num=task_num,
        candidate=candidate,
        gate_result_path=gate_result_path,
        settings=settings,
        builder_path=builder if builder.is_file() else None,
        explanation_path=explanation if explanation.is_file() else None,
        notes="Automatically promoted after an eligible fully passing gate result.",
    )
    return {
        "status": "promoted",
        "candidate_sha256": champion.candidate_sha256,
        "objective": champion.objective,
        "points": champion.points,
        "onnx_path": champion.onnx_path,
    }


def auto_promote_worker_result(
    *,
    task_num: int,
    worker_result: dict[str, object],
    settings: NeuroGolfSettings,
    workspace: Path,
) -> dict[str, Any]:
    """Resolve paths from a validated Codex result and apply auto-promotion."""

    if worker_result.get("status") != "passed":
        return {"status": "not_requested", "reason": "worker did not report passed"}
    candidate = worker_result.get("candidate_path")
    gate_result = worker_result.get("gate_result_path")
    if not isinstance(candidate, str) or not isinstance(gate_result, str):
        return {"status": "not_requested", "reason": "worker did not report candidate/gate paths"}
    return auto_promote_gate_result(
        task_num=task_num,
        candidate=Path(candidate),
        gate_result_path=Path(gate_result),
        settings=settings,
        workspace=workspace,
    )
