"""High-level fresh/resume/repair/golf epoch selection and persistence."""

from __future__ import annotations

import asyncio
import json
from typing import Literal

from neurogolf.config.models import NeuroGolfSettings
from neurogolf.judge.reports import validate_report
from neurogolf.orchestrator.codex import CodexInvocation, run_codex_epoch
from neurogolf.orchestrator.prepare import prepare_task
from neurogolf.orchestrator.promotion import auto_promote_worker_result
from neurogolf.orchestrator.schedule import run_bounded
from neurogolf.orchestrator.status import read_status, write_status

PromptKind = Literal["solve", "repair", "golf", "review"]


def _invocation(
    task_num: int,
    settings: NeuroGolfSettings,
    kind: PromptKind,
    epoch: int,
    *,
    resume: bool,
) -> CodexInvocation:
    workspace = settings.paths.task_workspace_root / f"task{task_num:03d}"
    state = workspace / "state"
    previous = read_status(state / "orchestrator.json")
    session_id = previous.get("session_id") if resume and previous and epoch > 1 else None
    prompt_name = {
        "solve": "solve_task.md",
        "repair": "repair_task.md",
        "golf": "golf_task.md",
        "review": "review_task.md",
    }[kind]
    return CodexInvocation(
        task_num=task_num,
        workspace=workspace,
        prompt_path=settings.paths.repository_root / "prompts" / prompt_name,
        event_log=state / f"codex_epoch_{epoch:02d}.jsonl",
        final_message=state / f"codex_epoch_{epoch:02d}_final.json",
        output_schema=settings.paths.repository_root / "neurogolf/schemas/codex_result.schema.json",
        mode="resume" if session_id else "fresh",
        session_id=session_id,
    )


async def solve_task(
    task_num: int,
    settings: NeuroGolfSettings,
    *,
    kind: PromptKind = "solve",
    epoch: int = 1,
    resume: bool = True,
) -> dict[str, object]:
    if epoch > settings.codex.max_epochs:
        raise ValueError(f"Epoch {epoch} exceeds configured maximum {settings.codex.max_epochs}")
    prepare_task(task_num, settings)
    invocation = _invocation(task_num, settings, kind, epoch, resume=resume)
    result = await run_codex_epoch(invocation, settings.codex)
    payload = result.to_dict()
    worker_result: dict[str, object] | None = None
    worker_result_error: str | None = None
    if result.final_message:
        try:
            parsed = json.loads(result.final_message)
            if not isinstance(parsed, dict):
                raise ValueError("Codex final output is not a JSON object")
            validate_report(parsed, "codex_result.schema.json")
            worker_result = parsed
        except Exception as error:
            worker_result_error = f"{type(error).__name__}: {error}"
    elif result.status == "completed":
        worker_result_error = "Codex final output is missing"
    payload.update(
        {
            "epoch": epoch,
            "kind": kind,
            "invocation_mode": invocation.mode,
            "worker_result": worker_result,
            "worker_result_error": worker_result_error,
        }
    )
    if settings.gate.auto_promote_eligible and worker_result is not None:
        try:
            payload["auto_promotion"] = auto_promote_worker_result(
                task_num=task_num,
                worker_result=worker_result,
                settings=settings,
                workspace=invocation.workspace,
            )
        except Exception as error:
            payload["codex_status"] = payload["status"]
            payload["status"] = "orchestrator_error"
            payload["auto_promotion"] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
    else:
        payload["auto_promotion"] = {
            "status": "disabled" if not settings.gate.auto_promote_eligible else "not_requested"
        }
    write_status(invocation.workspace / "state" / "orchestrator.json", payload)
    return payload


async def solve_many(
    task_numbers: list[int],
    settings: NeuroGolfSettings,
    *,
    kind: PromptKind = "solve",
    epoch: int = 1,
    resume: bool = True,
) -> dict[int, object]:
    return await run_bounded(
        task_numbers,
        lambda task_num: solve_task(task_num, settings, kind=kind, epoch=epoch, resume=resume),
        limit=settings.codex.max_parallel_tasks,
    )


async def solve_from_status(
    task_numbers: list[int],
    settings: NeuroGolfSettings,
    *,
    kind: PromptKind,
    resume: bool,
) -> dict[int, object]:
    async def worker(task_num: int) -> dict[str, object]:
        status_path = (
            settings.paths.task_workspace_root / f"task{task_num:03d}" / "state/orchestrator.json"
        )
        previous = read_status(status_path) or {}
        epoch = int(previous.get("epoch", 0)) + 1
        return await solve_task(
            task_num,
            settings,
            kind=kind,
            epoch=epoch,
            resume=resume,
        )

    return await run_bounded(
        task_numbers,
        worker,
        limit=settings.codex.max_parallel_tasks,
    )


def run_solve_task(
    task_num: int,
    settings: NeuroGolfSettings,
    *,
    kind: PromptKind = "solve",
    epoch: int = 1,
    resume: bool = True,
) -> dict[str, object]:
    return asyncio.run(solve_task(task_num, settings, kind=kind, epoch=epoch, resume=resume))
