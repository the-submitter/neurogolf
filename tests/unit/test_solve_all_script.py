from __future__ import annotations

from pathlib import Path

import pytest

from neurogolf.config import load_settings
from scripts import solve_all_tasks


@pytest.mark.anyio
async def test_solve_all_schedules_every_mapped_task(
    repository_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings(
        repository_root=repository_root,
        overrides={"codex": {"max_parallel_tasks": 3}},
    )
    captured: dict[str, object] = {}

    async def fake_solve_many(task_numbers, passed_settings, **kwargs):
        captured["task_numbers"] = task_numbers
        captured["settings"] = passed_settings
        captured["kwargs"] = kwargs
        return {task_num: {"status": "completed"} for task_num in task_numbers}

    monkeypatch.setattr(solve_all_tasks, "solve_many", fake_solve_many)
    results = await solve_all_tasks.solve_all(settings, kind="solve", epoch=1, resume=True)
    task_numbers = captured["task_numbers"]
    assert isinstance(task_numbers, list)
    assert task_numbers == list(range(1, 401))
    assert len(results) == 400
    assert captured["settings"] is settings
    assert captured["kwargs"] == {"kind": "solve", "epoch": 1, "resume": True}


def test_solve_all_summary_tracks_worker_outcomes() -> None:
    results = {
        1: {"status": "completed", "worker_result": {"status": "passed"}},
        2: {"status": "failed", "worker_result": None},
        3: {"status": "completed", "worker_result_error": "invalid JSON"},
    }
    assert solve_all_tasks._status_counts(results) == {
        "orchestrator:completed": 2,
        "orchestrator:failed": 1,
        "worker:invalid": 1,
        "worker:passed": 1,
    }
    assert solve_all_tasks._successful(results[1], "solve")
    assert not solve_all_tasks._successful(results[2], "solve")
    assert not solve_all_tasks._successful(results[3], "solve")
