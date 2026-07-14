from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurogolf.config import load_settings
from neurogolf.config.models import CodexConfig
from neurogolf.orchestrator.codex import (
    CodexEpochResult,
    CodexInvocation,
    build_codex_command,
)
from neurogolf.orchestrator.promotion import auto_promote_gate_result
from neurogolf.orchestrator.schedule import run_bounded
from neurogolf.orchestrator.solve import solve_task
from neurogolf.orchestrator.status import read_status, write_status


def _invocation(
    tmp_path: Path, *, mode: str = "fresh", profile: str | None = "neurogolf-xhigh"
) -> CodexInvocation:
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Work only on the prepared task.", encoding="utf-8")
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    return CodexInvocation(
        task_num=7,
        workspace=tmp_path,
        prompt_path=prompt,
        event_log=tmp_path / "events.jsonl",
        final_message=tmp_path / "final.json",
        output_schema=schema,
        mode=mode,  # type: ignore[arg-type]
        session_id="session-123" if mode == "resume" else None,
        profile=profile,
    )


def test_codex_command_is_isolated_structured_and_resumable(tmp_path: Path) -> None:
    config = CodexConfig(
        executable="codex",
        model="gpt-5.6-sol",
        reasoning_effort="xhigh",
        max_parallel_tasks=2,
        timeout_seconds=60,
        max_epochs=4,
    )
    fresh = build_codex_command(_invocation(tmp_path), config)
    assert fresh[:4] == ["codex", "exec", "--profile", "neurogolf-xhigh"]
    assert "--ask-for-approval" not in fresh
    assert "--strict-config" in fresh
    assert "--cd" in fresh and str(tmp_path) in fresh
    assert "--json" in fresh
    assert "--output-schema" in fresh and "--output-last-message" in fresh
    assert fresh[-1] == "Work only on the prepared task."

    resumed = build_codex_command(_invocation(tmp_path, mode="resume"), config)
    assert resumed.index("resume") > resumed.index("exec")
    assert resumed[-2] == "session-123"
    with pytest.raises(ValueError, match="session ID"):
        invalid = _invocation(tmp_path, mode="resume")
        object.__setattr__(invalid, "session_id", None)
        build_codex_command(invalid, config)


def test_codex_command_without_profile_uses_exec_config_overrides(tmp_path: Path) -> None:
    config = CodexConfig(
        model="gpt-5.6-sol",
        reasoning_effort="max",
        timeout_seconds=60,
    )
    command = build_codex_command(_invocation(tmp_path, profile=None), config)
    assert command[:2] == ["codex", "exec"]
    assert "--profile" not in command
    assert command[6:8] == ["--config", 'approval_policy="never"']
    assert 'model_reasoning_effort="max"' in command
    assert "--ask-for-approval" not in command


def test_status_round_trip_and_invalid_input(tmp_path: Path) -> None:
    path = tmp_path / "state" / "status.json"
    write_status(path, {"status": "completed", "session_id": "abc"})
    assert read_status(path) == {"session_id": "abc", "status": "completed"}
    path.write_text("not json", encoding="utf-8")
    assert read_status(path) is None


def test_all_worker_instructions_warn_that_helpers_are_opset_specific(
    repository_root: Path,
) -> None:
    paths = [
        repository_root / "AGENTS.md",
        repository_root / "prompts/solve_task.md",
        repository_root / "prompts/repair_task.md",
        repository_root / "prompts/golf_task.md",
        repository_root / "prompts/review_task.md",
        repository_root / ".agents/skills/solve-neurogolf-task/SKILL.md",
    ]
    for path in paths:
        assert "not universally opset-aware" in path.read_text(encoding="utf-8"), path
        assert "onnx.helper" in path.read_text(encoding="utf-8"), path


def test_auto_promotion_calls_hash_bound_transaction(
    repository_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings(
        repository_root=repository_root,
        overrides={"paths": {"champion_root": tmp_path / "champions"}},
    )
    candidate = tmp_path / "candidate.onnx"
    candidate.write_bytes(b"fixture")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "judge_version": "fixture",
                "status": "passed",
                "eligible_for_promotion": True,
                "task_num": 1,
                "task_id": "007bbfb7",
                "candidate_path": str(candidate),
                "candidate_sha256": "a" * 64,
                "candidate_size_bytes": 7,
                "integrity": {},
                "legality": {},
                "suites": {},
                "score": {"objective": 90, "points": 20.5},
                "champion_comparison": {},
                "failure_packet_path": None,
                "fuzz_nonce": 1,
                "errors": [],
                "started_at": "2026-01-01T00:00:00+00:00",
                "completed_at": "2026-01-01T00:00:01+00:00",
                "duration_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )

    class EmptyRegistry:
        def __init__(self, _root: Path) -> None:
            pass

        def current(self, _task_num: int):
            return None

    captured: dict[str, object] = {}

    def fake_promote(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            candidate_sha256="a" * 64,
            objective=90,
            points=20.5,
            onnx_path="champions/model.onnx",
        )

    monkeypatch.setattr("neurogolf.orchestrator.promotion.ChampionRegistry", EmptyRegistry)
    monkeypatch.setattr("neurogolf.orchestrator.promotion.promote_candidate", fake_promote)
    result = auto_promote_gate_result(
        task_num=1,
        candidate=candidate,
        gate_result_path=gate_path,
        settings=settings,
        workspace=tmp_path,
    )
    assert result == {
        "status": "promoted",
        "candidate_sha256": "a" * 64,
        "objective": 90,
        "points": 20.5,
        "onnx_path": "champions/model.onnx",
    }
    assert captured["candidate"] == candidate


@pytest.mark.anyio
async def test_bounded_schedule_limits_parallelism_and_isolates_failures() -> None:
    active = 0
    maximum = 0

    async def worker(item: int) -> int:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        if item == 3:
            raise RuntimeError("fixture failure")
        return item * 2

    results = await run_bounded([1, 2, 3, 4], worker, limit=2)
    assert maximum == 2
    assert results[1] == 2
    assert results[3] == {"status": "orchestrator_error", "error": "fixture failure"}


@pytest.mark.anyio
async def test_solve_task_uses_mock_epoch_without_launching_codex(
    repository_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings(
        repository_root=repository_root,
        overrides={
            "paths": {"task_workspace_root": tmp_path / "tasks"},
            "codex": {"max_epochs": 2},
        },
    )
    workspace = settings.paths.task_workspace_root / "task001"

    def fake_prepare(_task_num: int, _settings) -> None:
        (workspace / "state").mkdir(parents=True, exist_ok=True)

    async def fake_epoch(invocation, _config) -> CodexEpochResult:
        assert invocation.workspace == workspace
        return CodexEpochResult(
            task_num=1,
            status="completed",
            return_code=0,
            timed_out=False,
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            event_log=str(invocation.event_log),
            final_message=json.dumps(
                {
                    "schema_version": "1.0",
                    "task_num": 1,
                    "task_id": "007bbfb7",
                    "status": "passed",
                    "summary": "mock only",
                    "candidate_path": None,
                    "candidate_sha256": None,
                    "gate_result_path": None,
                    "commands": [],
                    "tests": [],
                    "next_action": "none",
                }
            ),
            session_id="fixture-session",
        )

    monkeypatch.setattr("neurogolf.orchestrator.solve.prepare_task", fake_prepare)
    monkeypatch.setattr("neurogolf.orchestrator.solve.run_codex_epoch", fake_epoch)
    result = await solve_task(1, settings, kind="repair", epoch=1)
    assert result["status"] == "completed"
    assert result["kind"] == "repair"
    persisted = read_status(workspace / "state/orchestrator.json")
    assert persisted is not None and persisted["session_id"] == "fixture-session"
    assert persisted["worker_result"]["status"] == "passed"
    assert persisted["auto_promotion"]["status"] == "not_requested"
    resumed = await solve_task(1, settings, kind="repair", epoch=2)
    assert resumed["invocation_mode"] == "resume"
    with pytest.raises(ValueError, match="maximum"):
        await solve_task(1, settings, epoch=3)
