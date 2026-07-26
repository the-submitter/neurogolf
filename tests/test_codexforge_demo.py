from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.build_replay_fixture import build_fixture
from demo.codexforge_dashboard import (
    apply_live_event,
    DashboardState,
    FixtureError,
    JsonlObserver,
    Renderer,
    WorkerState,
    build_live_commands,
    detect_artifacts,
    expand_task_specs,
    extract_readme_metrics,
    initial_worker_state,
    load_fixture,
    parse_args,
    parse_jsonl_event,
    parse_runner_line,
    redact_text,
    refresh_worker_elapsed,
    run_replay,
    run_summary_stamp,
    throttle_replay_timeline,
    validate_fixture,
)


def namespace(**overrides: object) -> argparse.Namespace:
    values = {
        "tasks": ["11-12"],
        "parallel": 2,
        "force": False,
        "with_supervisor": False,
        "allow_reset_credit": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_committed_fixture_schema_and_regeneration() -> None:
    committed = load_fixture(ROOT / "demo" / "fixtures" / "codexforge_demo.json")
    regenerated = build_fixture(ROOT)
    assert validate_fixture(committed) == committed
    assert regenerated == committed
    assert [task["task"] for task in committed["tasks"]] == ["task001", "task002", "task010"]


def test_extracts_factual_metrics_from_existing_readmes() -> None:
    task001 = extract_readme_metrics((ROOT / "tasks" / "task001" / "README.md").read_text())
    task002 = extract_readme_metrics((ROOT / "tasks" / "task002" / "README.md").read_text())
    task006 = extract_readme_metrics((ROOT / "tasks" / "task006" / "README.md").read_text())
    task010 = extract_readme_metrics((ROOT / "tasks" / "task010" / "README.md").read_text())
    assert task001["objective_cost"] == 129
    assert task001["validation_examples"] == {"passed": 268, "total": 268}
    assert task002["estimated_score"] == pytest.approx(15.141405)
    assert task002["objective_cost"] == 19122
    assert task006["validation_examples"] == {"passed": 266, "total": 266}
    assert task006["intermediate_memory_bytes"] == 0
    assert task006["parameters"] == 100
    assert task006["objective_cost"] == 100
    assert task006["estimated_score"] == pytest.approx(20.394830)
    assert task006["onnx_node_count"] == 1
    assert task010["onnx_node_count"] == 4
    assert task010["validation_examples"] == {"passed": 265, "total": 265}


def test_jsonl_event_parsing_is_readable_and_redacted() -> None:
    line = json.dumps(
        {
            "type": "item.started",
            "item": {
                "type": "command_execution",
                "command": f"cd {ROOT} && OPENAI_API_KEY=secret-value python tasks/task011/build.py",
            },
        }
    )
    event = parse_jsonl_event(line, "task011")
    assert event is not None
    assert event["task"] == "task011"
    assert event["phase"] == "building ONNX"
    assert "secret-value" not in event["action"]
    assert str(ROOT) not in event["action"]


@pytest.mark.parametrize(
    "secret",
    [
        "Authorization: Bearer abcdefghijklmnop",
        "api_key=super-secret",
        "OPENAI_API_KEY=hidden-value",
        "github_pat-abcdefghijklmnopqrstuvwxyz",
        "person@example.com",
    ],
)
def test_secret_redaction(secret: str) -> None:
    rendered = redact_text(f"prefix {secret} suffix")
    assert secret not in rendered
    assert "prefix" in rendered and "suffix" in rendered


def test_missing_artifact_files_are_resilient(tmp_path: Path) -> None:
    artifacts = detect_artifacts(tmp_path, "task123")
    assert artifacts == {
        "readme": False,
        "task_onnx": [],
        "submission": False,
        "result": False,
        "events": False,
        "event_files": [],
    }


def test_live_observer_ignores_old_lines_then_reads_truncated_run(tmp_path: Path) -> None:
    path = tmp_path / "task001.events.jsonl"
    path.write_text("historical completed event that must not leak into a rerun\n")
    observer = JsonlObserver()
    observer.prime(path)

    assert observer.read(path) == []

    path.write_text("new event\n")
    assert observer.read(path) == ["new event\n"]


def test_run_summary_stamp_changes_when_summary_is_replaced(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    summary = logs / "codex-run-summary.json"
    assert run_summary_stamp(tmp_path) is None

    summary.write_text("{}\n")
    first = run_summary_stamp(tmp_path)
    replacement = logs / "replacement.json"
    replacement.write_text('{"tasks": []}\n')
    replacement.replace(summary)

    assert first is not None
    assert run_summary_stamp(tmp_path) != first


def test_no_color_rendering_has_no_color_sequences() -> None:
    stream = io.StringIO()
    state = DashboardState(mode="replay", workers={"task001": WorkerState("task001")}, parallel=1)
    renderer = Renderer(color=False, stream=stream)
    renderer.render(state)
    output = stream.getvalue()
    assert "\x1b[" not in output
    assert "DEMO REPLAY" in output


def test_runner_startup_line_populates_live_model_metadata() -> None:
    state = DashboardState(mode="live", workers={}, parallel=2)
    parse_runner_line(
        "selected 2 runnable task(s), parallelism=2, "
        "model=gpt-5.6-sol, reasoning=high",
        state,
    )
    assert state.model == "gpt-5.6-sol"
    assert state.reasoning == "high"


def test_live_worker_elapsed_advances_and_stops_on_completion() -> None:
    worker = WorkerState("task006")
    state = DashboardState(mode="live", workers={worker.task: worker}, parallel=1)

    parse_runner_line(
        "[2026-07-23T12:00:00+05:30] task006: starting attempt 1/3",
        state,
        now=100.0,
    )
    refresh_worker_elapsed(state, 165.0)

    assert worker.status == "running"
    assert worker.started_at == 100.0
    assert worker.elapsed == 65.0
    assert "01:05" in Renderer(color=False, stream=io.StringIO()).frame(state)

    apply_live_event(
        worker,
        {
            "task": "task006",
            "phase": "completed",
            "status": "completed",
            "action": "Codex worker finished",
        },
        now=170.0,
    )
    refresh_worker_elapsed(state, 200.0)

    assert worker.status == "completed"
    assert worker.elapsed == 70.0


def test_live_initial_state_skips_existing_submissions_unless_forced(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    submission.mkdir()
    for task in ("task012", "task013", "task014"):
        (submission / f"{task}.onnx").write_bytes(b"existing submission")

    workers = {
        task: initial_worker_state(
            tmp_path,
            task,
            "test principle",
            skip_existing_submission=True,
        )
        for task in ("task012", "task013", "task014", "task015")
    }
    forced = initial_worker_state(
        tmp_path,
        "task012",
        "supernova",
        skip_existing_submission=False,
    )

    skipped = workers["task012"]
    assert (skipped.phase, skipped.status, skipped.attempt) == ("skipped", "skipped", 0)
    assert "Existing submission" in skipped.action
    completed_statuses = [
        workers[task].status for task in ("task012", "task013", "task014")
    ]
    assert completed_statuses == ["skipped", "skipped", "skipped"]
    assert workers["task015"].status == "queued"
    assert (forced.phase, forced.status, forced.attempt) == ("queued", "queued", 1)


def test_replay_parallel_override_throttles_fixture_timeline(tmp_path: Path) -> None:
    fixture = build_fixture(ROOT)
    events, duration = throttle_replay_timeline(
        fixture["events"],
        fixture["tasks"],
        parallel=2,
        duration=float(fixture["duration_seconds"]),
    )
    active: set[str] = set()
    maximum_active = 0
    for event in events:
        task = event.get("task")
        if task is None:
            continue
        if event["status"] == "running":
            active.add(task)
        elif event["status"] in {"completed", "failed", "skipped"}:
            active.discard(task)
        maximum_active = max(maximum_active, len(active))

    assert maximum_active == 2
    assert duration > fixture["duration_seconds"]

    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture))

    class ParallelCapture:
        dynamic = False

        def __init__(self) -> None:
            self.values: list[int] = []

        def render(self, state: DashboardState) -> None:
            self.values.append(state.parallel)

    renderer = ParallelCapture()
    args = argparse.Namespace(
        fixture=str(fixture_path),
        tasks=None,
        parallel=2,
        no_color=True,
        speed=1.0,
        max_runtime=0.001,
    )
    assert run_replay(args, renderer) == 0
    assert renderer.values and set(renderer.values) == {2}


def test_replay_makes_no_network_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = build_fixture(ROOT)
    fixture["duration_seconds"] = 0.01
    fixture["tasks"] = fixture["tasks"][:1]
    fixture["events"] = [
        {
            "at": 0,
            "task": "task001",
            "phase": "completed",
            "status": "completed",
            "action": "Repository evidence replayed",
        }
    ]
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))

    def forbidden_socket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("replay attempted a network call")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    args = argparse.Namespace(
        fixture=str(path),
        tasks=None,
        no_color=True,
        speed=100.0,
        max_runtime=0.0,
    )
    assert run_replay(args, Renderer(color=False, stream=io.StringIO())) == 0


def test_replay_rejects_selected_tasks_missing_from_fixture(tmp_path: Path) -> None:
    fixture = build_fixture(ROOT)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))
    args = argparse.Namespace(
        fixture=str(path),
        tasks=["1,6"],
        no_color=True,
        speed=100.0,
        max_runtime=0.0,
    )
    with pytest.raises(FixtureError, match=r"task006.*--live"):
        run_replay(args, Renderer(color=False, stream=io.StringIO()))


def test_custom_fixture_can_snapshot_previous_live_tasks() -> None:
    fixture = build_fixture(ROOT, ("task001", "task006"))
    assert [task["task"] for task in fixture["tasks"]] == ["task001", "task006"]
    task006 = fixture["tasks"][1]
    assert task006["metrics"]["objective_cost"] == 100
    assert fixture["parallel_limit"] == 2
    assert fixture["duration_seconds"] == 68


def test_default_launcher_selects_replay_without_codex(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "demo" / "fixtures").mkdir(parents=True)
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "launch_demo.sh", tmp_path / "scripts" / "launch_demo.sh")
    (tmp_path / "demo" / "codexforge_dashboard.py").write_text("# placeholder\n")
    (tmp_path / "demo" / "fixtures" / "codexforge_demo.json").write_text("{}\n")
    (tmp_path / ".venv" / "bin" / "activate").write_text(
        'export PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P):${PATH}"\n'
    )
    capture = tmp_path / "python-args"
    fake_python = tmp_path / ".venv" / "bin" / "python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${CAPTURE}"\n')
    fake_python.chmod(0o755)
    codex_marker = tmp_path / "codex-called"
    fake_codex = tmp_path / ".venv" / "bin" / "codex"
    fake_codex.write_text('#!/usr/bin/env bash\ntouch "${CODEX_MARKER}"\n')
    fake_codex.chmod(0o755)
    result = subprocess.run(
        ["bash", "scripts/launch_demo.sh"],
        cwd=tmp_path,
        env={**os.environ, "CAPTURE": str(capture), "CODEX_MARKER": str(codex_marker)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--replay" in capture.read_text().splitlines()
    assert not codex_marker.exists()
    assert "quota-free" in result.stdout


def test_default_mode_and_reset_credit_are_safe() -> None:
    args = parse_args([])
    assert not args.live and not args.attach
    assert args.parallel is None
    assert parse_args(["--replay", "--parallel", "2"]).parallel == 2
    commands = build_live_commands(namespace())
    assert len(commands) == 1
    observed = build_live_commands(namespace(with_supervisor=True))
    assert observed[1][-1] == "--dry-run"
    enabled = build_live_commands(namespace(with_supervisor=True, allow_reset_credit=True))
    assert "--dry-run" not in enabled[1]
    with pytest.raises(SystemExit):
        parse_args(["--allow-reset-credit"])


def test_live_command_safely_forwards_tasks_and_parallelism() -> None:
    command = build_live_commands(namespace(tasks=["11-12", "task020"], parallel=2))[0]
    assert command[-5:] == ["11", "12", "20", "--parallel", "2"]
    assert command[1] == str(ROOT / "run_codex_tasks.py")
    assert expand_task_specs(["11-12", "task020"]) == ["task011", "task012", "task020"]
    assert expand_task_specs(["1,6"]) == ["task001", "task006"]
    default_command = build_live_commands(namespace(parallel=None))[0]
    assert default_command[-2:] == ["--parallel", "2"]
    with pytest.raises(ValueError):
        expand_task_specs(["11; touch /tmp/not-safe"])
