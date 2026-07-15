#!/usr/bin/env python3
"""Run one Codex CLI process per NeuroGolf task."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from typing import Iterable


DEFAULT_PARALLELISM = 10
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_PROFILE = "neurogolf-high"
DEFAULT_REASONING = "high"
PROMPT_TEMPLATE_FILENAME = "codex_task_prompt.md"
TASK_NUMBER_PLACEHOLDER = "{{TASK_NUMBER}}"
UTILS_DIRNAME = "utils"
REQUIRED_UTILS_FILENAMES = (
    "neurogolf_utils.py",
    "task_map.json",
    "task_principles.json",
    "the-2026-neurogolf-championship.py",
)

RATE_LIMIT_PATTERNS = (
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"rate[ _-]?limit",
        r"usage[ _-]?limit",
        r"usageLimitExceeded",
        r"too many requests",
        r"quota.*exceed",
    )
)
RATE_LIMIT_PATTERNS = tuple(RATE_LIMIT_PATTERNS)


@dataclasses.dataclass(frozen=True)
class TaskResult:
    task: str
    status: str
    returncode: int | None
    attempts: int
    elapsed_seconds: float
    message: str = ""


class Runner:
    def __init__(
        self, args: argparse.Namespace, repo_root: Path, prompt_template: str
    ) -> None:
        self.args = args
        self.repo_root = repo_root
        self.prompt_template = prompt_template
        self.stop_event = threading.Event()
        self.interrupted = False
        self._print_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active: set[subprocess.Popen[str]] = set()

    def log(self, message: str) -> None:
        timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        with self._print_lock:
            print(f"[{timestamp}] {message}", flush=True)

    def request_stop(self, signum: int | None = None) -> None:
        if signum is not None:
            self.interrupted = True
            self.log(f"received signal {signum}; stopping active Codex processes")
        self.stop_event.set()

    def terminate_active(self) -> None:
        with self._active_lock:
            processes = list(self._active)
        for process in processes:
            self._terminate_process(process)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass

    def build_command(self, task: str, attempt: int) -> tuple[list[str], Path, Path]:
        result_path = self.repo_root / "results" / f"{task}.md"

        suffix = "" if attempt == 1 else f".attempt{attempt}"
        events_path = self.repo_root / "logs" / f"{task}{suffix}.events.jsonl"
        stderr_path = self.repo_root / "logs" / f"{task}{suffix}.stderr"

        task_number = task.removeprefix("task")
        prompt = self.prompt_template.replace(TASK_NUMBER_PLACEHOLDER, task_number)
        command = [
            self.args.codex_bin,
            "exec",
            "--profile",
            self.args.profile,
            "--model",
            self.args.model,
            "--config",
            f'model_reasoning_effort="{self.args.reasoning}"',
            "--cd",
            str(self.repo_root),
            "--json",
            "--output-last-message",
            str(result_path),
            prompt,
        ]
        return command, events_path, stderr_path

    def run_task(self, task: str) -> TaskResult:
        started = time.monotonic()
        if self.stop_event.is_set():
            return TaskResult(task, "cancelled", None, 0, 0.0, "runner stopped")

        lock_path = self.repo_root / "logs" / "locks" / f"{task}.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return TaskResult(
                    task,
                    "locked",
                    None,
                    0,
                    time.monotonic() - started,
                    "another runner owns this task",
                )

            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(f"pid={os.getpid()} started={dt.datetime.now().astimezone().isoformat()}\n")
            lock_file.flush()

            total_attempts = 1 + self.args.rate_limit_retries
            for attempt in range(1, total_attempts + 1):
                if self.stop_event.is_set():
                    return TaskResult(
                        task,
                        "cancelled",
                        None,
                        attempt - 1,
                        time.monotonic() - started,
                        "runner stopped",
                    )

                command, events_path, stderr_path = self.build_command(task, attempt)
                if self.args.dry_run:
                    self.log(f"{task}: {shlex.join(command)}")
                    return TaskResult(task, "dry-run", 0, 0, 0.0)

                self.log(f"{task}: starting attempt {attempt}/{total_attempts}")
                returncode, timed_out = self._run_process(command, events_path, stderr_path)

                if returncode == 0:
                    elapsed = time.monotonic() - started
                    submission_path = self.repo_root / "submission" / f"{task}.onnx"
                    if submission_path.is_file():
                        self.log(f"{task}: completed in {elapsed:.1f}s")
                        return TaskResult(task, "completed", 0, attempt, elapsed)
                    message = f"Codex exited successfully but did not create {submission_path.name}"
                    self.log(f"{task}: failed ({message})")
                    return TaskResult(task, "failed", 0, attempt, elapsed, message)

                if self.stop_event.is_set():
                    return TaskResult(
                        task,
                        "cancelled",
                        returncode,
                        attempt,
                        time.monotonic() - started,
                        "runner stopped",
                    )

                if timed_out:
                    message = f"timed out after {self.args.timeout:g}s"
                    self.log(f"{task}: {message}")
                    return TaskResult(
                        task,
                        "failed",
                        returncode,
                        attempt,
                        time.monotonic() - started,
                        message,
                    )

                rate_limited = _looks_rate_limited(events_path, stderr_path)
                if rate_limited and attempt < total_attempts:
                    self.log(
                        f"{task}: quota/rate limit detected; retrying in "
                        f"{self.args.retry_delay:g}s"
                    )
                    if self.stop_event.wait(self.args.retry_delay):
                        continue
                    continue

                message = "quota/rate limit" if rate_limited else "Codex exited non-zero"
                self.log(f"{task}: failed with exit code {returncode} ({message})")
                return TaskResult(
                    task,
                    "failed",
                    returncode,
                    attempt,
                    time.monotonic() - started,
                    message,
                )

        raise AssertionError("unreachable")

    def _run_process(
        self, command: list[str], events_path: Path, stderr_path: Path
    ) -> tuple[int, bool]:
        with (
            events_path.open("w", encoding="utf-8") as stdout_file,
            stderr_path.open("w", encoding="utf-8") as stderr_file,
        ):
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                start_new_session=(os.name == "posix"),
            )
            with self._active_lock:
                self._active.add(process)

            deadline = (
                time.monotonic() + self.args.timeout if self.args.timeout > 0 else None
            )
            timed_out = False
            try:
                while process.poll() is None:
                    if self.stop_event.wait(1.0):
                        self._terminate_process(process)
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        timed_out = True
                        self._terminate_process(process)
                        break
                returncode = process.wait()
            finally:
                with self._active_lock:
                    self._active.discard(process)
            return returncode, timed_out


def _tail_text(path: Path, max_bytes: int = 256 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _looks_rate_limited(*paths: Path) -> bool:
    text = "\n".join(_tail_text(path) for path in paths)
    return any(pattern.search(text) for pattern in RATE_LIMIT_PATTERNS)


def _parse_task_tokens(values: Iterable[str]) -> list[str]:
    selected: set[int] = set()
    for value in values:
        for raw_token in value.split(","):
            token = raw_token.strip().lower().removeprefix("task")
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start = int(start_text.removeprefix("task"))
                end = int(end_text.removeprefix("task"))
                if start > end:
                    raise ValueError(f"invalid descending task range: {raw_token!r}")
                selected.update(range(start, end + 1))
            else:
                selected.add(int(token))

    invalid = sorted(number for number in selected if not 1 <= number <= 400)
    if invalid:
        raise ValueError(f"task numbers must be in 001..400: {invalid}")
    return [f"task{number:03d}" for number in sorted(selected)]


def _validate_repo(repo_root: Path, tasks: list[str]) -> None:
    utils_dir = repo_root / UTILS_DIRNAME
    missing_utils = [
        path.relative_to(repo_root).as_posix()
        for filename in REQUIRED_UTILS_FILENAMES
        if not (path := utils_dir / filename).is_file()
    ]
    if missing_utils:
        raise RuntimeError(
            "missing required utility file(s): " + ", ".join(missing_utils)
        )

    map_path = utils_dir / "task_map.json"
    with map_path.open(encoding="utf-8") as handle:
        task_map = json.load(handle)

    missing: list[str] = []
    for task in tasks:
        task_id = task_map.get(task)
        expected = (
            repo_root / "ARC-GEN" / "tasks" / f"task_{task_id}.py"
            if task_id
            else None
        )
        if not (repo_root / "kaggle_tasks_data" / f"{task}.json").is_file():
            missing.append(f"kaggle_tasks_data/{task}.json")
        if expected is None or not expected.is_file():
            missing.append(f"ARC-GEN generator for {task}")
    if missing:
        preview = ", ".join(missing[:10])
        extra = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
        raise RuntimeError(f"repository preflight failed: {preview}{extra}")


def _load_prompt_template(repo_root: Path) -> str:
    prompt_path = repo_root / PROMPT_TEMPLATE_FILENAME
    try:
        prompt_template = prompt_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"cannot read prompt template {prompt_path}: {error}") from error
    if TASK_NUMBER_PLACEHOLDER not in prompt_template:
        raise RuntimeError(
            f"prompt template {prompt_path} must contain {TASK_NUMBER_PLACEHOLDER}"
        )
    return prompt_template


def _write_summary(path: Path, results: list[TaskResult], args: argparse.Namespace) -> None:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    payload = {
        "finished_at": dt.datetime.now().astimezone().isoformat(),
        "model": args.model,
        "reasoning": args.reasoning,
        "profile": args.profile,
        "parallelism": args.parallel,
        "counts": counts,
        "tasks": [dataclasses.asdict(result) for result in sorted(results, key=lambda r: r.task)],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Codex CLI process per NeuroGolf task."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["1-400"],
        metavar="SPEC",
        help="task numbers/ranges (default: 1-400; example: 1-20,42 task100)",
    )
    parser.add_argument(
        "-n",
        "--parallel",
        type=int,
        default=DEFAULT_PARALLELISM,
        help=f"maximum concurrent Codex processes (default: {DEFAULT_PARALLELISM})",
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning", default=DEFAULT_REASONING)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument(
        "--force",
        action="store_true",
        help="run tasks even when submission/taskNNN.onnx already exists",
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=2,
        help="bounded retries after a detected quota/rate-limit failure (default: 2)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=90.0,
        help="seconds before retrying a quota-limited task (default: 90)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="per-attempt timeout in seconds; 0 disables it (default: 0)",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    if args.rate_limit_retries < 0:
        parser.error("--rate-limit-retries cannot be negative")
    if args.retry_delay < 0:
        parser.error("--retry-delay cannot be negative")
    if args.timeout < 0:
        parser.error("--timeout cannot be negative")

    try:
        tasks = _parse_task_tokens(args.tasks)
    except ValueError as error:
        parser.error(str(error))
    if not tasks:
        parser.error("no tasks selected")

    repo_root = Path(__file__).resolve().parent
    try:
        _validate_repo(repo_root, tasks)
        prompt_template = _load_prompt_template(repo_root)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.error(str(error))

    if shutil.which(args.codex_bin) is None:
        parser.error(f"Codex executable not found: {args.codex_bin!r}")

    for directory in (
        repo_root / "tasks",
        repo_root / "submission",
        repo_root / "results",
        repo_root / "logs",
        repo_root / "logs" / "locks",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        (repo_root / "tasks" / task).mkdir(parents=True, exist_ok=True)

    skipped = []
    if not args.force:
        runnable = []
        for task in tasks:
            if (repo_root / "submission" / f"{task}.onnx").is_file():
                skipped.append(TaskResult(task, "skipped", 0, 0, 0.0, "submission exists"))
            else:
                runnable.append(task)
        tasks = runnable

    runner = Runner(args, repo_root, prompt_template)
    for caught_signal in (signal.SIGINT, signal.SIGTERM):
        signal.signal(
            caught_signal,
            lambda signum, _frame, active_runner=runner: active_runner.request_stop(signum),
        )

    runner.log(
        f"selected {len(tasks)} runnable task(s), skipped {len(skipped)}, "
        f"parallelism={args.parallel}, model={args.model}, reasoning={args.reasoning}"
    )

    results = list(skipped)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(runner.run_task, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as error:  # keep other independent tasks observable
                    task = futures[future]
                    runner.log(f"{task}: internal runner error: {error}")
                    result = TaskResult(task, "failed", None, 0, 0.0, repr(error))
                results.append(result)
                if args.fail_fast and result.status in {"failed", "locked"}:
                    runner.request_stop()
                    runner.terminate_active()
    finally:
        runner.request_stop()
        runner.terminate_active()

    summary_path = repo_root / "logs" / "codex-run-summary.json"
    _write_summary(summary_path, results, args)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    runner.log(f"summary: {counts}; details: {summary_path}")

    if runner.interrupted:
        return 130
    return 1 if any(result.status in {"failed", "locked", "cancelled"} for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
