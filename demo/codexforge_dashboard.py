#!/usr/bin/env python3
"""Terminal dashboard for factual CodexForge replay and live observation."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import queue
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from typing import Any, Iterable, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "demo" / "fixtures" / "codexforge_demo.json"
DEFAULT_LIVE_MODEL = "gpt-5.6-sol"
DEFAULT_LIVE_REASONING = "high"
VALID_PHASES = {
    "queued",
    "inspecting examples",
    "reading generator",
    "inferring rule",
    "building ONNX",
    "running ONNX Runtime",
    "validating examples",
    "exhaustive validation",
    "profiling official score",
    "optimizing graph",
    "writing documentation",
    "completed",
    "failed",
    "skipped",
}
VALID_STATUSES = {"queued", "running", "completed", "failed", "skipped"}
SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+"),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)(\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"),
    re.compile(r"\b(?:sk|sess|ghp)[-_][A-Za-z0-9_-]{8,}\b|\bgithub_pat[-_][A-Za-z0-9_]{8,}\b"),
    re.compile(r"\b[A-Z_][A-Z0-9_]{2,}=(?:\"[^\"]*\"|'[^']*'|[^\s]+)"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b|\beyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_-]+){1,2}\b"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
)
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


class FixtureError(ValueError):
    """Raised when a replay fixture is malformed."""


@dataclasses.dataclass
class WorkerState:
    task: str
    principle: str = "N/A"
    phase: str = "queued"
    attempt: int = 1
    started_at: float | None = None
    elapsed: float = 0.0
    action: str = "Waiting for a worker slot"
    status: str = "queued"
    artifact: str = "pending"
    artifact_paths: list[str] = dataclasses.field(default_factory=list)
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DashboardState:
    mode: str
    workers: dict[str, WorkerState]
    parallel: int
    model: str = "N/A"
    reasoning: str = "N/A"
    quota_remaining: str = "N/A"
    reset_credits: str = "N/A"
    elapsed: float = 0.0
    activities: list[str] = dataclasses.field(default_factory=list)
    command: str | None = None
    paused: bool = False
    show_detail: bool = True
    show_artifacts: bool = True


def _number(text: str) -> int | None:
    match = re.search(r"\d[\d,]*", text)
    return int(match.group(0).replace(",", "")) if match else None


def _decimal(text: str) -> float | None:
    match = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    return float(match.group(0).replace(",", "")) if match else None


def extract_readme_metrics(text: str) -> dict[str, Any]:
    """Extract only explicitly documented task measurements from Markdown."""

    metrics: dict[str, Any] = {}
    for raw_line in text.splitlines():
        if "|" not in raw_line:
            continue
        cells = [cell.strip().strip("*") for cell in raw_line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        label, value = cells[0].lower(), cells[1]
        if not value or set(value) <= {"-", ":", " "}:
            continue
        pair = re.search(r"(\d[\d,]*)\s*/\s*(\d[\d,]*)", value)
        if any(term in label for term in ("supplied examples", "packaged examples", "provided examples")) and pair:
            metrics["validation_examples"] = {
                "passed": int(pair.group(1).replace(",", "")),
                "total": int(pair.group(2).replace(",", "")),
            }
        elif "exhaustive" in label and pair:
            metrics["exhaustive_cases"] = {
                "passed": int(pair.group(1).replace(",", "")),
                "total": int(pair.group(2).replace(",", "")),
            }
        elif any(term in label for term in ("intermediate tensors", "intermediate memory")) or label == "memory":
            value_number = _number(value)
            if value_number is not None:
                metrics["intermediate_memory_bytes"] = value_number
        elif any(term in label for term in ("initializer parameters", "parameter count")) or label == "parameters":
            value_number = _number(value)
            if value_number is not None:
                metrics["parameters"] = value_number
        elif "score" in label or label.startswith("points"):
            value_number = _decimal(value)
            if value_number is not None:
                metrics["estimated_score"] = value_number
        elif "cost" in label or label.startswith("objective"):
            value_number = _number(value)
            if value_number is not None:
                metrics["objective_cost"] = value_number
        elif "serialized" in label or "onnx file size" in label:
            value_number = _number(value)
            if value_number is not None:
                metrics["serialized_model_bytes"] = value_number

    if "validation_examples" not in metrics:
        match = re.search(r"passes all\s+(\d[\d,]*)\s+(?:provided\s+)?(?:examples|cases)", text, re.IGNORECASE)
        if match:
            count = int(match.group(1).replace(",", ""))
            metrics["validation_examples"] = {"passed": count, "total": count}

    if "exhaustive_cases" not in metrics:
        patterns = (
            r"\(([\d,]+)\s+cases\),\s+with zero failures",
            r"exhaustive(?:ly)?[^\n]{0,100}?([\d,]+)\s+cases[^\n]{0,40}?zero failures",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                count = int(match.group(1).replace(",", ""))
                metrics["exhaustive_cases"] = {"passed": count, "total": count}
                break

    node_match = re.search(r"(?:with|only)\s+(\d+|" + "|".join(NUMBER_WORDS) + r")\s+nodes?", text, re.IGNORECASE)
    if node_match:
        raw = node_match.group(1).lower()
        metrics["onnx_node_count"] = int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
    return metrics


def validate_fixture(data: Any) -> dict[str, Any]:
    """Validate and return a CodexForge replay fixture."""

    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise FixtureError("fixture schema_version must be 1")
    for key in ("title", "duration_seconds", "model", "reasoning", "parallel_limit", "tasks", "events"):
        if key not in data:
            raise FixtureError(f"fixture is missing {key!r}")
    if not isinstance(data["duration_seconds"], (int, float)) or data["duration_seconds"] <= 0:
        raise FixtureError("duration_seconds must be positive")
    if not isinstance(data["parallel_limit"], int) or data["parallel_limit"] < 1:
        raise FixtureError("parallel_limit must be a positive integer")
    if not isinstance(data["tasks"], list) or not data["tasks"]:
        raise FixtureError("tasks must be a non-empty list")
    task_names: set[str] = set()
    for task in data["tasks"]:
        if not isinstance(task, dict):
            raise FixtureError("each task must be an object")
        name = task.get("task")
        if not isinstance(name, str) or not re.fullmatch(r"task\d{3}", name):
            raise FixtureError(f"invalid task name: {name!r}")
        if name in task_names:
            raise FixtureError(f"duplicate task: {name}")
        task_names.add(name)
        if not isinstance(task.get("principle"), str) or not isinstance(task.get("metrics"), dict):
            raise FixtureError(f"invalid metadata for {name}")
        if not isinstance(task.get("artifacts"), list):
            raise FixtureError(f"invalid artifacts for {name}")
    previous = -1.0
    for event in data["events"]:
        if not isinstance(event, dict):
            raise FixtureError("each event must be an object")
        at = event.get("at")
        if not isinstance(at, (int, float)) or at < previous:
            raise FixtureError("events must be ordered by non-negative 'at' values")
        previous = float(at)
        task = event.get("task")
        if task is not None and task not in task_names:
            raise FixtureError(f"event references unknown task: {task}")
        if event.get("phase") not in VALID_PHASES:
            raise FixtureError(f"invalid replay phase: {event.get('phase')!r}")
        if event.get("status") not in VALID_STATUSES:
            raise FixtureError(f"invalid replay status: {event.get('status')!r}")
        if not isinstance(event.get("action"), str) or not event["action"].strip():
            raise FixtureError("each replay event needs a non-empty action")
    if data["events"] and data["events"][-1]["at"] > data["duration_seconds"]:
        raise FixtureError("last event exceeds duration_seconds")
    return data


def load_fixture(path: Path) -> dict[str, Any]:
    try:
        return validate_fixture(json.loads(path.read_text(encoding="utf-8")))
    except OSError as error:
        raise FixtureError(f"cannot read replay fixture {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise FixtureError(f"invalid JSON in replay fixture {path}: {error}") from error


def redact_text(text: str, *, root: Path = REPO_ROOT, limit: int = 180) -> str:
    """Remove likely secrets/account identifiers and shorten local paths."""

    clean = text.replace(str(root), ".")
    clean = re.sub(r"/home/[^/\s]+", "~", clean)
    for index, pattern in enumerate(SECRET_PATTERNS):
        if index == 0:
            clean = pattern.sub(r"\1[REDACTED]", clean)
        elif index == 1:
            clean = pattern.sub(r"\1\2[REDACTED]", clean)
        elif index == len(SECRET_PATTERNS) - 1:
            clean = pattern.sub("[redacted-account]", clean)
        else:
            clean = pattern.sub("[REDACTED]", clean)
    clean = " ".join(clean.replace("\x00", "").split())
    return clean if len(clean) <= limit else clean[: max(1, limit - 1)] + "…"


def infer_phase(text: str, fallback: str = "queued") -> str:
    lower = text.lower()
    checks = (
        ("failed", (" failed", "failure", "traceback", "error:")),
        ("writing documentation", ("readme", "documentation", "documenting")),
        ("optimizing graph", ("optimizing", "reduced", "smaller", "candidate", "improved")),
        ("profiling official score", ("official score", "scorer", "profiling", "memory + parameters", "cost")),
        ("exhaustive validation", ("exhaustive", "generated cases", "all generator")),
        ("validating examples", ("validating", "validation", "examples", "test cases")),
        ("running ONNX Runtime", ("onnx runtime", "onnxruntime", "inference")),
        ("building ONNX", ("building", "build", "onnx graph", ".onnx")),
        ("inferring rule", ("inferring", "derive", "rule", "transformation")),
        ("reading generator", ("generator", "task_", "arc-gen")),
        ("inspecting examples", ("inspect", "dataset", "sample")),
    )
    for phase, terms in checks:
        if any(term in lower for term in terms):
            return phase
    return fallback


def parse_jsonl_event(line: str, task: str) -> dict[str, str] | None:
    """Convert one Codex JSONL record into a redacted dashboard event."""

    try:
        event = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(event, dict):
        return None
    event_type = str(event.get("type", ""))
    if event_type == "turn.completed":
        return {"task": task, "phase": "completed", "status": "completed", "action": "Codex worker finished"}
    if event_type in {"error", "turn.failed"}:
        action = redact_text(str(event.get("message") or event.get("error") or "Codex worker failed"))
        return {"task": task, "phase": "failed", "status": "failed", "action": action}
    item = event.get("item")
    if event_type not in {"item.started", "item.completed", "item.updated"} or not isinstance(item, dict):
        return None
    item_type = item.get("type")
    action = ""
    if item_type == "agent_message":
        action = str(item.get("text", ""))
    elif item_type == "command_execution":
        command = str(item.get("command", ""))
        verb = "Finished" if event_type == "item.completed" else "Running"
        action = f"{verb}: {command}"
        if item.get("exit_code") not in (None, 0):
            action += f" (exit {item['exit_code']})"
    elif item_type == "todo_list":
        items = item.get("items")
        if isinstance(items, list):
            current = next((entry for entry in items if isinstance(entry, dict) and not entry.get("completed")), None)
            if current:
                action = str(current.get("text", ""))
    elif item_type in {"web_search", "mcp_tool_call"}:
        action = "Consulting an external reference"
    if not action:
        return None
    action = redact_text(action)
    phase = infer_phase(action, "inspecting examples")
    status = "failed" if phase == "failed" else "running"
    return {"task": task, "phase": phase, "status": status, "action": action}


def expand_task_specs(specs: Iterable[str]) -> list[str]:
    tasks: set[int] = set()
    for spec in specs:
        for token in spec.split(","):
            token = token.strip().lower().removeprefix("task")
            if not token:
                raise ValueError("empty task specification")
            if "-" in token:
                parts = token.split("-")
                if len(parts) != 2 or not all(part.isdigit() for part in parts):
                    raise ValueError(f"invalid task range: {spec!r}")
                start, end = map(int, parts)
                if start > end:
                    raise ValueError(f"descending task range: {spec!r}")
                tasks.update(range(start, end + 1))
            elif token.isdigit():
                tasks.add(int(token))
            else:
                raise ValueError(f"invalid task specification: {spec!r}")
    if not tasks or min(tasks) < 1 or max(tasks) > 400:
        raise ValueError("tasks must be between 1 and 400")
    return [f"task{number:03d}" for number in sorted(tasks)]


def detect_artifacts(root: Path, task: str) -> dict[str, Any]:
    workspace = root / "tasks" / task
    onnx_files = sorted(workspace.glob("*.onnx")) if workspace.is_dir() else []
    event_files = sorted((root / "logs").glob(f"{task}*.events.jsonl"))
    paths = {
        "readme": workspace / "README.md",
        "submission": root / "submission" / f"{task}.onnx",
        "result": root / "results" / f"{task}.md",
    }
    return {
        "readme": paths["readme"].is_file(),
        "task_onnx": [path.relative_to(root).as_posix() for path in onnx_files],
        "submission": paths["submission"].is_file(),
        "result": paths["result"].is_file(),
        "events": bool(event_files),
        "event_files": [path.relative_to(root).as_posix() for path in event_files],
    }


def artifact_label(artifacts: dict[str, Any]) -> str:
    found = sum(
        (
            bool(artifacts.get("readme")),
            bool(artifacts.get("task_onnx")),
            bool(artifacts.get("submission")),
            bool(artifacts.get("result")),
            bool(artifacts.get("events")),
        )
    )
    return f"{found}/5 ready"


def artifact_paths(artifacts: dict[str, Any], task: str) -> list[str]:
    paths: list[str] = []
    if artifacts.get("readme"):
        paths.append(f"tasks/{task}/README.md")
    paths.extend(str(path) for path in artifacts.get("task_onnx", []))
    if artifacts.get("submission"):
        paths.append(f"submission/{task}.onnx")
    if artifacts.get("result"):
        paths.append(f"results/{task}.md")
    if artifacts.get("events"):
        paths.extend(str(path) for path in artifacts.get("event_files", []))
    return paths


def read_task_metrics(root: Path, task: str) -> dict[str, Any]:
    path = root / "tasks" / task / "README.md"
    try:
        return extract_readme_metrics(path.read_text(encoding="utf-8"))
    except OSError:
        return {}


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, second = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{second:02d}" if hours else f"{minutes:02d}:{second:02d}"


def metric_summary(metrics: dict[str, Any]) -> str:
    parts: list[str] = []
    validation = metrics.get("validation_examples")
    if isinstance(validation, dict):
        parts.append(f"examples {validation.get('passed')}/{validation.get('total')}")
    exhaustive = metrics.get("exhaustive_cases")
    if isinstance(exhaustive, dict):
        parts.append(f"exhaustive {exhaustive.get('passed')}/{exhaustive.get('total')}")
    labels = (
        ("intermediate_memory_bytes", "memory", "B"),
        ("parameters", "params", ""),
        ("objective_cost", "cost", ""),
        ("estimated_score", "score", ""),
        ("onnx_node_count", "nodes", ""),
        ("serialized_model_bytes", "model", "B"),
    )
    for key, label, suffix in labels:
        value = metrics.get(key)
        if value is not None:
            rendered = f"{value:,.6g}" if isinstance(value, float) else f"{value:,}"
            parts.append(f"{label} {rendered}{suffix}")
    return " · ".join(parts) if parts else "No documented measurements found"


class Palette:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def apply(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def cyan(self, text: str) -> str:
        return self.apply("1;36", text)

    def green(self, text: str) -> str:
        return self.apply("1;32", text)

    def yellow(self, text: str) -> str:
        return self.apply("1;33", text)

    def red(self, text: str) -> str:
        return self.apply("1;31", text)

    def dim(self, text: str) -> str:
        return self.apply("2", text)


class Renderer:
    def __init__(self, *, color: bool, stream: TextIO = sys.stdout) -> None:
        self.stream = stream
        self.dynamic = stream.isatty()
        self.palette = Palette(color and stream.isatty())
        self._rendered = False

    @staticmethod
    def _crop(text: str, width: int) -> str:
        return text if len(text) <= width else text[: max(1, width - 1)] + "…"

    def frame(self, state: DashboardState) -> str:
        width = max(60, min(140, shutil.get_terminal_size((120, 30)).columns))
        p = self.palette
        badge = {"replay": "DEMO REPLAY", "live": "LIVE", "attached": "ATTACHED"}[state.mode]
        counts = {status: 0 for status in VALID_STATUSES}
        for worker in state.workers.values():
            counts[worker.status] = counts.get(worker.status, 0) + 1
        lines = [
            p.cyan("CodexForge") + "  " + p.dim("Built by Codex for Codex") + "  " + p.yellow(f"[{badge}]"),
            self._crop("A Codex-built orchestration system: visual rules → validated executable ONNX programs", width),
            "─" * width,
            (
                f"Tasks {len(state.workers)}  Queued {counts['queued']}  Running {counts['running']}  "
                f"Completed {counts['completed']}  Failed {counts['failed']}  Skipped {counts['skipped']}  "
                f"Elapsed {format_duration(state.elapsed)}  Workers ≤{state.parallel}"
            ),
            self._crop(
                f"Model {state.model} / {state.reasoning}  Quota remaining {state.quota_remaining}  "
                f"Reset credits {state.reset_credits}",
                width,
            ),
        ]
        if state.command:
            lines.append(self._crop(f"Command: {state.command}", width))
        lines.extend(["", p.cyan("Parallel workers")])
        if width >= 105:
            action_width = max(18, width - 78)
            header = f"{'Task':<8} {'Principle':<12} {'Phase':<22} {'Try':>3} {'Elapsed':>8} {'Status':<10} {'Artifact':<11} Action"
            lines.append(p.dim(self._crop(header, width)))
            for worker in state.workers.values():
                status = worker.status
                colored = p.green(status) if status == "completed" else p.red(status) if status == "failed" else p.yellow(status) if status == "running" else status
                row = (
                    f"{worker.task:<8} {self._crop(worker.principle, 12):<12} {self._crop(worker.phase, 22):<22} "
                    f"{worker.attempt:>3} {format_duration(worker.elapsed):>8} {colored:<10} "
                    f"{self._crop(worker.artifact, 11):<11} {self._crop(worker.action, action_width)}"
                )
                lines.append(row)
        else:
            for worker in state.workers.values():
                lines.append(self._crop(f"{worker.task} [{worker.status}] {worker.phase} · {format_duration(worker.elapsed)} · {worker.artifact}", width))
                lines.append(self._crop(f"  {worker.action}", width))

        lines.extend(["", p.cyan("Latest meaningful Codex activity")])
        activities = state.activities[-8 if state.show_detail else -3 :]
        if activities:
            lines.extend(self._crop(f"• {activity}", width) for activity in activities)
        else:
            lines.append(p.dim("• Waiting for meaningful activity"))

        if state.show_artifacts:
            lines.extend(["", p.cyan("Verified task evidence and artifacts")])
            for worker in state.workers.values():
                if worker.status == "completed" or state.mode != "replay":
                    lines.append(self._crop(f"{worker.task}  {metric_summary(worker.metrics)}", width))
                    if worker.artifact_paths:
                        lines.append(self._crop("  ↳ " + " · ".join(worker.artifact_paths), width))
                elif state.mode == "replay":
                    lines.append(f"{worker.task}  Evidence appears when this replay worker completes")

        pause = "PAUSED · " if state.paused else ""
        controls = (
            "q quit  p pause/resume  r restart  + / - speed  l activity detail  a artifacts"
            if state.mode == "replay"
            else "q quit  l toggle activity detail  a toggle artifact panel"
        )
        lines.extend(
            [
                "",
                "─" * width,
                self._crop(f"{pause}{controls}", width),
                p.dim("Replay is an authored timeline; every final measurement comes from committed repository evidence.")
                if state.mode == "replay"
                else p.dim("Observing repository files generated by the existing runner and supervisor."),
            ]
        )
        return "\n".join(lines) + "\n"

    def render(self, state: DashboardState) -> None:
        if self.dynamic:
            self.stream.write("\033[2J\033[H")
        elif self._rendered:
            self.stream.write("\n")
        self.stream.write(self.frame(state))
        self.stream.flush()
        self._rendered = True


@contextlib.contextmanager
def terminal_input() -> Iterable[None]:
    if not sys.stdin.isatty():
        yield
        return
    descriptor = sys.stdin.fileno()
    old = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, old)


def read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return sys.stdin.read(1) if ready else None


def apply_replay_event(state: DashboardState, event: dict[str, Any], now: float) -> None:
    task = event.get("task")
    action = redact_text(event["action"])
    if task is not None:
        worker = state.workers[task]
        if worker.started_at is None and event["status"] == "running":
            worker.started_at = now
        worker.phase = event["phase"]
        worker.status = event["status"]
        worker.action = action
        worker.attempt = int(event.get("attempt", worker.attempt))
        if worker.status == "completed":
            worker.artifact = "5/5 ready"
            if worker.started_at is not None:
                worker.elapsed = now - worker.started_at
    state.activities.append(f"{task or 'system':<8} {action}")


def run_replay(args: argparse.Namespace, renderer: Renderer | None = None) -> int:
    fixture = load_fixture(Path(args.fixture))
    selected = set(expand_task_specs(args.tasks)) if args.tasks else None
    task_records = [task for task in fixture["tasks"] if selected is None or task["task"] in selected]
    if not task_records:
        raise FixtureError("none of the selected tasks are available in this replay fixture")
    allowed = {task["task"] for task in task_records}
    events = [event for event in fixture["events"] if event.get("task") in (None, *allowed)]
    workers = {
        task["task"]: WorkerState(
            task=task["task"],
            principle=task["principle"],
            metrics=task["metrics"],
            artifact_paths=[str(record["path"]) for record in task["artifacts"]],
        )
        for task in task_records
    }
    state = DashboardState(
        mode="replay",
        workers=workers,
        parallel=min(fixture["parallel_limit"], max(1, len(workers))),
        model=fixture["model"],
        reasoning=fixture["reasoning"],
    )
    renderer = renderer or Renderer(color=not args.no_color)
    speed = args.speed
    duration = float(fixture["duration_seconds"])
    stop = False
    index = 0
    replay_elapsed = 0.0
    real_started = time.monotonic()
    last_tick = time.monotonic()
    last_render = -1.0
    renderer.render(state)
    with terminal_input():
        while not stop and replay_elapsed < duration:
            now = time.monotonic()
            delta = now - last_tick
            last_tick = now
            key = read_key()
            if key == "q":
                break
            if key == "p":
                state.paused = not state.paused
            elif key == "r":
                index = 0
                replay_elapsed = 0.0
                for worker in state.workers.values():
                    worker.phase, worker.status = "queued", "queued"
                    worker.action, worker.artifact = "Waiting for a worker slot", "pending"
                    worker.started_at, worker.elapsed = None, 0.0
                state.activities.clear()
            elif key in {"+", "="}:
                speed = min(32.0, speed * 2)
            elif key == "-":
                speed = max(0.25, speed / 2)
            elif key == "l":
                state.show_detail = not state.show_detail
            elif key == "a":
                state.show_artifacts = not state.show_artifacts
            if not state.paused:
                replay_elapsed += delta * speed
            changed = False
            while index < len(events) and float(events[index]["at"]) <= replay_elapsed:
                apply_replay_event(state, events[index], replay_elapsed)
                index += 1
                changed = True
            state.elapsed = replay_elapsed
            for worker in state.workers.values():
                if worker.status == "running" and worker.started_at is not None:
                    worker.elapsed = replay_elapsed - worker.started_at
            if changed or (renderer.dynamic and replay_elapsed - last_render >= 1.0):
                renderer.render(state)
                last_render = replay_elapsed
            if args.max_runtime and now - real_started >= args.max_runtime:
                stop = True
            time.sleep(0.05)
    state.elapsed = min(replay_elapsed, duration)
    renderer.render(state)
    return 0


class JsonlObserver:
    def __init__(self) -> None:
        self.offsets: dict[Path, tuple[int, int]] = {}

    def prime(self, path: Path) -> None:
        """Treat a file's current contents as the observation baseline."""
        try:
            stat = path.stat()
            self.offsets[path] = (stat.st_ino, stat.st_size)
        except OSError:
            pass

    def read(self, path: Path) -> list[str]:
        try:
            stat = path.stat()
            inode, offset = self.offsets.get(path, (stat.st_ino, 0))
            if inode != stat.st_ino or stat.st_size < offset:
                offset = 0
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                lines = handle.readlines()
                self.offsets[path] = (stat.st_ino, handle.tell())
            return lines
        except OSError:
            return []


def discover_tasks(root: Path) -> list[str]:
    names = {match.group(1) for path in (root / "logs").glob("task*.events.jsonl") if (match := re.search(r"(task\d{3})", path.name))}
    names.update(path.stem for path in (root / "submission").glob("task???.onnx"))
    return sorted(names)


def load_run_summary(root: Path) -> dict[str, Any]:
    try:
        data = json.loads((root / "logs" / "codex-run-summary.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def run_summary_stamp(root: Path) -> tuple[int, int] | None:
    """Return an identity/change stamp for the runner's summary file."""
    try:
        stat = (root / "logs" / "codex-run-summary.json").stat()
        return stat.st_ino, stat.st_mtime_ns
    except OSError:
        return None


def update_from_summary(state: DashboardState, root: Path) -> None:
    summary = load_run_summary(root)
    if isinstance(summary.get("parallelism"), int):
        state.parallel = summary["parallelism"]
    state.model = str(summary.get("model") or state.model)
    state.reasoning = str(summary.get("reasoning") or state.reasoning)
    for record in summary.get("tasks", []):
        if not isinstance(record, dict) or record.get("task") not in state.workers:
            continue
        worker = state.workers[record["task"]]
        status = str(record.get("status", worker.status))
        if status in VALID_STATUSES:
            worker.status = status
            worker.phase = status if status in {"completed", "failed", "skipped"} else worker.phase
        worker.attempt = int(record.get("attempts") or worker.attempt)
        worker.elapsed = float(record.get("elapsed_seconds") or worker.elapsed)


def parse_quota_line(line: str, state: DashboardState) -> None:
    match = re.search(r"remaining=(\d+)%", line)
    if match:
        state.quota_remaining = f"{match.group(1)}%"
    credits = re.search(r"reset credits=(\d+)", line)
    if credits:
        state.reset_credits = credits.group(1)


def parse_runner_line(line: str, state: DashboardState) -> None:
    """Update live metadata emitted by run_codex_tasks.py at startup."""
    match = re.search(r"\bmodel=([^,\s]+),\s*reasoning=([^,\s]+)", line)
    if match:
        state.model = match.group(1)
        state.reasoning = match.group(2)


def _reader_thread(stream: TextIO, output: queue.Queue[str]) -> None:
    try:
        for line in stream:
            output.put(line.rstrip())
    finally:
        stream.close()


def build_live_commands(args: argparse.Namespace, root: Path = REPO_ROOT, python: str = sys.executable) -> list[list[str]]:
    tasks = expand_task_specs(args.tasks or ["11-12"])
    runner = [
        python,
        str(root / "run_codex_tasks.py"),
        "--tasks",
        *[str(int(task.removeprefix("task"))) for task in tasks],
        "--parallel",
        str(args.parallel),
    ]
    if args.force:
        runner.append("--force")
    commands = [runner]
    if args.with_supervisor:
        supervisor = [python, str(root / "codex_quota_supervisor.py")]
        if not args.allow_reset_credit:
            supervisor.append("--dry-run")
        commands.append(supervisor)
    return commands


def terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 8
    for process in processes:
        if process.poll() is None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
    for process in processes:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)


def run_observer(args: argparse.Namespace, *, live: bool) -> int:
    root = REPO_ROOT
    task_names = expand_task_specs(args.tasks) if args.tasks else (expand_task_specs(["11-12"]) if live else discover_tasks(root))
    if not task_names:
        print("No task logs or submissions are available to attach to.", file=sys.stderr)
        return 2
    principles = {}
    try:
        raw = json.loads((root / "utils" / "task_principles.json").read_text(encoding="utf-8"))
        principles = {name: str(value.get("principle", "N/A")) for name, value in raw.items() if isinstance(value, dict)}
    except (OSError, json.JSONDecodeError):
        pass
    workers = {
        task: WorkerState(
            task=task,
            principle=principles.get(task, "N/A"),
            metrics=read_task_metrics(root, task),
            artifact=artifact_label(detected := detect_artifacts(root, task)),
            artifact_paths=artifact_paths(detected, task),
        )
        for task in task_names
    }
    state = DashboardState(
        mode="live" if live else "attached",
        workers=workers,
        parallel=args.parallel,
        model=DEFAULT_LIVE_MODEL if live else "N/A",
        reasoning=DEFAULT_LIVE_REASONING if live else "N/A",
    )
    commands = build_live_commands(args, root) if live else []
    if commands:
        state.command = shlex.join(commands[0])
        print(f"LIVE command: {state.command}", file=sys.stderr)
        if len(commands) > 1:
            label = "reset enabled" if args.allow_reset_credit else "observation only / dry-run"
            print(f"Supervisor command ({label}): {shlex.join(commands[1])}", file=sys.stderr)
    renderer = Renderer(color=not args.no_color)
    observer = JsonlObserver()
    summary_baseline = run_summary_stamp(root) if live else None
    if live:
        # A live launch represents a new run. Ignore historical worker output
        # until the runner truncates/replaces it or appends fresh events. Attach
        # mode intentionally reads those files from the beginning instead.
        for task in task_names:
            for pattern in (f"{task}*.events.jsonl", f"{task}*.stderr"):
                for path in (root / "logs").glob(pattern):
                    observer.prime(path)
    watched_mtimes: dict[Path, int] = {}
    output: queue.Queue[str] = queue.Queue()
    processes: list[subprocess.Popen[str]] = []
    start = time.monotonic()
    stop = False
    try:
        for command in commands:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            processes.append(process)
            assert process.stdout is not None
            threading.Thread(target=_reader_thread, args=(process.stdout, output), daemon=True).start()
        with terminal_input():
            while not stop:
                changed = False
                for task, worker in state.workers.items():
                    event_paths = sorted((root / "logs").glob(f"{task}*.events.jsonl"))
                    for event_path in event_paths:
                        for line in observer.read(event_path):
                            parsed = parse_jsonl_event(line, task)
                            if parsed:
                                worker.phase = parsed["phase"]
                                worker.status = parsed["status"]
                                worker.action = parsed["action"]
                                state.activities.append(f"{task:<8} {parsed['action']}")
                                changed = True
                    stderr_paths = sorted((root / "logs").glob(f"{task}*.stderr"))
                    for stderr_path in stderr_paths:
                        for line in observer.read(stderr_path):
                            clean = redact_text(line)
                            if clean:
                                state.activities.append(f"{task:<8} stderr: {clean}")
                                changed = True
                    artifacts = detect_artifacts(root, task)
                    label = artifact_label(artifacts)
                    if label != worker.artifact:
                        worker.artifact = label
                        worker.artifact_paths = artifact_paths(artifacts, task)
                        worker.metrics = read_task_metrics(root, task)
                        changed = True
                quota_state = root / "logs" / "quota-supervisor-state.json"
                supervisor_stderr = root / "logs" / "quota-supervisor-app-server.stderr"
                for watched, label in (
                    (quota_state, "Quota supervisor state file changed (account details withheld)"),
                    (supervisor_stderr, "Quota supervisor diagnostic file changed (details withheld)"),
                ):
                    try:
                        stamp = watched.stat().st_mtime_ns
                    except OSError:
                        continue
                    if watched_mtimes.get(watched) != stamp:
                        watched_mtimes[watched] = stamp
                        state.activities.append(label)
                        changed = True
                while True:
                    try:
                        line = output.get_nowait()
                    except queue.Empty:
                        break
                    clean = redact_text(line)
                    if clean:
                        parse_quota_line(clean, state)
                        parse_runner_line(clean, state)
                        state.activities.append(clean)
                        changed = True
                summary_stamp = run_summary_stamp(root)
                if not live or summary_stamp != summary_baseline:
                    update_from_summary(state, root)
                    if live:
                        summary_baseline = summary_stamp
                    changed = True
                state.elapsed = time.monotonic() - start
                key = read_key()
                if key == "q":
                    break
                if key == "l":
                    state.show_detail = not state.show_detail
                    changed = True
                elif key == "a":
                    state.show_artifacts = not state.show_artifacts
                    changed = True
                if changed or renderer.dynamic:
                    renderer.render(state)
                if live and processes and processes[0].poll() is not None:
                    update_from_summary(state, root)
                    renderer.render(state)
                    return processes[0].returncode or 0
                if args.max_runtime and state.elapsed >= args.max_runtime:
                    stop = True
                time.sleep(0.5)
    finally:
        terminate_processes(processes)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CodexForge terminal demonstration dashboard")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--replay", action="store_true", help="run the quota-free factual replay (default)")
    modes.add_argument("--live", action="store_true", help="launch and visualize the existing Codex runner")
    modes.add_argument("--attach", action="store_true", help="observe independently launched runner/supervisor files")
    parser.add_argument("--tasks", nargs="+", help="task numbers/ranges; live default: 11-12")
    parser.add_argument("--parallel", type=int, default=2, help="live worker limit (default: 2)")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colour and screen control")
    parser.add_argument("--force", action="store_true", help="forward --force to the live runner")
    parser.add_argument("--with-supervisor", action="store_true", help="explicitly start the quota supervisor in live mode")
    parser.add_argument(
        "--allow-reset-credit",
        action="store_true",
        help="allow the explicitly started supervisor to consume an eligible reset credit",
    )
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE), help=argparse.SUPPRESS)
    parser.add_argument("--max-runtime", type=float, default=0.0, help=argparse.SUPPRESS)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    if args.speed <= 0:
        parser.error("--speed must be greater than zero")
    if args.max_runtime < 0:
        parser.error("--max-runtime cannot be negative")
    if args.allow_reset_credit and not args.with_supervisor:
        parser.error("--allow-reset-credit requires --with-supervisor")
    if (args.with_supervisor or args.allow_reset_credit or args.force) and not args.live:
        parser.error("supervisor and force options are available only with --live")
    if args.tasks:
        try:
            expand_task_specs(args.tasks)
        except ValueError as error:
            parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.live:
            return run_observer(args, live=True)
        if args.attach:
            return run_observer(args, live=False)
        return run_replay(args)
    except FixtureError as error:
        print(f"Replay fixture error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
