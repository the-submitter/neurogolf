#!/usr/bin/env python3
"""Build a deterministic CodexForge replay fixture from repository evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from codexforge_dashboard import (  # noqa: E402
    expand_task_specs,
    extract_readme_metrics,
    validate_fixture,
)


DEFAULT_TASKS = ("task001", "task002", "task010")
STRATEGIES = {
    "task001": {
        "rule": "Inferring the Kronecker-square fractal rule",
        "build": "Building a single terminal Einsum with shared coordinate factors",
    },
    "task002": {
        "rule": "Inferring green pot-border detection and yellow interior fill",
        "build": "Building quantized template detectors with QLinearConv",
    },
    "task010": {
        "rule": "Inferring bar-height ranking from four distinct columns",
        "build": "Building a four-node Einsum + TopK + Gather graph",
    },
}
DEFAULT_OFFSETS = {
    "task001": (3, 11, 19, 28, 38, 50, 61),
    "task002": (5, 13, 22, 31, 42, 54, 68),
    "task010": (7, 15, 24, 34, 45, 57, 73),
}


def timeline(tasks: tuple[str, ...]) -> tuple[dict[str, tuple[int, ...]], int]:
    if tasks == DEFAULT_TASKS:
        return DEFAULT_OFFSETS, 78
    offsets = {}
    parallel = min(3, len(tasks))
    for index, task in enumerate(tasks):
        batch, slot = divmod(index, parallel)
        start = 3 + batch * 63 + slot * 2
        offsets[task] = (
            start,
            start + 8,
            start + 16,
            start + 25,
            start + 35,
            start + 47,
            start + 58,
        )
    duration = max(values[-1] for values in offsets.values()) + 5
    return offsets, duration


def artifact_records(root: Path, task: str) -> list[dict[str, Any]]:
    candidates = [
        root / "tasks" / task / "README.md",
        root / "submission" / f"{task}.onnx",
        root / "results" / f"{task}.md",
        root / "logs" / f"{task}.events.jsonl",
    ]
    candidates.extend(sorted((root / "tasks" / task).glob("*.onnx")))
    records = []
    for path in candidates:
        if path.is_file():
            records.append({"path": path.relative_to(root).as_posix()})
    return records


def build_fixture(root: Path = REPO_ROOT, tasks: tuple[str, ...] = DEFAULT_TASKS) -> dict[str, Any]:
    principles = json.loads((root / "utils" / "task_principles.json").read_text(encoding="utf-8"))
    records = []
    for task in tasks:
        readme = root / "tasks" / task / "README.md"
        if not readme.is_file():
            raise FileNotFoundError(f"missing evidence README: {readme}")
        metadata = principles.get(task)
        if not isinstance(metadata, dict):
            raise ValueError(f"missing task metadata for {task}")
        records.append(
            {
                "task": task,
                "task_id": metadata["task_id"],
                "principle": metadata["principle"],
                "metrics": extract_readme_metrics(readme.read_text(encoding="utf-8")),
                "artifacts": artifact_records(root, task),
                "evidence": readme.relative_to(root).as_posix(),
            }
        )

    events: list[dict[str, Any]] = [
        {
            "at": 0,
            "task": None,
            "phase": "queued",
            "status": "queued",
            "action": (
                "Replay preflight passed; selected "
                f"{len(records)} repository-backed task stories"
            ),
        }
    ]
    offsets, duration = timeline(tasks)
    for record in records:
        task = record["task"]
        start, inspect, infer, build, validate, profile, complete = offsets[task]
        metrics = record["metrics"]
        validation_count = metrics.get("validation_examples", {}).get("total")
        validate_action = f"Validating the ONNX program against {validation_count} packaged examples" if validation_count else "Validating the ONNX program against packaged examples"
        strategy = STRATEGIES.get(
            task,
            {
                "rule": f"Inferring the documented {record['principle']} transformation",
                "build": "Building the documented optimized ONNX graph",
            },
        )
        events.extend(
            [
                {"at": start, "task": task, "phase": "inspecting examples", "status": "running", "action": "Worker activated; inspecting task examples and existing artifacts"},
                {"at": inspect, "task": task, "phase": "reading generator", "status": "running", "action": f"Reading ARC-GEN generator task_{record['task_id']}.py ({record['principle']})"},
                {"at": infer, "task": task, "phase": "inferring rule", "status": "running", "action": strategy["rule"]},
                {"at": build, "task": task, "phase": "building ONNX", "status": "running", "action": strategy["build"]},
                {"at": validate, "task": task, "phase": "validating examples", "status": "running", "action": validate_action},
                {"at": profile, "task": task, "phase": "profiling official score", "status": "running", "action": "Profiling documented memory, parameters, objective cost, and score"},
                {"at": complete, "task": task, "phase": "completed", "status": "completed", "action": "Verified model, submission, result, event log, and task README are preserved"},
            ]
        )
    events.append(
        {
            "at": duration - 2,
            "task": None,
            "phase": "completed",
            "status": "completed",
            "action": "Replay complete: parallel reasoning became validated executable ONNX artifacts",
        }
    )
    events.sort(key=lambda event: event["at"])
    fixture = {
        "schema_version": 1,
        "title": "CodexForge: Built by Codex for Codex",
        "description": "Authored replay timeline whose final facts are extracted from committed repository evidence.",
        "duration_seconds": duration,
        "model": "gpt-5.6-sol",
        "reasoning": "high",
        "parallel_limit": min(3, len(records)),
        "tasks": records,
        "events": events,
    }
    return validate_fixture(fixture)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEMO_DIR / "fixtures" / "codexforge_demo.json")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="task numbers/ranges to snapshot (default: 1, 2, and 10)",
    )
    parser.add_argument(
        "--check", action="store_true", help="fail if the saved fixture is not current"
    )
    args = parser.parse_args()
    try:
        tasks = tuple(expand_task_specs(args.tasks))
    except ValueError as error:
        parser.error(str(error))
    data = build_fixture(tasks=tasks)
    rendered = json.dumps(data, indent=2, sort_keys=False) + "\n"
    output = args.output.expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    try:
        display_output = output.relative_to(REPO_ROOT)
    except ValueError:
        display_output = output
    if args.check:
        try:
            current = output.read_text(encoding="utf-8")
        except OSError as error:
            print(error, file=sys.stderr)
            return 1
        if current != rendered:
            print(f"Replay fixture is stale: run {Path(__file__).relative_to(REPO_ROOT)}", file=sys.stderr)
            return 1
        print(f"Replay fixture is current: {display_output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {display_output} from {len(data['tasks'])} task READMEs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
