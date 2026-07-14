"""Prepare and solve every mapped task with bounded Codex concurrency."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from neurogolf.config import NeuroGolfSettings, load_settings
from neurogolf.orchestrator.codex import DEFAULT_CODEX_PROFILE
from neurogolf.orchestrator.solve import PromptKind, solve_many
from neurogolf.tasks.mapping import TaskMap


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare every mapped task workspace and run one bounded Codex solver epoch per task."
        )
    )
    parser.add_argument(
        "--max-parallel",
        type=_positive_int,
        help="Maximum simultaneous Codex processes (default: configured value, normally 4)",
    )
    parser.add_argument(
        "--kind",
        choices=("solve", "repair", "golf", "review"),
        default="solve",
        help="Prompt kind sent to every task worker",
    )
    parser.add_argument("--epoch", type=_positive_int, default=1)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start new sessions instead of resuming eligible prior sessions",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the batch plan without preparing tasks or launching Codex",
    )
    return parser


def _load_settings(max_parallel: int | None) -> NeuroGolfSettings:
    overrides = (
        {"codex": {"max_parallel_tasks": max_parallel}} if max_parallel is not None else None
    )
    return load_settings(overrides=overrides)


async def solve_all(
    settings: NeuroGolfSettings,
    *,
    kind: PromptKind = "solve",
    epoch: int = 1,
    resume: bool = True,
) -> dict[int, object]:
    """Run all mapped tasks; ``solve_task`` prepares each workspace before Codex starts."""

    task_numbers = [record.task_num for record in TaskMap(settings)]
    return await solve_many(
        task_numbers,
        settings,
        kind=kind,
        epoch=epoch,
        resume=resume,
    )


def _status_counts(results: Mapping[int, object]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results.values():
        if not isinstance(result, Mapping):
            counts["invalid_result"] += 1
            continue
        counts[f"orchestrator:{result.get('status', 'unknown')}"] += 1
        worker = result.get("worker_result")
        if isinstance(worker, Mapping):
            counts[f"worker:{worker.get('status', 'unknown')}"] += 1
        elif result.get("worker_result_error"):
            counts["worker:invalid"] += 1
    return dict(sorted(counts.items()))


def _successful(result: object, kind: PromptKind) -> bool:
    if not isinstance(result, Mapping) or result.get("status") != "completed":
        return False
    worker = result.get("worker_result")
    expected = {"reviewed", "passed"} if kind == "review" else {"passed"}
    return isinstance(worker, Mapping) and worker.get("status") in expected


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = _load_settings(args.max_parallel)
    task_numbers = [record.task_num for record in TaskMap(settings)]
    plan: dict[str, Any] = {
        "task_count": len(task_numbers),
        "first_task": min(task_numbers),
        "last_task": max(task_numbers),
        "max_parallel": settings.codex.max_parallel_tasks,
        "profile": DEFAULT_CODEX_PROFILE,
        "kind": args.kind,
        "epoch": args.epoch,
        "resume": not args.fresh,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan}, indent=2, sort_keys=True))
        return 0

    kind: PromptKind = args.kind
    results = asyncio.run(
        solve_all(
            settings,
            kind=kind,
            epoch=args.epoch,
            resume=not args.fresh,
        )
    )
    unsuccessful = sorted(
        task_num for task_num, result in results.items() if not _successful(result, kind)
    )
    print(
        json.dumps(
            {
                **plan,
                "status_counts": _status_counts(results),
                "successful_tasks": len(results) - len(unsuccessful),
                "unsuccessful_tasks": unsuccessful,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if unsuccessful else 0


if __name__ == "__main__":
    raise SystemExit(main())
