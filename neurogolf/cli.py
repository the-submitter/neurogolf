"""Typer command surface for all deterministic NeuroGolf infrastructure."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Literal

import onnx
import typer

from neurogolf import __version__
from neurogolf.arcgen.generator import GeneratorOracle
from neurogolf.arcgen.sampling import sample_many
from neurogolf.arcgen.validator import validate_oracle
from neurogolf.config import load_settings
from neurogolf.errors import NeuroGolfError
from neurogolf.judge.candidate_runner import CandidateRunner
from neurogolf.judge.gate import AcceptanceGate
from neurogolf.judge.legality import inspect_legality
from neurogolf.judge.official_adapter import OfficialAdapter
from neurogolf.judge.promotion import promote_candidate, rollback_champion
from neurogolf.judge.registry import ChampionRegistry
from neurogolf.logging import configure_logging
from neurogolf.onnx_lab.inspect import inspect_model
from neurogolf.onnx_lab.optimize import optimize_model
from neurogolf.onnx_lab.templates import identity_model
from neurogolf.orchestrator.prepare import prepare_all, prepare_task
from neurogolf.orchestrator.promotion import auto_promote_gate_result
from neurogolf.orchestrator.solve import run_solve_task, solve_from_status, solve_many
from neurogolf.orchestrator.status import read_status
from neurogolf.tasks.integrity import verify_integrity, write_manifest
from neurogolf.tasks.loader import load_kaggle_dataset
from neurogolf.tasks.mapping import TaskMap
from neurogolf.tasks.principles import load_task_principles, task_principle

app = typer.Typer(help="Codex-first NeuroGolf infrastructure", no_args_is_help=True)
tasks_app = typer.Typer(help="Inspect and validate canonical task mappings")
task_app = typer.Typer(help="Prepare task workspaces")
arcgen_app = typer.Typer(help="Invoke canonical ARC-GEN generators")
scorer_app = typer.Typer(help="Run immutable official scorer adapters")
onnx_app = typer.Typer(help="Inspect and conservatively optimize ONNX models")
gate_app = typer.Typer(help="Run the protected gate and auto-promote eligible exact hashes")
champions_app = typer.Typer(help="Inspect and explicitly change champion state")
solve_app = typer.Typer(help="Launch isolated Codex task epochs")
integrity_app = typer.Typer(help="Verify or intentionally refresh protected hashes")
app.add_typer(tasks_app, name="tasks")
app.add_typer(task_app, name="task")
app.add_typer(arcgen_app, name="arcgen")
app.add_typer(scorer_app, name="scorer")
app.add_typer(onnx_app, name="onnx")
app.add_typer(gate_app, name="gate")
app.add_typer(champions_app, name="champions")
app.add_typer(solve_app, name="solve")
app.add_typer(integrity_app, name="integrity")


@app.callback()
def root(
    debug: bool = typer.Option(False, "--debug", help="Show debug logging and tracebacks"),
    json_logs: bool = typer.Option(False, "--json-logs", help="Render logs as JSONL"),
) -> None:
    configure_logging(debug=debug, json_output=json_logs)


def _json(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


@app.command()
def doctor() -> None:
    """Verify dependencies, paths, protected hashes, mapping, and ONNX Runtime."""

    settings = load_settings()
    packages = [
        "numpy",
        "onnx",
        "onnxruntime",
        "onnx-tool",
        "pydantic",
        "pydantic-settings",
        "PyYAML",
        "jsonschema",
        "networkx",
        "sympy",
        "matplotlib",
        "pillow",
        "rich",
        "typer",
        "pytest",
        "hypothesis",
        "ruff",
        "pyright",
    ]
    versions: dict[str, str] = {}
    missing: list[str] = []
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    path_checks = {
        "arcgen_root": settings.paths.arcgen_root.is_dir(),
        "kaggle_data_root": settings.paths.kaggle_data_root.is_dir(),
        "task_map": settings.paths.task_map.is_file(),
        "official_utils": settings.paths.official_utils.is_file(),
        "sealed_manifest": settings.paths.sealed_manifest.is_file(),
    }
    integrity = verify_integrity(settings)
    mapping = TaskMap(settings)
    principles = load_task_principles(settings.paths.repository_root)
    principles_valid = len(principles) == len(mapping) and all(
        principles.get(f"task{record.task_num:03d}", {}).get("task_id") == record.task_id
        for record in mapping
    )
    adapter = OfficialAdapter(settings)
    model = adapter.sanitize(identity_model())
    adapter.create_session(model)
    report = {
        "ok": not missing
        and all(path_checks.values())
        and integrity["valid"]
        and len(mapping) == 400
        and principles_valid,
        "neurogolf_version": __version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": versions,
        "missing_packages": missing,
        "paths": path_checks,
        "integrity": integrity,
        "mapped_tasks": len(mapping),
        "mapped_task_principles": len(principles),
        "task_principles_valid": principles_valid,
        "onnx_runtime_smoke": "passed",
    }
    _json(report)
    if not report["ok"]:
        raise typer.Exit(code=1)


@tasks_app.command("validate-map")
def validate_map(
    shallow: bool = typer.Option(False, help="Skip importing/validating all generators"),
    output: Path | None = typer.Option(None, help="Optional JSON report path"),
) -> None:
    report = TaskMap(load_settings()).validate(deep=not shallow)
    payload = report.model_dump(mode="json")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _json(payload)
    if not report.valid:
        raise typer.Exit(code=1)


@tasks_app.command("show")
def show_task(task_num: int) -> None:
    settings = load_settings()
    record = TaskMap(settings).get(task_num)
    dataset = load_kaggle_dataset(record.kaggle_json_path)
    payload = record.model_dump(mode="json")
    payload["arcgen_principle"] = task_principle(
        settings.paths.repository_root, record.task_num, record.task_id
    )
    payload["counts"] = {
        "train": len(dataset.train),
        "test": len(dataset.test),
        "arc-gen": len(dataset.arc_gen),
    }
    _json(payload)


@tasks_app.command("list")
def list_tasks(json_output: bool = typer.Option(False, "--json")) -> None:
    settings = load_settings()
    records = list(TaskMap(settings))
    principles = load_task_principles(settings.paths.repository_root)
    if json_output:
        _json(
            [
                {
                    **record.model_dump(mode="json"),
                    "arcgen_principle": principles[f"task{record.task_num:03d}"]["principle"],
                }
                for record in records
            ]
        )
        return
    for record in records:
        principle = principles[f"task{record.task_num:03d}"]["principle"]
        typer.echo(
            f"{record.task_num:03d}  {record.task_id}  {record.arc_agi_split:<10}  {principle}"
        )


@task_app.command("prepare")
def prepare_one(task_num: int, force: bool = typer.Option(False)) -> None:
    _json(prepare_task(task_num, load_settings(), force=force))


@task_app.command("prepare-all")
def prepare_every_task(force: bool = typer.Option(False)) -> None:
    values = prepare_all(load_settings(), force=force)
    _json({"prepared": len(values), "tasks": values})


@arcgen_app.command("sample")
def arcgen_sample(
    task_num: int,
    count: int = typer.Option(1, min=1),
    seed: int = typer.Option(0),
) -> None:
    settings = load_settings()
    oracle = GeneratorOracle.for_task(task_num, settings)
    values = sample_many(oracle, count=count, base_seed=seed)
    _json([value.model_dump(mode="json") for value in values])


@arcgen_app.command("validate")
def arcgen_validate(
    task_num: int,
    count: int = typer.Option(100, min=1),
    seed: int = typer.Option(0),
) -> None:
    settings = load_settings()
    result = validate_oracle(
        GeneratorOracle.for_task(task_num, settings), sample_count=count, seed=seed
    )
    _json(result)
    if not result["valid"]:
        raise typer.Exit(code=1)


@scorer_app.command("inspect")
def scorer_inspect(model: Path) -> None:
    settings = load_settings()
    adapter = OfficialAdapter(settings)
    legality = inspect_legality(model, adapter)
    payload: dict[str, Any] = {"legality": legality.to_dict()}
    if legality.legal:
        payload["official_score"] = adapter.score_model(model).to_dict()
    _json(payload)
    if not legality.legal:
        raise typer.Exit(code=2)


@scorer_app.command("verify")
def scorer_verify(
    model: Path,
    task: int = typer.Option(..., "--task"),
) -> None:
    settings = load_settings()
    record = TaskMap(settings).get(task)
    dataset = load_kaggle_dataset(record.kaggle_json_path)
    adapter = OfficialAdapter(settings)
    runner = CandidateRunner(model, adapter)
    values = [*dataset.train, *dataset.test, *dataset.arc_gen]
    comparisons = runner.compare_many(values, family="public")
    score = adapter.score_model(model, [example.input for example in values]).to_dict()
    payload = {
        "task_num": task,
        "task_id": record.task_id,
        "passed": sum(item.passed for item in comparisons),
        "failed": sum(not item.passed for item in comparisons),
        "score": score,
        "first_failure": next((item.to_dict() for item in comparisons if not item.passed), None),
    }
    _json(payload)
    if payload["failed"]:
        raise typer.Exit(code=2)


@onnx_app.command("inspect")
def onnx_inspect(
    model: Path,
    cost_breakdown: bool = typer.Option(False, "--cost-breakdown"),
) -> None:
    payload = inspect_model(model)
    if not cost_breakdown and isinstance(payload["local_cost"], dict):
        payload["local_cost"].pop("tensors", None)
    _json(payload)


@onnx_app.command("optimize")
def onnx_optimize(model: Path, output: Path = typer.Option(...)) -> None:
    optimized = optimize_model(onnx.load(model))
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(optimized, output)
    _json({"input": str(model), "output": str(output), "inspection": inspect_model(output)})


@gate_app.command("run")
def gate_run(
    task: int = typer.Option(..., "--task"),
    candidate: Path = typer.Option(..., "--candidate"),
    output: Path = typer.Option(..., "--output"),
    failure_output: Path | None = typer.Option(None),
    fuzz_nonce: int | None = typer.Option(None),
    no_auto_promote: bool = typer.Option(
        False,
        "--no-auto-promote",
        help="Leave an eligible passing candidate unpromoted (used by isolated workers)",
    ),
) -> None:
    settings = load_settings()
    result = AcceptanceGate(settings).run(
        task_num=task,
        candidate=candidate,
        output=output,
        failure_output=failure_output,
        fuzz_nonce=fuzz_nonce,
    )
    payload = result.model_dump(mode="json")
    if settings.gate.auto_promote_eligible and not no_auto_promote:
        payload["auto_promotion"] = auto_promote_gate_result(
            task_num=task,
            candidate=candidate,
            gate_result_path=output,
            settings=settings,
            workspace=settings.paths.task_workspace_root / f"task{task:03d}",
        )
    else:
        payload["auto_promotion"] = {
            "status": "disabled",
            "reason": (
                "configuration" if not settings.gate.auto_promote_eligible else "CLI override"
            ),
        }
    _json(payload)
    if result.status != "passed":
        raise typer.Exit(code=2 if result.status == "candidate_failure" else 1)


@champions_app.command("list")
def champions_list(all_history: bool = typer.Option(False, "--all")) -> None:
    settings = load_settings()
    values = ChampionRegistry(settings.paths.champion_root).list(current_only=not all_history)
    _json([value.to_dict() for value in values])


@champions_app.command("show")
def champions_show(task_num: int) -> None:
    settings = load_settings()
    registry = ChampionRegistry(settings.paths.champion_root)
    current = registry.current(task_num)
    _json(
        {"current": current.to_dict() if current else None, "history": registry.history(task_num)}
    )


@champions_app.command("promote")
def champions_promote(
    task: int = typer.Option(..., "--task"),
    candidate: Path = typer.Option(...),
    gate_result: Path = typer.Option(...),
    builder: Path | None = typer.Option(None),
    explanation: Path | None = typer.Option(None),
    notes: str | None = typer.Option(None),
    allow_score_regression: bool = typer.Option(False),
) -> None:
    result = promote_candidate(
        task_num=task,
        candidate=candidate,
        gate_result_path=gate_result,
        settings=load_settings(),
        builder_path=builder,
        explanation_path=explanation,
        notes=notes,
        allow_score_regression=allow_score_regression,
    )
    _json(result)


@champions_app.command("rollback")
def champions_rollback(
    task: int = typer.Option(..., "--task"),
    sha: str = typer.Option(..., "--sha"),
    notes: str | None = typer.Option(None),
) -> None:
    _json(rollback_champion(task_num=task, sha256=sha, settings=load_settings(), notes=notes))


@integrity_app.command("verify")
def integrity_verify() -> None:
    result = verify_integrity(load_settings())
    _json(result)
    if not result["valid"]:
        raise typer.Exit(code=1)


@integrity_app.command("refresh")
def integrity_refresh(
    reviewed: bool = typer.Option(
        False,
        "--reviewed",
        help="Confirm this intentional protected-file update received review",
    ),
) -> None:
    if not reviewed:
        typer.echo("Refusing to refresh protected hashes without --reviewed", err=True)
        raise typer.Exit(code=2)
    _json(write_manifest(load_settings()))


@solve_app.command("task")
def solve_one(
    task_num: int,
    kind: Literal["solve", "repair", "golf", "review"] = typer.Option("solve"),
    epoch: int = typer.Option(1, min=1),
    fresh: bool = typer.Option(False, help="Start a new session instead of resuming"),
) -> None:
    _json(
        run_solve_task(
            task_num,
            load_settings(),
            kind=kind,
            epoch=epoch,
            resume=not fresh,
        )
    )


@solve_app.command("many")
def solve_selected(
    task_numbers: list[int],
    kind: Literal["solve", "repair", "golf", "review"] = typer.Option("solve"),
    epoch: int = typer.Option(1, min=1),
    fresh: bool = typer.Option(False),
) -> None:
    settings = load_settings()
    _json(
        asyncio.run(
            solve_many(
                task_numbers,
                settings,
                kind=kind,
                epoch=epoch,
                resume=not fresh,
            )
        )
    )


def _status_selected(settings: Any, wanted: str) -> list[int]:
    result: list[int] = []
    registry = ChampionRegistry(settings.paths.champion_root)
    for record in TaskMap(settings):
        status = read_status(
            settings.paths.task_workspace_root
            / f"task{record.task_num:03d}/state/orchestrator.json"
        )
        worker_status = (status.get("worker_result") or {}).get("status") if status else None
        if (
            wanted == "pending"
            and registry.current(record.task_num) is None
            and status is None
            or wanted == "failed"
            and status
            and (
                status.get("status") in {"failed", "timeout", "orchestrator_error"}
                or worker_status in {"failed", "blocked"}
                or status.get("worker_result_error")
            )
        ):
            result.append(record.task_num)
        elif wanted == "golfed":
            gate = (
                settings.paths.task_workspace_root
                / f"task{record.task_num:03d}/state/gate_result.json"
            )
            gate_payload = read_status(gate)
            if gate_payload and gate_payload.get("status") == "passed":
                result.append(record.task_num)
    return result


@solve_app.command("pending")
def solve_pending() -> None:
    settings = load_settings()
    selected = _status_selected(settings, "pending")
    _json(asyncio.run(solve_many(selected, settings, kind="solve")))


@solve_app.command("failed")
def solve_failed() -> None:
    settings = load_settings()
    selected = _status_selected(settings, "failed")
    _json(asyncio.run(solve_from_status(selected, settings, kind="repair", resume=True)))


@solve_app.command("golfed")
def solve_golfed() -> None:
    settings = load_settings()
    selected = _status_selected(settings, "golfed")
    _json(asyncio.run(solve_from_status(selected, settings, kind="golf", resume=False)))


def main() -> None:
    try:
        app()
    except NeuroGolfError as error:
        typer.echo(f"NeuroGolf error: {error}", err=True)
        raise typer.Exit(code=1) from error


if __name__ == "__main__":
    main()
