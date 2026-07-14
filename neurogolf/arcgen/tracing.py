"""Lightweight timing and source tracing for generator calls."""

from __future__ import annotations

import ast
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CallTrace:
    duration_seconds: float
    result_type: str
    exception_type: str | None = None


def traced_call(function: Callable[[], Any]) -> tuple[Any, CallTrace]:
    started = time.perf_counter()
    try:
        result = function()
    except Exception as error:
        duration = time.perf_counter() - started
        error.add_note(f"ARC-GEN call failed after {duration:.6f}s")
        raise
    return result, CallTrace(time.perf_counter() - started, type(result).__name__)


def source_ast_summary(path: Path) -> dict[str, Any]:
    """Return bounded structural metadata without executing generator source."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    calls: dict[str, int] = {}
    assignments = 0
    loops = 0
    conditions = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = "<dynamic>"
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            calls[name] = calls.get(name, 0) + 1
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            assignments += 1
        elif isinstance(node, (ast.For, ast.While, ast.comprehension)):
            loops += 1
        elif isinstance(node, (ast.If, ast.IfExp)):
            conditions += 1
    return {
        "line_count": len(source.splitlines()),
        "functions": [function.name for function in functions],
        "calls": dict(sorted(calls.items(), key=lambda item: (-item[1], item[0]))[:40]),
        "assignment_count": assignments,
        "loop_count": loops,
        "condition_count": conditions,
    }
