"""Load the generated ARC-GEN task-number/ID/principle mapping."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from neurogolf.errors import MappingError

_TASK_KEY = re.compile(r"^task\d{3}$")
_TASK_ID = re.compile(r"^[0-9a-f]{8}$")


def load_task_principles(repository_root: Path) -> dict[str, dict[str, str]]:
    path = repository_root / "configs/task_principles.json"
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MappingError(f"Could not load ARC-GEN principle mapping {path}: {error}") from error
    if not isinstance(payload, dict):
        raise MappingError(f"ARC-GEN principle mapping must be an object: {path}")
    result: dict[str, dict[str, str]] = {}
    for task_key, item in payload.items():
        if not isinstance(task_key, str) or not _TASK_KEY.fullmatch(task_key):
            raise MappingError(f"Invalid principle-map task key: {task_key!r}")
        if not isinstance(item, dict):
            raise MappingError(f"Principle-map entry {task_key} must be an object")
        task_id = item.get("task_id")
        principle = item.get("principle")
        if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
            raise MappingError(f"Invalid task ID in principle-map entry {task_key}: {task_id!r}")
        if not isinstance(principle, str) or not principle.strip():
            raise MappingError(f"Missing principle in principle-map entry {task_key}")
        if set(item) != {"task_id", "principle"}:
            raise MappingError(f"Unexpected fields in principle-map entry {task_key}")
        result[task_key] = {"task_id": task_id, "principle": principle.strip()}
    return result


def task_principle(repository_root: Path, task_num: int, task_id: str) -> str:
    task_key = f"task{task_num:03d}"
    entry = load_task_principles(repository_root).get(task_key)
    if entry is None:
        raise MappingError(f"No ARC-GEN principle is mapped for {task_key}/{task_id}")
    if entry["task_id"] != task_id:
        raise MappingError(
            f"ARC-GEN principle mapping mismatch for {task_key}: "
            f"expected {task_id}, found {entry['task_id']}"
        )
    return entry["principle"]
