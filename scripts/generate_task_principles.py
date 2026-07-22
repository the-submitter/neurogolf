#!/usr/bin/env python3
"""Regenerate utils/task_principles.json from ARC-GEN task-list comments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ENTRY = re.compile(
    r'^\s*"(?P<task_id>[0-9a-f]{8})"\s*:\s*\[[^\]]+\]\s*,?\s*#\s*(?P<principle>\S.*?)\s*$'
)


def extract_principles(task_list_path: Path) -> dict[str, str]:
    source = task_list_path.read_text(encoding="utf-8")
    marker = "def task_list():"
    if marker not in source:
        raise ValueError(f"Could not find {marker!r} in {task_list_path}")
    body = source.split(marker, 1)[1]
    principles: dict[str, str] = {}
    for line in body.splitlines():
        match = ENTRY.fullmatch(line)
        if match is None:
            continue
        task_id = match.group("task_id")
        if task_id in principles:
            raise ValueError(f"Duplicate task-list entry for {task_id}")
        principles[task_id] = match.group("principle").strip()
    return principles


def build_mapping(repository_root: Path) -> dict[str, dict[str, str]]:
    canonical: Any = json.loads(
        (repository_root / "utils/task_map.json").read_text(encoding="utf-8")
    )
    if not isinstance(canonical, dict):
        raise ValueError("utils/task_map.json must contain an object")
    principles = extract_principles(repository_root / "ARC-GEN/task_list.py")
    result: dict[str, dict[str, str]] = {}
    for task_key, task_id in canonical.items():
        if not isinstance(task_key, str) or not isinstance(task_id, str):
            raise ValueError("utils/task_map.json contains a non-string mapping")
        principle = principles.get(task_id)
        if principle is None:
            raise ValueError(
                f"ARC-GEN task_list.py has no principle comment for {task_key}/{task_id}"
            )
        result[task_key] = {"task_id": task_id, "principle": principle}
    extra = sorted(set(principles) - set(canonical.values()))
    if extra:
        raise ValueError(f"ARC-GEN task_list.py contains unmapped task IDs: {extra}")
    return result


def serialized(mapping: dict[str, dict[str, str]]) -> str:
    return json.dumps(mapping, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the generated file is stale")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    destination = root / "utils/task_principles.json"
    content = serialized(build_mapping(root))
    if args.check:
        if not destination.is_file() or destination.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Generated principle mapping is stale: {destination}")
        print(f"validated {destination}")
        return 0
    destination.write_text(content, encoding="utf-8")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
