#!/usr/bin/env python3
"""Regenerate utils/task_map.json from NeuroGolf and ARC-AGI examples.

The NeuroGolf public examples may contain only a subset of the examples in the
corresponding ARC-AGI-1 task. This script therefore matches complete train and
test input/output pairs without relying on their order. Generated ``arc-gen``
examples are deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KAGGLE_DIR = REPOSITORY_ROOT / "kaggle_tasks_data"
DEFAULT_ARC_DIR = REPOSITORY_ROOT / "ARC-GEN" / "external" / "ARC-AGI" / "data"
DEFAULT_GENERATORS_DIR = REPOSITORY_ROOT / "ARC-GEN" / "tasks"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "utils" / "task_map.json"
TASK_FILE_PATTERN = re.compile(r"task(\d+)\.json$")


def _load_public_examples(path: Path) -> list[dict[str, Any]]:
    try:
        task = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read task JSON {path}: {error}") from error

    examples: list[dict[str, Any]] = []
    for section in ("train", "test"):
        section_examples = task.get(section)
        if not isinstance(section_examples, list):
            raise ValueError(f"{path} does not contain a {section!r} list")
        for index, example in enumerate(section_examples):
            if (
                not isinstance(example, dict)
                or "input" not in example
                or "output" not in example
            ):
                raise ValueError(
                    f"{path} {section} example {index} must contain 'input' and 'output'"
                )
            examples.append(example)
    if not examples:
        raise ValueError(f"{path} does not contain any public examples")
    return examples


def _signature(example: dict[str, Any]) -> str:
    """Return a stable, hashable representation of one complete example pair."""
    return json.dumps(
        {"input": example["input"], "output": example["output"]},
        sort_keys=True,
        separators=(",", ":"),
    )


def build_mapping(
    kaggle_dir: Path, arc_dir: Path, generators_dir: Path
) -> dict[str, str]:
    """Build and validate a one-to-one NeuroGolf/ARC-AGI task mapping."""
    kaggle_files = sorted(
        (path for path in kaggle_dir.glob("task*.json") if TASK_FILE_PATTERN.match(path.name)),
        key=lambda path: int(TASK_FILE_PATTERN.match(path.name).group(1)),  # type: ignore[union-attr]
    )
    generator_ids = {
        path.stem.removeprefix("task_")
        for path in generators_dir.glob("task_*.py")
    }
    arc_files = sorted(
        path for path in arc_dir.rglob("*.json") if path.stem in generator_ids
    )
    if not kaggle_files:
        raise ValueError(f"No taskNNN.json files found in {kaggle_dir}")
    if not arc_files:
        raise ValueError(
            f"No ARC-AGI JSON files in {arc_dir} have generators in {generators_dir}"
        )

    arc_examples: dict[str, Counter[str]] = {}
    example_index: dict[str, set[str]] = defaultdict(set)
    for path in arc_files:
        arc_id = path.stem
        signatures = Counter(
            _signature(example) for example in _load_public_examples(path)
        )
        arc_examples[arc_id] = signatures
        for signature in signatures:
            example_index[signature].add(arc_id)

    mapping: dict[str, str] = {}
    reverse_mapping: dict[str, str] = {}
    for path in kaggle_files:
        task_number = path.stem
        subset = Counter(_signature(example) for example in _load_public_examples(path))
        candidate_sets = [example_index.get(signature, set()) for signature in subset]
        candidates = set.intersection(*candidate_sets) if candidate_sets else set()
        matches = sorted(
            arc_id
            for arc_id in candidates
            if all(count <= arc_examples[arc_id][signature] for signature, count in subset.items())
        )

        if not matches:
            raise ValueError(f"No ARC-AGI task matches the training examples in {path}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous match for {path}: {', '.join(matches)}")

        arc_id = matches[0]
        duplicate = reverse_mapping.get(arc_id)
        if duplicate is not None:
            raise ValueError(
                f"ARC-AGI task {arc_id} matches both {duplicate} and {task_number}"
            )
        mapping[task_number] = arc_id
        reverse_mapping[arc_id] = task_number

    return mapping


def serialized(mapping: dict[str, str]) -> str:
    return json.dumps(mapping, indent=2, sort_keys=True) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle-dir", type=Path, default=DEFAULT_KAGGLE_DIR)
    parser.add_argument("--arc-dir", type=Path, default=DEFAULT_ARC_DIR)
    parser.add_argument(
        "--generators-dir", type=Path, default=DEFAULT_GENERATORS_DIR
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail if the output is stale")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kaggle_dir = args.kaggle_dir.expanduser().resolve()
    arc_dir = args.arc_dir.expanduser().resolve()
    generators_dir = args.generators_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    mapping = build_mapping(kaggle_dir, arc_dir, generators_dir)
    content = serialized(mapping)
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Generated task mapping is stale: {output}")
        print(f"validated {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    count = len(mapping)
    print(f"mapped {count} tasks; wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
