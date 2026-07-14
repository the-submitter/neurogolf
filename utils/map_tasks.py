"""Map NeuroGolf task numbers to their original ARC-AGI-1 task IDs.

The NeuroGolf training examples may contain only a subset of the examples in
the corresponding ARC-AGI-1 task.  This script therefore matches complete
input/output example pairs without relying on their order.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bidict import ValueDuplicationError, bidict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KAGGLE_DIR = REPOSITORY_ROOT / "data"
DEFAULT_ARC_DIR = REPOSITORY_ROOT / "ARC-GEN" / "external" / "ARC-AGI" / "data"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "configs" / "task_map.json"
TASK_FILE_PATTERN = re.compile(r"task(\d+)\.json$")


def _load_train(path: Path) -> list[dict[str, Any]]:
    try:
        task = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read task JSON {path}: {error}") from error

    train = task.get("train")
    if not isinstance(train, list) or not train:
        raise ValueError(f"{path} does not contain a non-empty 'train' list")
    for index, example in enumerate(train):
        if not isinstance(example, dict) or "input" not in example or "output" not in example:
            raise ValueError(f"{path} train example {index} must contain 'input' and 'output'")
    return train


def _signature(example: dict[str, Any]) -> str:
    """Return a stable, hashable representation of one complete example pair."""
    return json.dumps(
        {"input": example["input"], "output": example["output"]},
        sort_keys=True,
        separators=(",", ":"),
    )


def build_mapping(kaggle_dir: Path, arc_dir: Path) -> bidict[str, str]:
    """Build and validate a one-to-one NeuroGolf/ARC-AGI task mapping."""
    kaggle_files = sorted(
        (path for path in kaggle_dir.glob("task*.json") if TASK_FILE_PATTERN.match(path.name)),
        key=lambda path: int(TASK_FILE_PATTERN.match(path.name).group(1)),  # type: ignore[union-attr]
    )
    arc_files = sorted(arc_dir.rglob("*.json"))
    if not kaggle_files:
        raise ValueError(f"No taskNNN.json files found in {kaggle_dir}")
    if not arc_files:
        raise ValueError(f"No ARC-AGI JSON files found in {arc_dir}")

    arc_examples: dict[str, Counter[str]] = {}
    example_index: dict[str, set[str]] = defaultdict(set)
    for path in arc_files:
        arc_id = path.stem
        signatures = Counter(_signature(example) for example in _load_train(path))
        arc_examples[arc_id] = signatures
        for signature in signatures:
            example_index[signature].add(arc_id)

    mapping: bidict[str, str] = bidict()
    for path in kaggle_files:
        task_number = path.stem
        subset = Counter(_signature(example) for example in _load_train(path))
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
        try:
            mapping[task_number] = arc_id
        except ValueDuplicationError as error:
            raise ValueError(
                f"ARC-AGI task {arc_id} matches both {mapping.inverse[arc_id]} and {task_number}"
            ) from error

    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle-dir", type=Path, default=DEFAULT_KAGGLE_DIR)
    parser.add_argument("--arc-dir", type=Path, default=DEFAULT_ARC_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = build_mapping(args.kaggle_dir, args.arc_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Persist a regular JSON object.  Consumers can wrap the loaded dict in
    # bidict when reverse lookup is needed.
    args.output.write_text(json.dumps(dict(mapping), indent=2) + "\n", encoding="utf-8")
    count = len(mapping)
    print(f"Mapped {count} tasks; wrote {args.output}")


if __name__ == "__main__":
    main()
