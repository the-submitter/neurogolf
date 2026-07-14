"""Generator validation reports and canonical ARC-AGI parity checks."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from neurogolf.arcgen.generator import GeneratorOracle
from neurogolf.tasks.loader import load_arc_agi_dataset


def _signature(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def validate_oracle(
    oracle: GeneratorOracle, *, sample_count: int = 10, seed: int = 0
) -> dict[str, Any]:
    validated = oracle.validate_generator()
    canonical = load_arc_agi_dataset(oracle.record.arc_agi_json_path)
    split_matches: dict[str, bool] = {}
    for split in ("train", "test"):
        left = Counter(_signature(item) for item in getattr(validated, split))
        right = Counter(_signature(item) for item in getattr(canonical, split))
        split_matches[split] = left == right
    samples = [oracle.generate(seed + index, validate=False) for index in range(sample_count)]
    return {
        "valid": all(split_matches.values()) and len(samples) == sample_count,
        "task_num": oracle.record.task_num,
        "task_id": oracle.record.task_id,
        "validate_matches_arc_agi": split_matches,
        "validated_counts": {"train": len(validated.train), "test": len(validated.test)},
        "generated_count": len(samples),
        "seeds": [sample.seed for sample in samples],
    }
