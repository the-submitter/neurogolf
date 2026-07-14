"""Diverse, concise, schema-valid candidate failure packets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from neurogolf.judge.candidate_runner import ExampleComparison
from neurogolf.judge.reports import validate_report
from neurogolf.tasks.models import ArcExample

Failure = tuple[ArcExample, ExampleComparison]


def failure_signature(comparison: ExampleComparison) -> str:
    if comparison.runtime_error:
        return f"runtime:{comparison.runtime_error.split(':', 1)[0]}"
    if comparison.shape_difference:
        return f"shape:{comparison.shape_difference}"
    if comparison.special_output_colors:
        colors = sorted({item["color"] for item in comparison.special_output_colors})
        return f"onehot:{colors}"
    changed = Counter((item.expected, item.actual) for item in comparison.pixel_differences[:200])
    return "pixels:" + hashlib.sha256(repr(changed.most_common(5)).encode()).hexdigest()[:12]


def select_diverse(failures: list[Failure], limit: int) -> list[Failure]:
    selected: list[Failure] = []
    used_indices: set[int] = set()
    seen_signatures: set[str] = set()
    for index, failure in enumerate(failures):
        signature = failure_signature(failure[1])
        if signature not in seen_signatures:
            selected.append(failure)
            used_indices.add(index)
            seen_signatures.add(signature)
            if len(selected) == limit:
                return selected
    for index, failure in enumerate(failures):
        if index not in used_indices:
            selected.append(failure)
            if len(selected) == limit:
                break
    return selected


def build_failure_packet(
    *,
    candidate_sha256: str,
    judge_version: str,
    task_num: int,
    task_id: str,
    failures: list[Failure],
    scorer_diagnostics: dict[str, Any],
    max_examples: int,
) -> dict[str, Any]:
    families = Counter(comparison.family or "unknown" for _, comparison in failures)
    signatures = Counter(failure_signature(comparison) for _, comparison in failures)
    representatives: list[dict[str, Any]] = []
    for example, comparison in select_diverse(failures, max_examples):
        value = comparison.to_dict()
        value["input"] = example.input
        value["failure_signature"] = failure_signature(comparison)
        value["pixel_difference_count"] = len(comparison.pixel_differences)
        representatives.append(value)
    packet = {
        "schema_version": "1.0",
        "candidate_sha256": candidate_sha256,
        "judge_version": judge_version,
        "task_num": task_num,
        "task_id": task_id,
        "failure_count": len(failures),
        "failure_families": dict(sorted(families.items())),
        "failure_signatures": dict(signatures.most_common()),
        "representative_failures": representatives,
        "scorer_diagnostics": scorer_diagnostics,
    }
    validate_report(packet, "failure_packet.schema.json")
    return packet


def write_failure_packet(packet: dict[str, Any], path: Path) -> Path:
    validate_report(packet, "failure_packet.schema.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
