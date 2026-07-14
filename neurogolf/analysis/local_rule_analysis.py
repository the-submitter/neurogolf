"""Local neighborhood change tables and bounded-receptive-field plausibility."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from neurogolf.tasks.models import ArcExample, Grid


def _patch(grid: Grid, row: int, col: int, radius: int) -> tuple[int, ...]:
    values: list[int] = []
    for row_delta in range(-radius, radius + 1):
        for col_delta in range(-radius, radius + 1):
            r, c = row + row_delta, col + col_delta
            values.append(grid[r][c] if 0 <= r < len(grid) and 0 <= c < len(grid[0]) else -1)
    return tuple(values)


def receptive_field_plausibility(
    examples: list[ArcExample], *, max_radius: int = 2
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    aligned = [
        example
        for example in examples
        if len(example.input) == len(example.output)
        and len(example.input[0]) == len(example.output[0])
    ]
    for radius in range(max_radius + 1):
        mapping: dict[tuple[int, ...], set[int]] = defaultdict(set)
        observations = 0
        for example in aligned:
            for row in range(len(example.input)):
                for col in range(len(example.input[0])):
                    mapping[_patch(example.input, row, col, radius)].add(example.output[row][col])
                    observations += 1
        conflicts = sum(len(outputs) > 1 for outputs in mapping.values())
        results.append(
            {
                "radius": radius,
                "aligned_examples": len(aligned),
                "observations": observations,
                "unique_patches": len(mapping),
                "conflicting_patches": conflicts,
                "plausible": bool(aligned) and conflicts == 0,
            }
        )
    return results


def local_change_table(examples: list[ArcExample], *, limit: int = 50) -> list[dict[str, int]]:
    transitions: Counter[tuple[int, int, int]] = Counter()
    for example in examples:
        if len(example.input) != len(example.output) or len(example.input[0]) != len(
            example.output[0]
        ):
            continue
        for row in range(len(example.input)):
            for col in range(len(example.input[0])):
                nonzero_neighbors = sum(
                    value > 0 for value in _patch(example.input, row, col, 1)[0:9]
                )
                transitions[
                    (example.input[row][col], example.output[row][col], nonzero_neighbors)
                ] += 1
    return [
        {"input_color": key[0], "output_color": key[1], "nonzero_3x3": key[2], "count": count}
        for key, count in transitions.most_common(limit)
    ]
