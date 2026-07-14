"""Color sets, transitions, and aligned pixel-change statistics."""

from __future__ import annotations

from collections import Counter
from typing import Any

from neurogolf.tasks.models import ArcExample, Grid


def color_histogram(grid: Grid) -> dict[str, int]:
    return {
        str(color): count
        for color, count in sorted(Counter(cell for row in grid for cell in row).items())
    }


def analyze_colors(example: ArcExample) -> dict[str, Any]:
    input_colors = {cell for row in example.input for cell in row}
    output_colors = {cell for row in example.output for cell in row}
    transitions: Counter[tuple[int, int]] = Counter()
    changed = 0
    comparable = len(example.input) == len(example.output) and len(example.input[0]) == len(
        example.output[0]
    )
    if comparable:
        for input_row, output_row in zip(example.input, example.output, strict=True):
            for source, target in zip(input_row, output_row, strict=True):
                transitions[(source, target)] += 1
                changed += source != target
    return {
        "input_colors": sorted(input_colors),
        "output_colors": sorted(output_colors),
        "added_colors": sorted(output_colors - input_colors),
        "removed_colors": sorted(input_colors - output_colors),
        "preserved_colors": sorted(input_colors & output_colors),
        "input_histogram": color_histogram(example.input),
        "output_histogram": color_histogram(example.output),
        "aligned": comparable,
        "changed_pixels": changed if comparable else None,
        "transition_matrix": {
            f"{source}->{target}": count for (source, target), count in sorted(transitions.items())
        },
    }
