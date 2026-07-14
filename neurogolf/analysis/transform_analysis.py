"""D4 transform matches and bounded fixed-translation rankings."""

from __future__ import annotations

from typing import Any

import numpy as np

from neurogolf.tasks.models import ArcExample, Grid


def d4_transforms(grid: Grid) -> dict[str, Grid]:
    array = np.asarray(grid, dtype=np.int8)
    transforms = {
        "identity": array,
        "rotate90": np.rot90(array, 1),
        "rotate180": np.rot90(array, 2),
        "rotate270": np.rot90(array, 3),
        "flip_horizontal": np.fliplr(array),
        "flip_vertical": np.flipud(array),
        "transpose": array.T,
        "anti_transpose": np.fliplr(np.flipud(array)).T,
    }
    return {name: value.tolist() for name, value in transforms.items()}


def best_translations(
    source: Grid, target: Grid, *, radius: int = 5, limit: int = 5
) -> list[dict[str, Any]]:
    target_height, target_width = len(target), len(target[0])
    source_height, source_width = len(source), len(source[0])
    rankings: list[dict[str, Any]] = []
    for row_delta in range(-radius, radius + 1):
        for col_delta in range(-radius, radius + 1):
            matches = compared = 0
            for row in range(target_height):
                for col in range(target_width):
                    source_row, source_col = row - row_delta, col - col_delta
                    if 0 <= source_row < source_height and 0 <= source_col < source_width:
                        compared += 1
                        matches += source[source_row][source_col] == target[row][col]
            rankings.append(
                {
                    "row_delta": row_delta,
                    "col_delta": col_delta,
                    "matches": matches,
                    "compared": compared,
                    "agreement": matches / compared if compared else 0.0,
                }
            )
    return sorted(
        rankings,
        key=lambda item: (
            -item["agreement"],
            -item["matches"],
            abs(item["row_delta"]) + abs(item["col_delta"]),
        ),
    )[:limit]


def analyze_transforms(example: ArcExample) -> dict[str, Any]:
    matches = [
        name for name, grid in d4_transforms(example.input).items() if grid == example.output
    ]
    return {
        "d4_matches": matches,
        "best_translations": best_translations(example.input, example.output),
    }
