"""Shapes, histograms, bounding boxes, periodicity, and symmetry indicators."""

from __future__ import annotations

from collections import Counter
from typing import Any

from neurogolf.tasks.models import Grid


def bounding_boxes(grid: Grid) -> dict[str, list[int]]:
    colors = sorted({cell for row in grid for cell in row})
    result: dict[str, list[int]] = {}
    for color in colors:
        cells = [
            (row, col)
            for row, values in enumerate(grid)
            for col, value in enumerate(values)
            if value == color
        ]
        rows, cols = zip(*cells, strict=True)
        result[str(color)] = [min(rows), min(cols), max(rows), max(cols)]
    return result


def _period(values: list[tuple[int, ...]]) -> int | None:
    for period in range(1, len(values)):
        if all(values[index] == values[index % period] for index in range(len(values))):
            return period
    return None


def analyze_grid_spatial(grid: Grid) -> dict[str, Any]:
    height, width = len(grid), len(grid[0])
    rows = [tuple(row) for row in grid]
    cols = [tuple(grid[row][col] for row in range(height)) for col in range(width)]
    row_histograms = [dict(sorted(Counter(row).items())) for row in grid]
    column_histograms = [dict(sorted(Counter(column).items())) for column in cols]
    return {
        "shape": [height, width],
        "row_histograms": row_histograms,
        "column_histograms": column_histograms,
        "bounding_boxes": bounding_boxes(grid),
        "row_period": _period(rows),
        "column_period": _period(cols),
        "symmetry": {
            "horizontal": all(row == row[::-1] for row in rows),
            "vertical": rows == rows[::-1],
            "main_diagonal": height == width
            and all(grid[r][c] == grid[c][r] for r in range(height) for c in range(width)),
            "anti_diagonal": height == width
            and all(
                grid[r][c] == grid[height - 1 - c][width - 1 - r]
                for r in range(height)
                for c in range(width)
            ),
        },
    }
