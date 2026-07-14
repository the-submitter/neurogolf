"""Expected/actual/difference text and image-friendly structures."""

from __future__ import annotations

from neurogolf.tasks.models import Grid
from neurogolf.visualization.text import render_grid, side_by_side


def difference_coordinates(expected: Grid, actual: Grid) -> list[dict[str, int | None]]:
    height = max(len(expected), len(actual))
    width = max(
        max((len(row) for row in expected), default=0),
        max((len(row) for row in actual), default=0),
    )
    result: list[dict[str, int | None]] = []
    for row in range(height):
        for col in range(width):
            left = expected[row][col] if row < len(expected) and col < len(expected[row]) else None
            right = actual[row][col] if row < len(actual) and col < len(actual[row]) else None
            if left != right:
                result.append({"row": row, "col": col, "expected": left, "actual": right})
    return result


def render_diff(expected: Grid, actual: Grid) -> str:
    coordinates = difference_coordinates(expected, actual)
    changed: set[tuple[int, int]] = set()
    for item in coordinates:
        row, col = item["row"], item["col"]
        if isinstance(row, int) and isinstance(col, int):
            changed.add((row, col))
    marker_height = max(len(expected), len(actual))
    marker_width = max(
        max((len(row) for row in expected), default=0),
        max((len(row) for row in actual), default=0),
    )
    marker = [
        [2 if (row, col) in changed else 0 for col in range(marker_width)]
        for row in range(marker_height)
    ]
    return side_by_side(
        [
            ("EXPECTED", render_grid(expected)),
            ("ACTUAL", render_grid(actual)),
            ("DIFF(R)", render_grid(marker)),
        ]
    )
