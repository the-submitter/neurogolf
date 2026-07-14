"""Four/eight-connected component summaries by color."""

from __future__ import annotations

from collections import deque
from typing import Any

from neurogolf.tasks.models import Grid


def connected_components(
    grid: Grid, *, diagonal: bool = False, include_background: bool = False
) -> list[dict[str, Any]]:
    height, width = len(grid), len(grid[0])
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonal:
        neighbors += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    seen: set[tuple[int, int]] = set()
    result: list[dict[str, Any]] = []
    for row in range(height):
        for col in range(width):
            if (row, col) in seen or (grid[row][col] == 0 and not include_background):
                continue
            color = grid[row][col]
            queue = deque([(row, col)])
            seen.add((row, col))
            cells: list[tuple[int, int]] = []
            while queue:
                current = queue.popleft()
                cells.append(current)
                for row_delta, col_delta in neighbors:
                    candidate = (current[0] + row_delta, current[1] + col_delta)
                    if (
                        0 <= candidate[0] < height
                        and 0 <= candidate[1] < width
                        and candidate not in seen
                        and grid[candidate[0]][candidate[1]] == color
                    ):
                        seen.add(candidate)
                        queue.append(candidate)
            rows, cols = zip(*cells, strict=True)
            result.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": [min(rows), min(cols), max(rows), max(cols)],
                    "touches_edge": any(
                        r in {0, height - 1} or c in {0, width - 1} for r, c in cells
                    ),
                }
            )
    return sorted(result, key=lambda item: (item["color"], -item["size"], item["bbox"]))


def component_summary(grid: Grid) -> dict[str, Any]:
    four = connected_components(grid)
    eight = connected_components(grid, diagonal=True)
    return {
        "four_connected": four,
        "eight_connected": eight,
        "four_count": len(four),
        "eight_count": len(eight),
    }
