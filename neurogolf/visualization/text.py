"""Terminal-first numeric, symbolic, ANSI, indexed, and side-by-side rendering."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from neurogolf.tasks.models import ArcExample, Grid

SYMBOLS = {
    0: ".",
    1: "B",
    2: "R",
    3: "G",
    4: "Y",
    5: "H",
    6: "M",
    7: "O",
    8: "C",
    9: "W",
    10: "-",
    11: "!",
}
ANSI = {
    0: "\x1b[30m",
    1: "\x1b[94m",
    2: "\x1b[91m",
    3: "\x1b[92m",
    4: "\x1b[93m",
    5: "\x1b[90m",
    6: "\x1b[95m",
    7: "\x1b[38;5;208m",
    8: "\x1b[96m",
    9: "\x1b[97m",
    10: "\x1b[37m",
    11: "\x1b[41m",
}


def grid_bbox(grid: Grid, *, background: int = 0) -> tuple[int, int, int, int] | None:
    cells = [
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value != background
    ]
    if not cells:
        return None
    rows, cols = zip(*cells, strict=True)
    return min(rows), min(cols), max(rows), max(cols)


def render_grid(
    grid: Grid,
    *,
    mode: Literal["numeric", "symbol", "ansi"] = "symbol",
    indices: bool = False,
    annotate_bbox: bool = False,
    changed: set[tuple[int, int]] | None = None,
    markdown_safe: bool = True,
) -> str:
    # Expected grids are rectangular, but official malformed-output diagnostics
    # can legitimately contain no rows or rows trimmed to different widths.
    if not grid:
        return "<empty>"
    normalized = [list(row) for row in grid]
    changed = changed or set()
    width = max((len(row) for row in normalized), default=0)
    cell_width = max(1, len(str(width - 1))) if mode == "numeric" else 1
    lines: list[str] = []
    if indices:
        prefix = " " * (len(str(len(normalized) - 1)) + 1)
        lines.append(prefix + " ".join(f"{col:>{cell_width}}" for col in range(width)))
    for row_index, row in enumerate(normalized):
        cells: list[str] = []
        for col_index, color in enumerate(row):
            token = str(color) if mode == "numeric" else SYMBOLS.get(color, "?")
            token = f"{token:>{cell_width}}"
            if mode == "ansi":
                token = f"{ANSI.get(color, '')}{token}\x1b[0m"
            elif (row_index, col_index) in changed:
                token = f"[{token}]"
            cells.append(token)
        prefix = f"{row_index:>{len(str(len(normalized) - 1))}} " if indices else ""
        lines.append(prefix + (" ".join(cells) if cells else "<empty-row>"))
    if annotate_bbox:
        bbox = grid_bbox(normalized)
        lines.append(f"bbox(nonzero): {bbox if bbox is not None else 'empty'}")
    rendered = "\n".join(lines)
    if markdown_safe:
        rendered = rendered.replace("|", "\\|")
    return rendered


def side_by_side(blocks: Iterable[tuple[str, str]], *, gap: str = "   ") -> str:
    values = [(label, content.splitlines()) for label, content in blocks]
    widths = [
        max([len(label), *(len(line) for line in lines)], default=len(label))
        for label, lines in values
    ]
    height = max((len(lines) for _, lines in values), default=0)
    output = [
        gap.join(label.ljust(width) for (label, _), width in zip(values, widths, strict=True))
    ]
    output.append(gap.join("-" * width for width in widths))
    for index in range(height):
        output.append(
            gap.join(
                (lines[index] if index < len(lines) else "").ljust(width)
                for (_, lines), width in zip(values, widths, strict=True)
            )
        )
    return "\n".join(output)


def render_pair(
    example: ArcExample,
    *,
    mode: Literal["numeric", "symbol", "ansi"] = "symbol",
    indices: bool = False,
) -> str:
    return side_by_side(
        [
            ("INPUT", render_grid(example.input, mode=mode, indices=indices)),
            ("OUTPUT", render_grid(example.output, mode=mode, indices=indices)),
        ]
    )


def render_examples(
    examples: Iterable[ArcExample], *, mode: Literal["numeric", "symbol", "ansi"] = "symbol"
) -> str:
    return "\n\n".join(
        f"Example {index}\n{render_pair(example, mode=mode)}"
        for index, example in enumerate(examples, start=1)
    )
