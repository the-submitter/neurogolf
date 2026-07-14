"""Pillow-based ARC rendering compatible with the official color palette."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from neurogolf.tasks.models import ArcExample, Grid

COLORS = [
    (0, 0, 0),
    (30, 147, 255),
    (250, 61, 49),
    (78, 204, 48),
    (255, 221, 0),
    (153, 153, 153),
    (229, 59, 163),
    (255, 133, 28),
    (136, 216, 241),
    (147, 17, 49),
    (240, 240, 240),
    (146, 117, 86),
]


def _draw_grid(draw: ImageDraw.ImageDraw, grid: Grid, left: int, top: int, cell: int) -> None:
    for row_index, row in enumerate(grid):
        for col_index, color in enumerate(row):
            x0 = left + col_index * cell
            y0 = top + row_index * cell
            draw.rectangle(
                (x0, y0, x0 + cell - 1, y0 + cell - 1),
                fill=COLORS[color] if 0 <= color < len(COLORS) else (255, 0, 255),
                outline=(60, 60, 60),
            )


def render_examples_image(
    examples: Iterable[ArcExample],
    path: Path,
    *,
    cell_size: int = 14,
) -> Path:
    values = list(examples)
    if not values:
        raise ValueError("At least one example is required for image rendering")
    margin, label_height, gap = 12, 18, 24
    row_widths = [
        len(example.input[0]) * cell_size + gap + len(example.output[0]) * cell_size
        for example in values
    ]
    row_heights = [
        max(len(example.input), len(example.output)) * cell_size + label_height + margin
        for example in values
    ]
    image = Image.new(
        "RGB",
        (max(row_widths) + 2 * margin, sum(row_heights) + margin),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    top = margin
    for index, (example, row_height) in enumerate(zip(values, row_heights, strict=True), start=1):
        left = margin
        draw.text((left, top), f"{index}: input", fill=(0, 0, 0), font=font)
        _draw_grid(draw, example.input, left, top + label_height, cell_size)
        left += len(example.input[0]) * cell_size + gap
        draw.text((left, top), "output", fill=(0, 0, 0), font=font)
        _draw_grid(draw, example.output, left, top + label_height, cell_size)
        top += row_height
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
