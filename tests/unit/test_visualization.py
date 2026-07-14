from __future__ import annotations

from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from neurogolf.tasks.models import ArcExample
from neurogolf.visualization.diff import difference_coordinates, render_diff
from neurogolf.visualization.image import render_examples_image
from neurogolf.visualization.text import grid_bbox, render_grid, render_pair


def test_symbol_numeric_indices_bbox_and_pair_rendering():
    example = ArcExample(input=[[0, 1], [2, 3]], output=[[3, 2], [1, 0]])
    assert ". B" in render_grid(example.input)
    assert "0 1" in render_grid(example.input, mode="numeric", indices=True)
    assert grid_bbox(example.input) == (0, 0, 1, 1)
    rendered = render_pair(example)
    assert "INPUT" in rendered and "OUTPUT" in rendered
    assert "|" not in rendered


@given(
    left=st.lists(
        st.lists(st.integers(0, 9), min_size=1, max_size=5), min_size=1, max_size=5
    ).filter(lambda rows: len({len(row) for row in rows}) == 1),
)
def test_text_rendering_never_loses_rows(left):
    rendered = render_grid(left)
    assert len(rendered.splitlines()) == len(left)


def test_diff_coordinates_and_changed_view():
    expected, actual = [[0, 1], [2, 3]], [[0, 1], [2, 4]]
    assert difference_coordinates(expected, actual) == [
        {"row": 1, "col": 1, "expected": 3, "actual": 4}
    ]
    assert "DIFF(R)" in render_diff(expected, actual)


def test_empty_and_ragged_diagnostic_outputs_render() -> None:
    assert render_grid([]) == "<empty>"
    assert "<empty-row>" in render_grid([[], [1, 2]])
    rendered = render_diff([[1, 2]], [[], [3]])
    assert "EXPECTED" in rendered and "ACTUAL" in rendered


def test_pillow_image_renderer(tmp_path: Path):
    path = render_examples_image(
        [ArcExample(input=[[0, 1]], output=[[1, 0]])], tmp_path / "examples.png"
    )
    assert path.is_file() and path.stat().st_size > 0
