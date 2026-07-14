"""Deterministic ONNX execution and structured expected/actual comparisons."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from neurogolf.judge.official_adapter import OfficialAdapter
from neurogolf.tasks.models import ArcExample, Grid


@dataclass(frozen=True)
class PixelDifference:
    row: int
    col: int
    expected: int | None
    actual: int | None


@dataclass
class ExampleComparison:
    passed: bool
    expected: Grid
    actual: Grid | None
    pixel_differences: list[PixelDifference] = field(default_factory=list)
    shape_difference: dict[str, list[int]] | None = None
    special_output_colors: list[dict[str, int]] = field(default_factory=list)
    malformed_one_hot: list[dict[str, int | str]] = field(default_factory=list)
    runtime_error: str | None = None
    seed: int | None = None
    family: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result


def _shape(grid: Grid) -> list[int]:
    return [len(grid), max((len(row) for row in grid), default=0)]


def _shape_difference(expected: Grid, actual: Grid) -> dict[str, list[int]] | None:
    expected_widths = [len(row) for row in expected]
    actual_widths = [len(row) for row in actual]
    if _shape(expected) == _shape(actual) and expected_widths == actual_widths:
        return None
    return {
        "expected": _shape(expected),
        "actual": _shape(actual),
        "expected_row_widths": expected_widths,
        "actual_row_widths": actual_widths,
    }


def _diff_grids(expected: Grid, actual: Grid) -> list[PixelDifference]:
    height = max(len(expected), len(actual))
    expected_width = max((len(row) for row in expected), default=0)
    actual_width = max((len(row) for row in actual), default=0)
    differences: list[PixelDifference] = []
    for row in range(height):
        for col in range(max(expected_width, actual_width)):
            expected_value = (
                expected[row][col] if row < len(expected) and col < len(expected[row]) else None
            )
            actual_value = (
                actual[row][col] if row < len(actual) and col < len(actual[row]) else None
            )
            if expected_value != actual_value:
                differences.append(PixelDifference(row, col, expected_value, actual_value))
    return differences


class CandidateRunner:
    """Load a sanitized model once and evaluate examples exactly like the official utility."""

    def __init__(self, model_path: Path, adapter: OfficialAdapter):
        self.model_path = model_path
        self.adapter = adapter
        self.model = adapter.load_and_sanitize(model_path)
        self.session: ort.InferenceSession = adapter.create_session(self.model)

    def run_tensor(self, grid: Grid) -> np.ndarray:
        return self.adapter.run_session(self.session, self.adapter.grid_to_tensor(grid))

    def run_grid(self, grid: Grid) -> tuple[Grid, np.ndarray]:
        thresholded = self.run_tensor(grid)
        return self.adapter.tensor_to_grid(thresholded), thresholded

    def compare(
        self,
        example: ArcExample,
        *,
        seed: int | None = None,
        family: str | None = None,
    ) -> ExampleComparison:
        try:
            actual, thresholded = self.run_grid(example.input)
        except Exception as error:
            return ExampleComparison(
                passed=False,
                expected=example.output,
                actual=None,
                runtime_error=str(error),
                seed=seed,
                family=family,
            )
        expected_tensor = self.adapter.convert_example(example)["output"]
        passed = bool(np.array_equal(thresholded, expected_tensor))
        differences = _diff_grids(example.output, actual)
        specials: list[dict[str, int]] = []
        for row_index, row in enumerate(actual):
            for col_index, color in enumerate(row):
                if color in (10, 11):
                    specials.append({"row": row_index, "col": col_index, "color": color})
        malformed: list[dict[str, int | str]] = []
        expected_height, expected_width = _shape(example.output)
        positive_counts = thresholded.sum(axis=1)[0]
        for row in range(30):
            for col in range(30):
                count = int(positive_counts[row, col])
                inside = row < expected_height and col < expected_width
                required = 1 if inside else 0
                if count != required:
                    malformed.append(
                        {
                            "row": row,
                            "col": col,
                            "positive_channels": count,
                            "region": "expected-grid" if inside else "padding",
                        }
                    )
        return ExampleComparison(
            passed=passed,
            expected=example.output,
            actual=actual,
            pixel_differences=differences,
            shape_difference=_shape_difference(example.output, actual),
            special_output_colors=specials,
            malformed_one_hot=malformed,
            seed=seed,
            family=family,
        )

    def compare_many(
        self,
        examples: Iterable[ArcExample],
        *,
        family: str,
        seeds: Iterable[int | None] | None = None,
    ) -> list[ExampleComparison]:
        seed_values = iter(seeds) if seeds is not None else None
        return [
            self.compare(
                example,
                seed=next(seed_values) if seed_values is not None else None,
                family=family,
            )
            for example in examples
        ]
