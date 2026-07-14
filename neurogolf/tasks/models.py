"""Strict typed records for ARC grids, mapped tasks, and validation reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Grid = list[list[int]]


def validate_grid(value: Any) -> Grid:
    """Validate and copy a non-empty rectangular ARC color grid."""

    if not isinstance(value, list) or not value:
        raise ValueError("grid must be a non-empty list of rows")
    width: int | None = None
    result: Grid = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or not row:
            raise ValueError(f"grid row {row_index} must be a non-empty list")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("grid must be rectangular")
        copied: list[int] = []
        for col_index, color in enumerate(row):
            if isinstance(color, bool) or not isinstance(color, int) or not 0 <= color <= 9:
                raise ValueError(
                    f"grid color at ({row_index}, {col_index}) must be an integer 0..9"
                )
            copied.append(color)
        result.append(copied)
    return result


class ArcExample(BaseModel):
    """One normalized ARC input/output example."""

    model_config = ConfigDict(extra="forbid")
    input: Grid
    output: Grid

    _validate_input = field_validator("input", mode="before")(validate_grid)
    _validate_output = field_validator("output", mode="before")(validate_grid)


class ArcAgiDataset(BaseModel):
    """Canonical ARC-AGI train/test split."""

    model_config = ConfigDict(extra="forbid")
    train: list[ArcExample]
    test: list[ArcExample]
    # Two canonical ARC-AGI files carry this harmless provenance field.
    name: str | None = None

    @model_validator(mode="after")
    def require_examples(self) -> "ArcAgiDataset":
        if not self.train or not self.test:
            raise ValueError("ARC-AGI train and test lists must both be non-empty")
        return self


class TaskDataset(ArcAgiDataset):
    """Kaggle dataset with its fixed ARC-GEN examples."""

    arc_gen: list[ArcExample] = Field(alias="arc-gen")

    @model_validator(mode="after")
    def require_arc_gen(self) -> "TaskDataset":
        if not self.arc_gen:
            raise ValueError("Kaggle 'arc-gen' list must be non-empty")
        return self


class TaskRecord(BaseModel):
    """Resolved one-to-one mapping of all canonical task sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    task_num: int = Field(ge=1, le=400)
    task_id: str = Field(pattern=r"^[0-9a-f]{8}$")
    kaggle_json_path: Path
    arcgen_generator_path: Path
    arc_agi_json_path: Path
    arc_agi_split: Literal["training", "evaluation"]


class GeneratedExample(BaseModel):
    """A reproducible example returned by the ARC-GEN oracle."""

    model_config = ConfigDict(extra="forbid")
    task_num: int = Field(ge=1, le=400)
    task_id: str = Field(pattern=r"^[0-9a-f]{8}$")
    seed: int
    input: Grid
    output: Grid
    metadata: dict[str, Any] = Field(default_factory=dict)

    _validate_input = field_validator("input", mode="before")(validate_grid)
    _validate_output = field_validator("output", mode="before")(validate_grid)


class MappingValidationReport(BaseModel):
    """Machine-readable result of shallow or deep map validation."""

    valid: bool
    expected_count: int
    mapped_count: int
    unique_task_numbers: int
    unique_task_ids: int
    checked_generators: int
    checked_datasets: int
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duration_seconds: float
