from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from neurogolf.errors import MappingError
from neurogolf.tasks.loader import load_arc_agi_dataset, load_kaggle_dataset
from neurogolf.tasks.models import ArcExample, validate_grid


@given(
    height=st.integers(min_value=1, max_value=10),
    width=st.integers(min_value=1, max_value=10),
    data=st.data(),
)
def test_rectangular_grid_property(height: int, width: int, data: st.DataObject):
    rows = data.draw(
        st.lists(
            st.lists(st.integers(min_value=0, max_value=9), min_size=width, max_size=width),
            min_size=height,
            max_size=height,
        )
    )
    assert validate_grid(rows) == rows


@pytest.mark.parametrize(
    "grid",
    [[], [[]], [[0], [0, 1]], [[10]], [[True]], [[-1]]],
)
def test_invalid_grids_are_rejected(grid):
    with pytest.raises((ValueError, ValidationError)):
        ArcExample(input=grid, output=[[0]])


def test_kaggle_loader_and_alias(settings):
    dataset = load_kaggle_dataset(settings.paths.kaggle_data_root / "task001.json")
    assert dataset.train and dataset.test and dataset.arc_gen
    dumped = dataset.model_dump(by_alias=True)
    assert "arc-gen" in dumped


def test_two_canonical_name_fields_are_supported(settings):
    root = settings.paths.arcgen_root / "external/ARC-AGI/data/training"
    assert load_arc_agi_dataset(root / "9edfc990.json").name == "9edfc990"
    assert load_arc_agi_dataset(root / "b230c067.json").name == "b230c067"


def test_malformed_json_diagnostic(tmp_path: Path):
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"train": [], "test": [], "arc-gen": []}), encoding="utf-8")
    with pytest.raises(MappingError, match="Malformed ARC task data"):
        load_kaggle_dataset(path)
