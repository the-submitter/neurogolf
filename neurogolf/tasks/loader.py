"""Canonical UTF-8 JSON loaders with strict schema diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from neurogolf.errors import MappingError
from neurogolf.tasks.models import ArcAgiDataset, TaskDataset

DatasetT = TypeVar("DatasetT", bound=BaseModel)


def _load(path: Path, model: type[DatasetT]) -> DatasetT:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise MappingError(f"Could not read task data {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise MappingError(f"Malformed JSON in {path}: {error}") from error
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        raise MappingError(f"Malformed ARC task data in {path}: {error}") from error


def load_kaggle_dataset(path: Path) -> TaskDataset:
    return _load(path, TaskDataset)


def load_arc_agi_dataset(path: Path) -> ArcAgiDataset:
    return _load(path, ArcAgiDataset)

