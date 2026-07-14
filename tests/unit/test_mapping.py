from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurogolf.errors import MappingError
from neurogolf.tasks.mapping import TaskMap
from neurogolf.tasks.principles import load_task_principles, task_principle
from scripts.generate_task_principles import build_mapping as build_principle_mapping
from scripts.generate_task_principles import serialized
from utils.map_tasks import DEFAULT_OUTPUT, REPOSITORY_ROOT, build_mapping


def test_mapping_is_bijective_and_resolves_all_forms(settings):
    mapping = TaskMap(settings)
    assert len(mapping) == 400
    records = list(mapping)
    assert len({record.task_num for record in records}) == 400
    assert len({record.task_id for record in records}) == 400
    first = mapping.get(1)
    assert mapping.get("task001") == first
    assert mapping.get(first.task_id) == first
    assert all(
        path.is_file()
        for path in (
            first.kaggle_json_path,
            first.arcgen_generator_path,
            first.arc_agi_json_path,
        )
    )


def test_unknown_task_is_rejected(settings):
    mapping = TaskMap(settings)
    with pytest.raises(MappingError):
        mapping.get(0)
    with pytest.raises(MappingError):
        mapping.get("not-a-task")


def test_arcgen_principles_cover_all_tasks_and_generated_file_is_current(
    repository_root: Path,
) -> None:
    principles = load_task_principles(repository_root)
    assert len(principles) == 400
    assert principles["task001"] == {"task_id": "007bbfb7", "principle": "fractal"}
    assert principles["task002"] == {"task_id": "00d62c1b", "principle": "honeypots"}
    assert task_principle(repository_root, 1, "007bbfb7") == "fractal"
    generated = build_principle_mapping(repository_root)
    assert generated == principles
    assert (repository_root / "configs/task_principles.json").read_text() == serialized(generated)


def test_mapping_utility_uses_repository_paths_and_both_arc_splits(
    repository_root: Path, tmp_path: Path
) -> None:
    assert repository_root == REPOSITORY_ROOT
    assert repository_root / "configs/task_map.json" == DEFAULT_OUTPUT
    kaggle = tmp_path / "kaggle"
    arc = tmp_path / "arc"
    kaggle.mkdir()
    pair = {"input": [[0]], "output": [[1]]}
    (kaggle / "task001.json").write_text(json.dumps({"train": [pair]}), encoding="utf-8")
    evaluation = arc / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "007bbfb7.json").write_text(json.dumps({"train": [pair]}), encoding="utf-8")
    assert dict(build_mapping(kaggle, arc)) == {"task001": "007bbfb7"}


@pytest.mark.slow
def test_all_400_mappings_data_generators_and_validators(settings):
    report = TaskMap(settings).validate(deep=True)
    assert report.valid, report.errors[:10]
    assert report.mapped_count == 400
    assert report.checked_datasets == 400
    assert report.checked_generators == 400
