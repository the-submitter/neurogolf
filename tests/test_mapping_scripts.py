from __future__ import annotations

from pathlib import Path

from scripts.generate_task_principles import build_mapping as build_principles
from scripts.generate_task_principles import serialized as serialize_principles
from scripts.map_tasks import build_mapping as build_task_map
from scripts.map_tasks import serialized as serialize_task_map


ROOT = Path(__file__).resolve().parents[1]


def test_task_map_regenerates_from_current_repository_layout() -> None:
    mapping = build_task_map(
        ROOT / "kaggle_tasks_data",
        ROOT / "ARC-GEN" / "external" / "ARC-AGI" / "data",
        ROOT / "ARC-GEN" / "tasks",
    )
    assert len(mapping) == 400
    assert serialize_task_map(mapping) == (ROOT / "utils" / "task_map.json").read_text()


def test_task_principles_regenerate_from_current_repository_layout() -> None:
    mapping = build_principles(ROOT)
    assert len(mapping) == 400
    assert serialize_principles(mapping) == (
        ROOT / "utils" / "task_principles.json"
    ).read_text()
