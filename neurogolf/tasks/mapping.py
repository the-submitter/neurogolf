"""Robust one-to-one task mapping and deep source validation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import time
from typing import Any, Iterator, Literal

from neurogolf.config.models import NeuroGolfSettings
from neurogolf.errors import MappingError
from neurogolf.tasks.loader import load_arc_agi_dataset, load_kaggle_dataset
from neurogolf.tasks.models import MappingValidationReport, TaskRecord

_KEY = re.compile(r"^task(\d{3})$")


def _pair_signature(example: Any) -> str:
    if hasattr(example, "model_dump"):
        example = example.model_dump()
    return json.dumps(example, sort_keys=True, separators=(",", ":"))


class TaskMap:
    """Load, resolve, and validate the canonical Kaggle/ARC-GEN mapping."""

    def __init__(self, settings: NeuroGolfSettings):
        self.settings = settings
        self._mapping = self._read_mapping(settings.paths.task_map)
        self._by_number = self._build_records()
        self._by_id = {record.task_id: record for record in self._by_number.values()}

    @staticmethod
    def _read_mapping(path: Path) -> dict[str, str]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MappingError(f"Could not load canonical task map {path}: {error}") from error
        if not isinstance(value, dict):
            raise MappingError(f"Task map must be a JSON object: {path}")
        mapping: dict[str, str] = {}
        for key, task_id in value.items():
            if not isinstance(key, str) or not _KEY.fullmatch(key):
                raise MappingError(f"Invalid task-map key {key!r}; expected taskNNN")
            if not isinstance(task_id, str) or not re.fullmatch(r"[0-9a-f]{8}", task_id):
                raise MappingError(f"Invalid ARC task ID for {key}: {task_id!r}")
            mapping[key] = task_id
        return mapping

    def _build_records(self) -> dict[int, TaskRecord]:
        records: dict[int, TaskRecord] = {}
        paths = self.settings.paths
        for key, task_id in self._mapping.items():
            match = _KEY.fullmatch(key)
            assert match is not None
            task_num = int(match.group(1))
            candidates: list[tuple[Literal["training", "evaluation"], Path]] = [
                ("training", paths.arcgen_root / "external/ARC-AGI/data/training" / f"{task_id}.json"),
                ("evaluation", paths.arcgen_root / "external/ARC-AGI/data/evaluation" / f"{task_id}.json"),
            ]
            present: list[tuple[Literal["training", "evaluation"], Path]] = [
                (split, path) for split, path in candidates if path.is_file()
            ]
            if len(present) != 1:
                detail = "none" if not present else "both training and evaluation"
                raise MappingError(f"ARC-AGI source for {key}/{task_id} found in {detail}")
            split, arc_path = present[0]
            records[task_num] = TaskRecord(
                task_num=task_num,
                task_id=task_id,
                kaggle_json_path=paths.kaggle_data_root / f"task{task_num:03d}.json",
                arcgen_generator_path=paths.arcgen_root / "tasks" / f"task_{task_id}.py",
                arc_agi_json_path=arc_path,
                arc_agi_split=split,
            )
        return records

    def __len__(self) -> int:
        return len(self._by_number)

    def __iter__(self) -> Iterator[TaskRecord]:
        for task_num in sorted(self._by_number):
            yield self._by_number[task_num]

    def get(self, task: int | str) -> TaskRecord:
        """Resolve a task number, `taskNNN` key, or eight-digit task ID."""

        if isinstance(task, str):
            if re.fullmatch(r"[0-9a-f]{8}", task):
                record = self._by_id.get(task)
                if record is None:
                    raise MappingError(f"Unknown ARC task ID: {task}")
                return record
            match = _KEY.fullmatch(task)
            if match:
                task = int(match.group(1))
            elif task.isdecimal():
                task = int(task)
            else:
                raise MappingError(f"Invalid task reference: {task!r}")
        record = self._by_number.get(task)
        if record is None:
            raise MappingError(f"Unknown Kaggle task number: {task}")
        return record

    def validate(self, *, deep: bool = True, expected_count: int = 400) -> MappingValidationReport:
        """Check cardinality, all canonical paths/data, mapping semantics, and generators."""

        started = time.monotonic()
        errors: list[str] = []
        warnings: list[str] = []
        ids = list(self._mapping.values())
        numbers = list(self._by_number)
        if len(self._mapping) != expected_count:
            errors.append(f"expected {expected_count} mappings, found {len(self._mapping)}")
        expected_numbers = set(range(1, expected_count + 1))
        if set(numbers) != expected_numbers:
            missing = sorted(expected_numbers - set(numbers))
            extra = sorted(set(numbers) - expected_numbers)
            errors.append(f"task-number coverage mismatch; missing={missing}, extra={extra}")
        duplicate_ids = sorted(task_id for task_id, count in Counter(ids).items() if count > 1)
        if duplicate_ids:
            errors.append(f"duplicate ARC task IDs: {duplicate_ids}")

        checked_datasets = 0
        checked_generators = 0
        for record in self:
            prefix = f"task{record.task_num:03d}/{record.task_id}"
            for label, path in (
                ("Kaggle JSON", record.kaggle_json_path),
                ("ARC-GEN generator", record.arcgen_generator_path),
                ("ARC-AGI JSON", record.arc_agi_json_path),
            ):
                if not path.is_file():
                    errors.append(f"{prefix}: missing {label}: {path}")
            if errors and not all(
                path.is_file()
                for path in (
                    record.kaggle_json_path,
                    record.arcgen_generator_path,
                    record.arc_agi_json_path,
                )
            ):
                continue
            try:
                kaggle = load_kaggle_dataset(record.kaggle_json_path)
                source = load_arc_agi_dataset(record.arc_agi_json_path)
                checked_datasets += 1
                for split in ("train", "test"):
                    available = Counter(_pair_signature(item) for item in getattr(source, split))
                    requested = Counter(_pair_signature(item) for item in getattr(kaggle, split))
                    if requested - available:
                        errors.append(f"{prefix}: Kaggle {split} is not a subset of ARC-AGI {split}")
            except MappingError as error:
                errors.append(f"{prefix}: {error}")
                continue

            if not deep:
                continue
            try:
                from neurogolf.arcgen.importer import import_generator

                imported = import_generator(record, self.settings.paths.arcgen_root)
                checked_generators += 1
                validation = imported.validate()
                if not isinstance(validation, dict):
                    raise TypeError("validate() must return a dictionary")
                validated = {
                    "train": validation.get("train"),
                    "test": validation.get("test"),
                }
                from neurogolf.tasks.models import ArcAgiDataset

                ArcAgiDataset.model_validate(validated)
            except Exception as error:  # diagnostics intentionally aggregate all tasks
                errors.append(f"{prefix}: generator validation failed: {type(error).__name__}: {error}")

        return MappingValidationReport(
            valid=not errors,
            expected_count=expected_count,
            mapped_count=len(self._mapping),
            unique_task_numbers=len(set(numbers)),
            unique_task_ids=len(set(ids)),
            checked_generators=checked_generators,
            checked_datasets=checked_datasets,
            errors=errors,
            warnings=warnings,
            duration_seconds=time.monotonic() - started,
        )
