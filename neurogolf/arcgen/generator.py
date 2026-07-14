"""Canonical deterministic generator-oracle API."""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

from pydantic import ValidationError

from neurogolf.arcgen.importer import ImportedGenerator, import_generator, import_lock
from neurogolf.arcgen.tracing import traced_call
from neurogolf.config.models import NeuroGolfSettings
from neurogolf.errors import GeneratorError
from neurogolf.tasks.mapping import TaskMap
from neurogolf.tasks.models import ArcAgiDataset, ArcExample, GeneratedExample, TaskRecord


class GeneratorOracle:
    """Seed-isolated ARC-GEN execution with validation and bounded retries."""

    def __init__(
        self,
        record: TaskRecord,
        settings: NeuroGolfSettings,
        imported: ImportedGenerator | None = None,
    ) -> None:
        self.record = record
        self.settings = settings
        self.imported = imported or import_generator(record, settings.paths.arcgen_root)
        self._validation: ArcAgiDataset | None = None

    @classmethod
    def for_task(
        cls, task: int | str, settings: NeuroGolfSettings, task_map: TaskMap | None = None
    ) -> GeneratorOracle:
        mapping = task_map or TaskMap(settings)
        return cls(mapping.get(task), settings)

    def validate_generator(self) -> ArcAgiDataset:
        """Invoke canonical validate() once and validate its complete return structure."""

        if self._validation is not None:
            return self._validation
        common_random = getattr(self.imported.common_module, "random", random)
        try:
            with import_lock():
                state = common_random.getstate()
                try:
                    common_random.seed(0)
                    payload, _trace = traced_call(self.imported.validate)
                finally:
                    common_random.setstate(state)
            self._validation = ArcAgiDataset.model_validate(payload)
        except (Exception, ValidationError) as error:
            raise GeneratorError(
                f"validate() failed for task {self.record.task_id}: {type(error).__name__}: {error}"
            ) from error
        return self._validation

    def _call_seeded(self, seed: int) -> tuple[dict[str, Any], float]:
        common_random = getattr(self.imported.common_module, "random", random)
        with import_lock():
            state = common_random.getstate()
            try:
                common_random.seed(seed)
                payload, trace = traced_call(self.imported.generate)
            finally:
                common_random.setstate(state)
        if not isinstance(payload, dict):
            raise GeneratorError(
                f"generate() for {self.record.task_id} returned "
                f"{type(payload).__name__}, expected dict"
            )
        return payload, trace.duration_seconds

    def generate(self, seed: int, *, validate: bool = True) -> GeneratedExample:
        """Generate one normalized example, retrying deterministically on rejection."""

        if validate:
            self.validate_generator()
        errors: list[str] = []
        limit = self.settings.generation.max_attempts_per_seed
        for attempt in range(limit):
            actual_seed = seed + attempt * 1_000_003
            try:
                payload, duration = self._call_seeded(actual_seed)
                example = ArcExample.model_validate(payload)
                metadata = {
                    "requested_seed": seed,
                    "attempt": attempt,
                    "generate_signature": self.imported.metadata.generate_signature,
                    "generator_sha256": self.imported.metadata.source_sha256,
                    "duration_seconds": duration,
                    "validation_exercised": validate,
                    "input_shape": [len(example.input), len(example.input[0])],
                    "output_shape": [len(example.output), len(example.output[0])],
                }
                return GeneratedExample(
                    task_num=self.record.task_num,
                    task_id=self.record.task_id,
                    seed=actual_seed,
                    input=example.input,
                    output=example.output,
                    metadata=metadata,
                )
            except Exception as error:
                errors.append(f"seed {actual_seed}: {type(error).__name__}: {error}")
        recent = "; ".join(errors[-3:])
        raise GeneratorError(
            f"Generator {self.record.task_id} failed after {limit} attempts "
            f"from seed {seed}: {recent}"
        )

    def metadata(self) -> dict[str, Any]:
        return asdict(self.imported.metadata)
