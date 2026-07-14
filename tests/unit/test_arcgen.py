from __future__ import annotations

import random
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from neurogolf.arcgen.generator import GeneratorOracle
from neurogolf.arcgen.importer import import_generator
from neurogolf.arcgen.sampling import boundary_biased_sample, sample_many, seed_schedule
from neurogolf.arcgen.validator import validate_oracle
from neurogolf.tasks.mapping import TaskMap
from neurogolf.tasks.models import GeneratedExample


def test_absolute_import_does_not_pollute_common(settings):
    record = TaskMap(settings).get(1)
    previous = sys.modules.get("common")
    imported = import_generator(record, settings.paths.arcgen_root)
    assert callable(imported.generate) and callable(imported.validate)
    assert sys.modules.get("common") is previous
    assert imported.metadata.generate_signature.startswith("(")


@pytest.mark.parametrize("task_reference", [1, "97a05b5b", "0e206a2e"])
def test_representative_generators_are_reproducible(settings, task_reference):
    record = TaskMap(settings).get(task_reference)
    oracle = GeneratorOracle(record, settings)
    first = oracle.generate(12345)
    second = oracle.generate(12345)
    assert first.input == second.input
    assert first.output == second.output
    assert first.seed == second.seed
    assert first.metadata["validation_exercised"] is True


def test_validator_matches_canonical_arc_agi(settings):
    oracle = GeneratorOracle.for_task(1, settings)
    state = random.getstate()
    report = validate_oracle(oracle, sample_count=3, seed=11)
    assert random.getstate() == state
    assert report["valid"]
    assert report["validate_matches_arc_agi"] == {"train": True, "test": True}


def test_seed_namespaces_and_boundary_selection(settings):
    oracle = GeneratorOracle.for_task(2, settings)
    development = seed_schedule("development", 2, 10, 4)
    regression = seed_schedule("regression", 2, 10, 4)
    assert len(set(development)) == 4
    assert development != regression
    assert len(sample_many(oracle, count=2, base_seed=1)) == 2
    assert len(boundary_biased_sample(oracle, count=2, base_seed=1)) == 2


def test_boundary_selection_fills_when_all_signatures_match(settings):
    class UniformOracle:
        record = SimpleNamespace(task_num=1)

        @staticmethod
        def generate(seed: int) -> GeneratedExample:
            return GeneratedExample(
                task_num=1,
                task_id="007bbfb7",
                seed=seed,
                input=[[0]],
                output=[[0]],
            )

    oracle = cast(Any, UniformOracle())
    selected = boundary_biased_sample(oracle, count=3, base_seed=8)
    assert len(selected) == 3
    assert len({item.seed for item in selected}) == 3
