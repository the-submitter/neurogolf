"""Stable, namespace-separated seed scheduling and boundary-biased sampling."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable

from neurogolf.arcgen.generator import GeneratorOracle
from neurogolf.tasks.models import GeneratedExample


def derive_seed(namespace: str, task_num: int, base_seed: int, index: int) -> int:
    payload = f"neurogolf:{namespace}:v1:{task_num}:{base_seed}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


def seed_schedule(namespace: str, task_num: int, base_seed: int, count: int) -> list[int]:
    return [derive_seed(namespace, task_num, base_seed, index) for index in range(count)]


def sample_many(
    oracle: GeneratorOracle,
    *,
    count: int,
    base_seed: int,
    namespace: str = "development",
    deadline: float | None = None,
) -> list[GeneratedExample]:
    result: list[GeneratedExample] = []
    for seed in seed_schedule(namespace, oracle.record.task_num, base_seed, count):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"Generation deadline exceeded in {namespace}")
        result.append(oracle.generate(seed))
    return result


def _boundary_score(example: GeneratedExample) -> tuple[int, int, int, int]:
    in_height, in_width = len(example.input), len(example.input[0])
    out_height, out_width = len(example.output), len(example.output[0])
    colors = len({cell for row in example.input for cell in row})
    edge_distance = min(in_height, in_width, 31 - in_height, 31 - in_width)
    return edge_distance, -max(in_height * in_width, out_height * out_width), -colors, example.seed


def boundary_biased_sample(
    oracle: GeneratorOracle,
    *,
    count: int,
    base_seed: int,
    pool_multiplier: int = 4,
    deadline: float | None = None,
) -> list[GeneratedExample]:
    """Select deterministic shape/color extremes from a larger oracle pool."""

    pool = sample_many(
        oracle,
        count=max(count, count * pool_multiplier),
        base_seed=base_seed,
        namespace="boundary-pool",
        deadline=deadline,
    )
    ordered = sorted(pool, key=_boundary_score)
    selected: list[GeneratedExample] = []
    selected_seeds: set[int] = set()
    seen: set[tuple[int, int, int, int, int]] = set()
    for example in ordered:
        signature = (
            len(example.input),
            len(example.input[0]),
            len(example.output),
            len(example.output[0]),
            len({cell for row in example.input for cell in row}),
        )
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(example)
        selected_seeds.add(example.seed)
        if len(selected) == count:
            return selected
    # If the generator has fewer distinct shape/color signatures than requested,
    # deterministically fill the suite instead of silently returning too few cases.
    for example in ordered:
        if example.seed in selected_seeds:
            continue
        selected.append(example)
        if len(selected) == count:
            return selected
    return selected


def unique_examples(examples: Iterable[GeneratedExample]) -> list[GeneratedExample]:
    seen: set[str] = set()
    result: list[GeneratedExample] = []
    for example in examples:
        digest = hashlib.sha256(repr((example.input, example.output)).encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            result.append(example)
    return result
