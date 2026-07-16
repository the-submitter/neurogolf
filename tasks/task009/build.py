#!/usr/bin/env python3
"""Build, score, and stress-test the compact task009 ONNX graph."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[2]
TASK_DATA = ROOT / "kaggle_tasks_data" / "task009.json"
GENERATOR = ROOT / "ARC-GEN" / "tasks" / "task_06df4c85.py"
DEFAULT_MODEL = Path(__file__).with_name("task009.onnx")


def tensor(name: str, dims: list[int], values: list[float]) -> onnx.TensorProto:
    return helper.make_tensor(name, TensorProto.FLOAT, dims, values)


def build_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 10, 30, 30]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 10, 30, 30]
    )

    # Eleven latent positions represent the ten possible grid cells plus a
    # dummy copy position. The cell map is shared for all six spatial axes.
    # Cells 0..9 select their two interior pixels; dummy cell 10 accepts every
    # pixel, allowing the local input operand to perform a direct copy.
    cell = [
        float(latent == 10 or pixel in (3 * latent, 3 * latent + 1))
        for latent in range(11)
        for pixel in range(30)
    ]
    # q=0 selects the copy cell; q=1 selects actual lattice cells.
    cell_kind = [
        float((kind == 0 and latent == 10) or (kind == 1 and latent < 10))
        for kind in range(2)
        for latent in range(11)
    ]

    # Relation kind 0 is equality (also valid for dummy cell 10). Relation
    # kind 1 is strict ordering on the ten real lattice cells. Reusing this
    # tensor for rows and columns saves an entire mode-dependent relation.
    relation = [
        (
            float(left == right)
            if kind == 0
            else float(left < right and right < 10)
        )
        for kind in range(2)
        for left in range(11)
        for right in range(11)
    ]

    # A four-term monomial basis replaces three dense 10x10 channel matrices.
    # Copy mode evaluates .5-(u-z)^2, which is positive exactly when output
    # channel u equals local one-hot channel z. Connection mode evaluates
    # c*(.5-(u-c)^2): multiplying by endpoint color c suppresses background
    # endpoint pairs, and the sign is positive only on the matching color.
    channel_feature = [
        float(channel**power)
        for power in range(4)
        for channel in range(10)
    ]
    connection_scale = 1.0e9
    # The two channel polynomials share a 3x4x4 behavior tensor. Behavior 0 is
    # copy, behavior 1 selects the constant feature, and behavior 2 connects.
    # Multiplying two selected behaviors replaces the former 2x4x4x4 core.
    channel_behavior = [0.0] * (3 * 4 * 4)

    def behavior_index(behavior: int, source_power: int, out_power: int) -> int:
        return (behavior * 4 + source_power) * 4 + out_power

    for source_power, out_power, coefficient in (
        (0, 0, 0.5), (0, 2, -1.0), (1, 1, 2.0), (2, 0, -1.0)
    ):
        channel_behavior[behavior_index(0, source_power, out_power)] = coefficient
    for out_power in range(4):
        channel_behavior[behavior_index(1, 0, out_power)] = 1.0
    for source_power, out_power, coefficient in (
        (1, 0, 0.5 * connection_scale),
        (1, 2, -connection_scale),
        (2, 1, 2.0 * connection_scale),
        (3, 0, -connection_scale),
    ):
        channel_behavior[behavior_index(2, source_power, out_power)] = coefficient
    initializers = [
        tensor("cell", [11, 30], cell),
        tensor("cell_kind", [2, 11], cell_kind),
        tensor("relation", [2, 11, 11], relation),
        tensor("channel_feature", [4, 10], channel_feature),
        tensor("channel_behavior", [3, 4, 4], channel_behavior),
        tensor("local_behavior", [2, 3], [1, 0, 0, 0, 1, 0]),
        tensor("endpoint_behavior", [2, 3], [0, 1, 0, 0, 0, 1]),
        # (copy, horizontal, vertical) are respectively
        # (q,row_relation,col_relation) = (0,0,0), (1,0,1), (1,1,0).
        tensor("mode", [2, 2, 2], [1, 0, 0, 0, 0, 1, 1, 0]),
    ]

    # The local input makes copy mode direct, eliminating all four dense 30x30
    # pixel-routing factors from the previous construction. The two remaining
    # input occurrences are candidate endpoints. Factorizing spatial mode,
    # relation mode, and channel polynomial keeps the dense cost below 800.
    node = helper.make_node(
        "Einsum",
        [
            # Interleave coordinate factors with the endpoint inputs. ORT's
            # Einsum planner can then contract each 30-pixel axis before the
            # next large operand arrives instead of materializing a huge
            # Cartesian product of all three input occurrences.
            "input", "cell", "cell", "cell_kind", "cell_kind",
            "input", "cell", "cell", "cell_kind", "cell_kind",
            "relation", "relation", "relation", "relation",
            "cell_kind", "cell_kind", "cell", "cell",
            "channel_feature", "channel_feature", "channel_behavior",
            "endpoint_behavior", "mode", "input", "channel_feature",
            "channel_behavior", "local_behavior",
        ],
        ["output"],
        equation=(
            "bcad,ia,ld,qi,ql,bcef,me,nf,qm,qn,"
            "wik,wkm,vlj,vjn,qk,qj,kr,js,yc,ou,gyo,qg,qwv,"
            "bzrs,xz,hxo,qh->burs"
        ),
    )

    graph = helper.make_graph(
        [node],
        "task009_sparse_gridline_completion",
        [input_info],
        [output_info],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 12)],
        producer_name="neurogolf-task009",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def to_one_hot(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            result[0, color, row, col] = 1.0
    return result


def check_example(session: ort.InferenceSession, example: dict) -> bool:
    actual = session.run(["output"], {"input": to_one_hot(example["input"])})[0] > 0
    expected = to_one_hot(example["output"]).astype(bool)
    return bool(np.array_equal(actual, expected))


def load_generator():
    sys.path.insert(0, str(ROOT / "ARC-GEN"))
    spec = importlib.util.spec_from_file_location("task_06df4c85", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_session(
    model_path: Path, *, profile: bool = False, optimize: bool = False
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    options.enable_profiling = profile
    if profile:
        options.profile_file_prefix = str(model_path.with_suffix(""))
    return ort.InferenceSession(model_path.read_bytes(), options)


def verify(model_path: Path, random_cases: int) -> None:
    # Optimization only accelerates the large stress suite; official_cost()
    # separately loads and profiles the graph with all optimization disabled.
    session = make_session(model_path, optimize=True)
    examples = json.loads(TASK_DATA.read_text())
    failures = []
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[split]):
            if not check_example(session, example):
                failures.append(f"{split}[{index}]")

    generator = load_generator()
    random.seed(6009485)
    for index in range(random_cases):
        if not check_example(session, generator.generate()):
            failures.append(f"random[{index}]")
            break

    if failures:
        raise AssertionError("failed examples: " + ", ".join(failures[:10]))
    known_count = sum(len(examples[name]) for name in ("train", "test", "arc-gen"))
    print(f"verified {known_count} packaged examples + {random_cases} random generator cases")


def official_cost(model_path: Path) -> tuple[int, int, float]:
    # Importing this module is intentionally delayed because it also imports
    # notebook/plotting dependencies irrelevant to building the graph.
    sys.path.insert(0, str(ROOT / "utils"))
    import neurogolf_utils as ng  # pylint: disable=import-outside-toplevel

    model = ng.sanitize_model(onnx.load(model_path))
    if model is None:
        raise RuntimeError("official sanitizer rejected the model")
    session = make_session(model_path, profile=True)
    # One run is sufficient to give the profiler the sole output shape.
    sample = json.loads(TASK_DATA.read_text())["train"][0]
    session.run(["output"], {"input": to_one_hot(sample["input"])})
    trace_path = session.end_profiling()
    try:
        memory, params = ng.score_network(model, trace_path)
    finally:
        Path(trace_path).unlink(missing_ok=True)
    if memory is None or params is None:
        raise RuntimeError("official scorer could not measure the model")
    points = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
    return memory, params, points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--random-cases", type=int, default=1000)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), args.output)
    print(f"saved {args.output} ({args.output.stat().st_size} bytes)")
    verify(args.output, args.random_cases)
    memory, params, points = official_cost(args.output)
    print(f"official cost: {memory} bytes + {params} params = {memory + params}")
    print(f"official score: {points:.6f}")


if __name__ == "__main__":
    main()
