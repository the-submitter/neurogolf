#!/usr/bin/env python3
"""Build the optimized NeuroGolf model for task 006 (ARC 0520fde7)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[2]
TASK_DATA = ROOT / "kaggle_tasks_data" / "task006.json"
GENERATOR = ROOT / "ARC-GEN" / "tasks" / "task_0520fde7.py"
WORK_MODEL = Path(__file__).resolve().parent / "task006.onnx"
SUBMISSION_MODEL = ROOT / "submission" / "task006.onnx"

INPUT_SHAPE = (1, 10, 30, 30)


def build_model() -> onnx.ModelProto:
    # Dilation 4 aligns each left-panel cell at tap 0 with its corresponding
    # right-panel cell at tap 1.  For valid panel cells, let Lb/Lu and Rb/Ru
    # denote black/blue one-hot values.  The only positive outputs are:
    #
    #   black = -Lu + 2*Rb + Ru   (positive unless both cells are blue)
    #   red   = -Lb + Ru          (positive only when both cells are blue)
    #
    # These bias-free inequalities also keep the unpaired right panel at
    # output columns 4..6 nonpositive.  Only five weights are nonzero.  The
    # tensor remains dense because ONNX 1.21 full shape inference (which the
    # official scorer invokes) rejects a sparse tensor as a Conv weight.
    weight_shape = [10, 5, 1, 2]
    flat_indices = [
        1,   # output black, right-tap black: +2
        2,   # output black, left-tap blue:  -1
        3,   # output black, right-tap blue: +1
        20,  # output red,   left-tap black:  -1
        23,  # output red,   right-tap blue: +1
    ]
    values = [2.0, -1.0, 1.0, -1.0, 1.0]
    weight_values = np.zeros(math.prod(weight_shape), dtype=np.float32)
    weight_values[flat_indices] = values
    weights = helper.make_tensor(
        "W", TensorProto.FLOAT, weight_shape, weight_values
    )

    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["input", "W"], ["output"], name="output",
                kernel_shape=[1, 2], dilations=[1, 4], group=2,
                pads=[0, 0, 0, 4],
            )
        ],
        "task006_intersection",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, INPUT_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, INPUT_SHAPE)],
        [weights],
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
        producer_name="neurogolf-task006",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def one_hot(grid: list[list[int]]) -> np.ndarray:
    value = np.zeros(INPUT_SHAPE, dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            value[0, color, row, col] = 1.0
    return value


def verify_model(path: Path) -> tuple[int, int]:
    examples = json.loads(TASK_DATA.read_text())
    session = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
    passed = total = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            total += 1
            actual = (session.run(["output"], {"input": one_hot(example["input"])})[0] > 0).astype(np.float32)
            expected = one_hot(example["output"])
            if np.array_equal(actual, expected):
                passed += 1
            else:
                raise AssertionError(f"task006 failed {split} example {total}")
    return passed, total


def verify_truth_table(path: Path) -> tuple[int, int]:
    """Check every black/blue pair at all nine aligned panel positions."""

    session = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
    passed = 0
    for left_color in (0, 1):
        for right_color in (0, 1):
            input_grid = [
                [left_color] * 3 + [5] + [right_color] * 3
                for _ in range(3)
            ]
            output_color = 2 if left_color == right_color == 1 else 0
            expected_grid = [[output_color] * 3 for _ in range(3)]
            actual = session.run(["output"], {"input": one_hot(input_grid)})[0] > 0
            if not np.array_equal(actual, one_hot(expected_grid).astype(bool)):
                raise AssertionError(
                    f"truth-table failure for colors {left_color}/{right_color}"
                )
            passed += 1
    return passed, 4


def verify_generated(path: Path, cases: int, seed: int = 6006) -> tuple[int, int]:
    """Check fresh default-parameter samples from the authoritative generator."""

    if cases <= 0:
        return 0, 0

    arc_gen_root = str(GENERATOR.parents[1])
    if arc_gen_root not in sys.path:
        sys.path.insert(0, arc_gen_root)
    spec = importlib.util.spec_from_file_location("task006_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load generator: {GENERATOR}")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    random.seed(seed)
    session = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
    for index in range(cases):
        example = generator.generate()
        actual = (
            session.run(["output"], {"input": one_hot(example["input"])})[0] > 0
        ).astype(np.float32)
        if not np.array_equal(actual, one_hot(example["output"])):
            raise AssertionError(f"task006 failed generated example {index + 1}")
    return cases, cases


def official_cost(path: Path) -> tuple[int, int, float]:
    sys.path.insert(0, str(ROOT))
    from utils.neurogolf_utils import sanitize_model, score_network

    model = sanitize_model(onnx.load(path))
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(path.parent / "task006-profile")
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    examples = json.loads(TASK_DATA.read_text())
    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            session.run(["output"], {"input": one_hot(example["input"])})
    trace_path = Path(session.end_profiling())
    memory, params = score_network(model, trace_path)
    trace_path.unlink(missing_ok=True)
    if memory is None or params is None:
        raise RuntimeError("official scorer rejected task006.onnx")
    points = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
    return memory, params, points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-copy", action="store_true", help="do not update submission/task006.onnx")
    parser.add_argument(
        "--random-cases", type=int, default=10_000,
        help="number of fresh ARC-GEN cases to verify (default: 10000)",
    )
    args = parser.parse_args()

    WORK_MODEL.parent.mkdir(parents=True, exist_ok=True)
    SUBMISSION_MODEL.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), WORK_MODEL)
    passed, total = verify_model(WORK_MODEL)
    truth_passed, truth_total = verify_truth_table(WORK_MODEL)
    generated_passed, generated_total = verify_generated(
        WORK_MODEL, args.random_cases
    )
    memory, params, points = official_cost(WORK_MODEL)
    if not args.no_copy:
        shutil.copy2(WORK_MODEL, SUBMISSION_MODEL)
    print(f"verified: {passed}/{total}")
    print(f"truth table: {truth_passed}/{truth_total}")
    print(f"fresh ARC-GEN: {generated_passed}/{generated_total}")
    print(f"official cost: {memory} bytes + {params} params = {memory + params}")
    print(f"estimated points: {points:.6f}")
    print(f"model: {WORK_MODEL} ({WORK_MODEL.stat().st_size} bytes)")
    if not args.no_copy:
        print(f"submission: {SUBMISSION_MODEL}")


if __name__ == "__main__":
    main()
