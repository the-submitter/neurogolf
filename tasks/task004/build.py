#!/usr/bin/env python3
"""Build and verify the compact NeuroGolf task004 (ARC 025d127b) model."""

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
TASK_DATA = ROOT / "kaggle_tasks_data" / "task004.json"
GENERATOR = ROOT / "ARC-GEN" / "tasks" / "task_025d127b.py"
WORK_MODEL = Path(__file__).with_name("task004.onnx")
SUBMISSION_MODEL = ROOT / "submission" / "task004.onnx"
SHAPE = (1, 10, 30, 30)


def dense(name: str, dtype: int, dims: list[int], values: list[int | float]):
    return helper.make_tensor(name, dtype, dims, values)


def build_model() -> onnx.ModelProto:
    # Every non-bottom outline row has another colored row immediately below
    # it.  Bottom rows do not.  Count colored cells in each row, then use a
    # two-tap 1-D convolution to emit [shift, keep] coefficients:
    #
    #     [next_count, 1 - next_count]
    #
    # On a non-bottom row next_count >= 2.  At locations shared by the shifted
    # and unshifted outlines the coefficients sum to one; at exclusive
    # locations their signs choose the shifted outline.  On a bottom row they
    # are exactly [0, 1], selecting the unchanged row.
    nodes = [
        helper.make_node(
            "Einsum",
            ["input", "safe_name_0"],
            ["safe_name_1"],
            equation="ncrw,kc->nkr",
        ),
        helper.make_node(
            "Conv",
            ["safe_name_1", "safe_name_2", "safe_name_3"],
            ["safe_name_4"],
            kernel_shape=[2],
            pads=[0, 1],
        ),
        # Spatial bases: p=0 shifts right and p=1 is identity.
        # Color bases: u=0 writes transformed colors and their negative into
        # black; u=1 writes the unchanged valid-grid mask into black.
        # The four-entry core combines these bases for shift/keep rows.  This
        # also leaves the 30x30 padding empty and repairs black at column zero.
        helper.make_node(
            "Einsum",
            [
                "input",
                "input",
                "input",
                "safe_name_4",
                "safe_name_4",
                "safe_name_5",
                "safe_name_5",
                "safe_name_6",
                "safe_name_7",
            ],
            ["output"],
            equation=(
                "nirw,nisw,nkrv,nar,nbr,h w v,q r s,u i o,a b h q u->norv"
            ).replace(" ", ""),
        ),
    ]

    # Sequential safe_name_* names make the official sanitizer idempotent.
    # All constants are deliberately dense.  ONNX 1.21 full shape inference
    # rejects sparse initializers as Einsum inputs, even though ORT can execute
    # them, so such a graph would not survive the official score_network check.
    initializers = [
        # Sum nonblack channels while retaining a singleton Conv channel.
        dense("safe_name_0", TensorProto.FLOAT, [1, 10], [0.0] + [1.0] * 9),
        # Cross-correlation taps read the following row.  Channel zero emits
        # next_count; channel one emits 1-next_count via its bias.
        dense("safe_name_2", TensorProto.FLOAT, [2, 1, 2], [0.0, 1.0, 0.0, -1.0]),
        dense("safe_name_3", TensorProto.FLOAT, [2], [0.0, 1.0]),
    ]

    spatial_values = [0.0] * (2 * 30 * 30)
    # p=0: source w appears at destination w+1.
    for w in range(29):
        spatial_values[w * 30 + (w + 1)] = 1.0
    # p=1: identity.
    for w in range(30):
        spatial_values[30 * 30 + w * 30 + w] = 1.0
    initializers.append(
        dense("safe_name_5", TensorProto.FLOAT, [2, 30, 30], spatial_values)
    )

    color_values = [0.0] * (2 * 10 * 10)
    # u=0: preserve every nonblack color and subtract its occupancy from black.
    for color in range(1, 10):
        color_values[color * 10] = -1.0
        color_values[color * 10 + color] = 1.0
    # u=1: sum all input channels into output black, i.e. the valid-grid mask.
    for color in range(10):
        color_values[100 + color * 10] = 1.0
    initializers.append(
        dense("safe_name_6", TensorProto.FLOAT, [2, 10, 10], color_values)
    )

    # The second copy of input is idempotent when q=identity: x*x=x.  When
    # q=successor it detects the one penultimate-row pixel whose same-colored
    # continuation is directly below.  That right endpoint must stay in place
    # rather than move beyond the outline's bottom-right corner.  The core
    # applies the base shift/keep rule at q=identity and adds
    # (identity - shift) for this matched endpoint at q=successor.
    # The mode map is supplied twice.  Besides leaving the base rule unchanged
    # because shift+keep=1, this supplies the quadratic correction
    # next_count*(next_count-2).  It is zero on top/interior transitions (whose
    # next row has two colored cells) and strong enough to flip the one matched
    # right endpoint when the following row is a horizontal bottom edge.
    # Axes are [row_mode a, row_mode b, horizontal h, vertical q,
    # color_basis u], where mode zero is successor/shift and mode one identity.
    core_values = [0.0] * 32

    def core_index(a: int, b: int, h: int, q: int, u: int) -> int:
        return (((a * 2 + b) * 2 + h) * 2 + q) * 2 + u

    # Base transformation, q=identity.  Repeat for both b values, whose mode
    # coefficients sum to one.
    for b in range(2):
        core_values[core_index(0, b, 0, 1, 0)] = 1.0  # shift colors
        core_values[core_index(0, b, 1, 1, 1)] = 1.0  # valid mask
        core_values[core_index(1, b, 1, 1, 0)] = 1.0  # keep colors
        core_values[core_index(1, b, 1, 1, 1)] = 1.0  # valid mask
    # Correction C = mode0 * (-mode0 - 2*mode1) = N*(N-2).
    for b, coefficient in ((0, -1.0), (1, -2.0)):
        core_values[core_index(0, b, 1, 0, 0)] = coefficient
        core_values[core_index(0, b, 0, 0, 0)] = -coefficient
    initializers.append(
        dense("safe_name_7", TensorProto.FLOAT, [2, 2, 2, 2, 2], core_values)
    )

    value_info = [
        helper.make_tensor_value_info("safe_name_1", TensorProto.FLOAT, [1, 1, 30]),
        helper.make_tensor_value_info("safe_name_4", TensorProto.FLOAT, [1, 2, 30]),
    ]
    graph = helper.make_graph(
        nodes,
        "task004_tilt",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        initializer=initializers,
        value_info=value_info,
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 12)],
        producer_name="neurogolf-task004",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def one_hot(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros(SHAPE, dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            result[0, color, row, col] = 1.0
    return result


def check_example(session: ort.InferenceSession, example: dict) -> bool:
    actual = session.run(["output"], {"input": one_hot(example["input"])})[0] > 0
    expected = one_hot(example["output"]).astype(bool)
    return bool(np.array_equal(actual, expected))


def check_reference(example: dict) -> bool:
    """Evaluate the final Einsum algebra directly, without slow ORT contraction."""

    source = one_hot(example["input"])[0]
    colored = source.copy()
    colored[0] = 0.0
    next_count = np.zeros(30, dtype=np.float32)
    next_count[:-1] = colored[:, 1:, :].sum(axis=(0, 2))

    shifted = np.zeros_like(colored)
    shifted[:, :, 1:] = colored[:, :, :-1]
    logits = (
        next_count[None, :, None] * shifted
        + (1.0 - next_count)[None, :, None] * colored
    )

    # A directly-below same-color pixel occurs at both top-left and
    # penultimate-right corners.  N*(N-2) suppresses the former (N=2) and
    # corrects the latter (N>=3) from shifted back to unchanged.
    below_match = np.zeros_like(colored)
    below_match[:, :-1, :] = colored[:, :-1, :] * colored[:, 1:, :]
    correction = next_count * (next_count - 2.0)
    correction = correction[None, :, None] * below_match
    shifted_correction = np.zeros_like(correction)
    shifted_correction[:, :, 1:] = correction[:, :, :-1]
    logits += correction - shifted_correction

    valid = source.sum(axis=0, keepdims=True)
    logits *= valid
    output = np.zeros_like(source)
    output[1:] = logits[1:]
    output[0] = valid[0] - logits[1:].sum(axis=0)
    expected = one_hot(example["output"])[0].astype(bool)
    return bool(np.array_equal(output > 0.0, expected))


def load_generator():
    sys.path.insert(0, str(ROOT / "ARC-GEN"))
    spec = importlib.util.spec_from_file_location("task_025d127b", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def verify(path: Path, random_cases: int) -> tuple[int, int]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(path.read_bytes(), options, providers=["CPUExecutionProvider"])

    examples = json.loads(TASK_DATA.read_text())
    passed = total = 0
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[split]):
            total += 1
            if not check_example(session, example):
                raise AssertionError(f"failed {split}[{index}]")
            passed += 1

    generator = load_generator()
    random.seed(25127)
    for index in range(random_cases):
        if not check_reference(generator.generate()):
            raise AssertionError(f"failed random algebraic generator case {index}")
    return passed, total


def official_cost(path: Path) -> tuple[int, int, float]:
    sys.path.insert(0, str(ROOT))
    from utils.neurogolf_utils import sanitize_model, score_network

    sanitized = sanitize_model(onnx.load(path))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected the model")
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(path.parent / "task004-profile")
    session = ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    session.run(["output"], {"input": one_hot([[0] * 16 for _ in range(16)])})
    trace_path = Path(session.end_profiling())
    memory, params = score_network(sanitized, trace_path)
    trace_path.unlink(missing_ok=True)
    if memory is None or params is None:
        raise RuntimeError("official scorer rejected the model")
    points = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
    return memory, params, points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=5000)
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()

    WORK_MODEL.parent.mkdir(parents=True, exist_ok=True)
    SUBMISSION_MODEL.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), WORK_MODEL)
    passed, total = verify(WORK_MODEL, args.random_cases)
    memory, params, points = official_cost(WORK_MODEL)
    if not args.no_copy:
        shutil.copy2(WORK_MODEL, SUBMISSION_MODEL)

    print(f"verified: {passed}/{total} packaged + {args.random_cases} random")
    print(f"official cost: {memory} bytes + {params} params = {memory + params}")
    print(f"estimated points: {points:.6f}")
    print(f"model: {WORK_MODEL} ({WORK_MODEL.stat().st_size} bytes)")
    if not args.no_copy:
        print(f"submission: {SUBMISSION_MODEL}")


if __name__ == "__main__":
    main()
