#!/usr/bin/env python3
"""Build and verify the optimized ONNX solution for NeuroGolf task 002."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
TASK_DATA = ROOT / "kaggle_tasks_data" / "task002.json"
DEFAULT_OUTPUT = Path(__file__).with_name("task002.onnx")
GRID_SHAPE = [1, 10, 30, 30]
KERNEL = 8
FILL_KERNEL = 6


def scalar(name: str, value: int | float, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def build_model() -> onnx.ModelProto:
    # Each detector is anchored at a candidate rectangle's top-left corner.
    # A pot specifies every cell except its four unconstrained corner cells:
    # green (channel 3) on the border and black (channel 0) in the interior.
    sizes = [(height, width) for height in range(3, 9) for width in range(3, 9)]
    # A signed template over the green plane is sufficient: +1 requires the
    # border to be green, while -1 rejects green in the expected black interior.
    detect_w = np.zeros((len(sizes), 1, KERNEL, KERNEL), dtype=np.int8)
    # Normalize every border template to the same sum.  Keeping a common
    # threshold makes the detector easier to audit; QLinearConv still requires
    # one bias value per output channel.
    border_sum = 24
    for channel, (height, width) in enumerate(sizes):
        border_cells = (
            [(0, col) for col in range(1, width - 1)]
            + [(height - 1, col) for col in range(1, width - 1)]
            + [(row, 0) for row in range(1, height - 1)]
            + [(row, width - 1) for row in range(1, height - 1)]
        )
        for index, (row, col) in enumerate(border_cells):
            # Each required cell has weight at least one and all positive
            # weights sum to 24.  Missing any border pixel therefore drops the
            # accumulator below the shared threshold.
            detect_w[channel, 0, row, col] = (
                border_sum // len(border_cells)
                + (index < border_sum % len(border_cells))
            )
        detect_w[channel, 0, 1 : height - 1, 1 : width - 1] = -1
    detect_b = np.full(len(sizes), -(border_sum - 1), dtype=np.int32)

    # A second quantized convolution expands each top-left detection over the
    # corresponding rectangle interior. Its asymmetric padding reverses the
    # offsets while preserving the fixed 30x30 spatial shape.
    fill_w = np.zeros((1, len(sizes), FILL_KERNEL, FILL_KERNEL), dtype=np.uint8)
    for channel, (height, width) in enumerate(sizes):
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                fill_w[0, channel, FILL_KERNEL - row, FILL_KERNEL - col] = 1

    initializers = [
        scalar("one_f", 1.0, np.float32),
        scalar("zero_u8", 0, np.uint8),
        scalar("zero_i8", 0, np.int8),
        numpy_helper.from_array(np.asarray([3, 0, 0], dtype=np.int64), name="slice_starts"),
        numpy_helper.from_array(np.asarray([4, 20, 20], dtype=np.int64), name="slice_ends"),
        numpy_helper.from_array(np.asarray([1, 2, 3], dtype=np.int64), name="slice_axes"),
        numpy_helper.from_array(detect_w, name="detect_w"),
        numpy_helper.from_array(detect_b, name="detect_b"),
        numpy_helper.from_array(fill_w, name="fill_w"),
        # One-hot yellow, broadcast by Where across both spatial dimensions.
        numpy_helper.from_array(
            np.eye(10, dtype=np.float32)[4].reshape(1, 10, 1, 1), name="yellow"
        ),
    ]

    nodes = [
        helper.make_node(
            "Slice",
            ["input", "slice_starts", "slice_ends", "slice_axes"],
            ["cropped"],
        ),
        helper.make_node("Cast", ["cropped"], ["quantized"], to=TensorProto.UINT8),
        helper.make_node(
            "QLinearConv",
            [
                "quantized",
                "one_f",
                "zero_u8",
                "detect_w",
                "one_f",
                "zero_i8",
                "one_f",
                "zero_u8",
                "detect_b",
            ],
            ["detections"],
            kernel_shape=[KERNEL, KERNEL],
            # The generator caps the meaningful grid at 20x20. Five cells of
            # implicit bottom/right padding retain all top-left anchors for
            # 3x3 pots while keeping only an 18x18 detector tensor.
            pads=[0, 0, 5, 5],
        ),
        helper.make_node(
            "QLinearConv",
            [
                "detections",
                "one_f",
                "zero_u8",
                "fill_w",
                "one_f",
                "zero_u8",
                "one_f",
                "zero_u8",
            ],
            ["fill_u8"],
            kernel_shape=[FILL_KERNEL, FILL_KERNEL],
            pads=[FILL_KERNEL, FILL_KERNEL, 11, 11],
        ),
        helper.make_node("Cast", ["fill_u8"], ["fill"], to=TensorProto.BOOL),
        helper.make_node("Where", ["fill", "yellow", "input"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes,
        "task002_honeypots",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, GRID_SHAPE)],
        initializer=initializers,
        value_info=[
            helper.make_tensor_value_info("cropped", TensorProto.FLOAT, [1, 1, 20, 20]),
            helper.make_tensor_value_info("quantized", TensorProto.UINT8, [1, 1, 20, 20]),
            helper.make_tensor_value_info("detections", TensorProto.UINT8, [1, 36, 18, 18]),
            helper.make_tensor_value_info("fill_u8", TensorProto.UINT8, [1, 1, 30, 30]),
            helper.make_tensor_value_info("fill", TensorProto.BOOL, [1, 1, 30, 30]),
        ],
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 19)],
        producer_name="neurogolf-task002",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def as_tensor(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros(GRID_SHAPE, dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            result[0, color, row, col] = 1.0
    return result


def verify(model_path: Path) -> None:
    examples = json.loads(TASK_DATA.read_text())
    session = ort.InferenceSession(model_path.read_bytes(), providers=["CPUExecutionProvider"])
    passed = failed = 0
    failures: list[str] = []
    for subset, subset_examples in examples.items():
        for index, example in enumerate(subset_examples):
            actual = session.run(["output"], {"input": as_tensor(example["input"])})[0]
            actual = (actual > 0).astype(np.float32)
            expected = as_tensor(example["output"])
            if np.array_equal(actual, expected):
                passed += 1
            else:
                failed += 1
                failures.append(f"{subset}[{index}]")
    if failed:
        raise AssertionError(f"{failed} failures: {', '.join(failures[:10])}")
    print(f"verified {passed} examples (0 failures)")

    # Run the exact local copy of the official sanitization and cost metric.
    sys.path.insert(0, str(ROOT))
    from utils import neurogolf_utils

    sanitized = neurogolf_utils.sanitize_model(copy.deepcopy(onnx.load(model_path)))
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(model_path.with_suffix(""))
    profiled = ort.InferenceSession(sanitized.SerializeToString(), options)
    for example in sum(examples.values(), []):
        profiled.run(["output"], {"input": as_tensor(example["input"])})
    trace_path = profiled.end_profiling()
    memory, params = neurogolf_utils.score_network(sanitized, trace_path)
    Path(trace_path).unlink(missing_ok=True)
    points = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
    print(f"official metric: {memory} bytes + {params} params = {memory + params} cost")
    print(f"task points: {points:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    if not args.no_verify:
        verify(args.output)


if __name__ == "__main__":
    main()
