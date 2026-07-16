#!/usr/bin/env python3
"""Build and verify an optimized NeuroGolf solver for task005 / 045e512c."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
TASK_DATA = ROOT / "kaggle_tasks_data" / "task005.json"
GENERATOR = ROOT / "ARC-GEN" / "tasks" / "task_045e512c.py"
WORK_MODEL = Path(__file__).with_name("task005.onnx")
SUBMISSION_MODEL = ROOT / "submission" / "task005.onnx"
SHAPE = (1, 10, 30, 30)
DIRS = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
]


def dense(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value), name=name)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    def init(name: str, value: np.ndarray) -> str:
        initializers.append(dense(name, value))
        return name

    # Encode a sampled foreground color c as the exact scalar 100+c.
    encoded_weight = np.zeros((1, 10, 1, 1), np.float32)
    encoded_weight[0, 1:, 0, 0] = [100.0 + color for color in range(1, 10)]
    init("encoded_weight", encoded_weight)

    # Channel zero alone gives occupancy by inversion.  Crop its only possible
    # anchor extent, average 3x3 background windows, then negate so the complete
    # sprite is the unique maximum.  This avoids a full 30x30 encoded plane.
    init("crop_starts", np.asarray([0, 6, 6], np.int64))
    init("crop_ends", np.asarray([1, 15, 15], np.int64))
    init("crop_axes", np.asarray([1, 2, 3], np.int64))
    nodes.append(helper.make_node("Slice", ["input", "crop_starts", "crop_ends", "crop_axes"], ["anchor_region"]))
    init("negative_window", -np.ones((1, 1, 3, 3), np.float32))
    nodes.append(helper.make_node("Conv", ["anchor_region", "negative_window"], ["candidates"]))
    nodes.append(
        helper.make_node(
            "MaxPool", ["candidates"], ["candidate_max", "anchor_index"], kernel_shape=[7, 7]
        )
    )

    seven = init("seven", np.asarray(7, np.int64))
    six = init("six", np.asarray(6, np.int64))
    nodes.append(helper.make_node("Div", ["anchor_index", seven], ["anchor_r0"]))
    nodes.append(helper.make_node("Mod", ["anchor_index", seven], ["anchor_c0"], fmod=0))
    nodes.append(helper.make_node("Add", ["anchor_r0", six], ["anchor_r"]))
    nodes.append(helper.make_node("Add", ["anchor_c0", six], ["anchor_c"]))

    # Sample all nine central cells.  A small generator-derived hitting set is
    # enough for each first-copy marker: one cell for cardinal directions and
    # two for diagonals.  Duplicating cardinal samples gives a regular 8x2
    # direction tensor while reducing 81 samples to 25.
    central_offsets = np.empty((1, 1, 9, 2), np.float32)
    for pixel, (pr, pc) in enumerate(np.ndindex(3, 3)):
        central_offsets[0, 0, pixel] = (2.0 * pc / 29.0, 2.0 * pr / 29.0)
    hit_pixels = {
        (-1, -1): [(1, 2), (2, 1)],
        (-1, 0): [(2, 1), (2, 1)],
        (-1, 1): [(1, 0), (2, 0)],
        (0, -1): [(1, 2), (1, 2)],
        (0, 1): [(1, 0), (1, 0)],
        (1, -1): [(0, 1), (0, 2)],
        (1, 0): [(0, 1), (0, 1)],
        (1, 1): [(0, 0), (0, 1)],
    }
    direction_offsets = np.empty((1, 8, 2, 2), np.float32)
    for direction, (dr, dc) in enumerate(DIRS):
        for sample, (pr, pc) in enumerate(hit_pixels[(dr, dc)]):
            direction_offsets[0, direction, sample] = (
                2.0 * (4 * dc + pc) / 29.0,
                2.0 * (4 * dr + pr) / 29.0,
            )
    init("central_offsets", central_offsets)
    init("direction_offsets", direction_offsets)
    init("two_over_29", np.asarray(2.0 / 29.0, np.float32))
    init("minus_one", np.asarray(-1.0, np.float32))
    nodes.append(helper.make_node("Concat", ["anchor_c", "anchor_r"], ["anchor_cr"], axis=3))
    nodes.append(helper.make_node("Cast", ["anchor_cr"], ["anchor_cr_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", ["anchor_cr_f", "two_over_29"], ["anchor_scaled"]))
    nodes.append(helper.make_node("Add", ["anchor_scaled", "minus_one"], ["anchor_norm"]))
    nodes.append(helper.make_node("Add", ["anchor_norm", "central_offsets"], ["central_grid"]))
    nodes.append(helper.make_node("Add", ["anchor_norm", "direction_offsets"], ["direction_grid"]))
    nodes.append(
        helper.make_node(
            "GridSample",
            ["input", "central_grid"],
            ["central_samples_raw"],
            align_corners=1,
            mode="nearest",
            padding_mode="zeros",
        )
    )
    nodes.append(
        helper.make_node(
            "GridSample",
            ["input", "direction_grid"],
            ["direction_samples_raw"],
            align_corners=1,
            mode="nearest",
            padding_mode="zeros",
        )
    )
    nodes.append(helper.make_node("Conv", ["central_samples_raw", "encoded_weight"], ["central_samples"]))
    nodes.append(helper.make_node("Conv", ["direction_samples_raw", "encoded_weight"], ["direction_samples"]))

    # Each object is monochrome, so its maximum is directly its exact code.
    nodes.append(helper.make_node("ReduceMax", ["central_samples"], ["center_code"], axes=[3], keepdims=0))
    nodes.append(helper.make_node("ReduceMax", ["direction_samples"], ["direction_codes"], axes=[3], keepdims=0))
    nodes.append(helper.make_node("Concat", ["center_code", "direction_codes"], ["object_code"], axis=2))

    # The nine cells of object zero are the complete sprite mask.
    axis2 = init("axis2", np.asarray([2], np.int64))
    nodes.append(helper.make_node("Squeeze", ["central_samples", axis2], ["sprite_code"]))
    nodes.append(helper.make_node("Cast", ["sprite_code"], ["sprite_values_i8"], to=TensorProto.INT8))
    nodes.append(helper.make_node("Sign", ["sprite_values_i8"], ["sprite_i8"]))

    # Render one central instance and three copies for each possible ray.
    instances = [(0, 0, 0, 0)]
    for obj, (dr, dc) in enumerate(DIRS, start=1):
        instances.extend((obj, dr, dc, step) for step in (1, 2, 3))
    selectors = init("selectors", np.asarray([obj for obj, _, _, _ in instances], np.int64))
    nodes.append(helper.make_node("Gather", ["object_code", selectors], ["instance_codes_3d"], axis=2))
    nodes.append(helper.make_node("Cast", ["instance_codes_3d"], ["instance_codes_i8"], to=TensorProto.INT8))
    nodes.append(helper.make_node("Transpose", ["instance_codes_i8"], ["instance_codes_column"], perm=[0, 2, 1]))
    nodes.append(helper.make_node("Mul", ["instance_codes_column", "sprite_i8"], ["updates_i8"]))

    # Flatten spatial coordinates so ScatterElements needs one index per
    # update instead of a two-component ScatterND index.  Any off-grid linear
    # index still wraps into the sentinel padding rather than the valid 21x21
    # cells.  A final Equal broadcasts the exact color codes.
    flat_offsets = np.empty((1, 1, 1, 225), np.int64)
    for inst, (_, dr, dc, step) in enumerate(instances):
        for pixel, (pr, pc) in enumerate(np.ndindex(3, 3)):
            flat_offsets[0, 0, 0, inst * 9 + pixel] = (
                30 * (4 * step * dr + pr) + 4 * step * dc + pc
            )
    init("flat_offsets", flat_offsets)
    init("thirty", np.asarray(30, np.int64))
    nodes.append(helper.make_node("Mul", ["anchor_r", "thirty"], ["anchor_row_flat"]))
    nodes.append(helper.make_node("Add", ["anchor_row_flat", "anchor_c"], ["anchor_flat"]))
    nodes.append(helper.make_node("Add", ["anchor_flat", "flat_offsets"], ["indices" ]))
    init("flat_updates_shape", np.asarray([1, 1, 1, 225], np.int64))
    nodes.append(helper.make_node("Reshape", ["updates_i8", "flat_updates_shape"], ["flat_updates"]))
    base = np.full((30, 30), 120, np.int8)
    base[:21, :21] = 100
    base = base.reshape(1, 1, 1, 900)
    init("base", base)
    nodes.append(helper.make_node("ScatterElements", ["base", "indices", "flat_updates"], ["color_flat"], axis=3, reduction="max"))
    init("grid_shape", np.asarray([30, 30], np.int64))
    nodes.append(helper.make_node("Reshape", ["color_flat", "grid_shape"], ["color_grid"]))
    init("color_codes", np.arange(100, 110, dtype=np.int8).reshape(1, 10, 1, 1))
    nodes.append(helper.make_node("Equal", ["color_grid", "color_codes"], ["output"]))

    graph = helper.make_graph(
        nodes,
        "task005_stamp",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, SHAPE)],
        initializers,
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 16)],
        producer_name="neurogolf-task005",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def one_hot(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros(SHAPE, np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            result[0, color, row, col] = 1.0
    return result


def check_example(session: ort.InferenceSession, example: dict) -> bool:
    actual = session.run(["output"], {"input": one_hot(example["input"])})[0] > 0
    return bool(np.array_equal(actual, one_hot(example["output"]).astype(bool)))


def load_generator():
    sys.path.insert(0, str(ROOT / "ARC-GEN"))
    spec = importlib.util.spec_from_file_location("task_045e512c", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def verify(path: Path, random_cases: int) -> tuple[int, int]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(path.read_bytes(), options, providers=["CPUExecutionProvider"])
    passed = total = 0
    examples = json.loads(TASK_DATA.read_text())
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[split]):
            total += 1
            if not check_example(session, example):
                raise AssertionError(f"failed {split} example {index}")
            passed += 1
    if random_cases:
        generator = load_generator()
        for index in range(random_cases):
            example = generator.generate()
            total += 1
            if not check_example(session, example):
                raise AssertionError(f"failed random generator example {index}")
            passed += 1
    return passed, total


def official_cost(path: Path) -> tuple[int, int, float]:
    sys.path.insert(0, str(ROOT))
    from utils.neurogolf_utils import sanitize_model, score_network

    model = sanitize_model(onnx.load(path))
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(path.parent / "task005-profile")
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    example = json.loads(TASK_DATA.read_text())["arc-gen"][0]
    session.run(["output"], {"input": one_hot(example["input"])})
    trace_path = Path(session.end_profiling())
    memory, params = score_network(model, trace_path)
    trace_path.unlink(missing_ok=True)
    if memory is None or params is None:
        raise RuntimeError("official scorer rejected task005.onnx")
    return memory, params, max(1.0, 25.0 - math.log(max(1.0, memory + params)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=1000)
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--output", type=Path, default=WORK_MODEL)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), args.output)
    passed, total = verify(args.output, args.random_cases)
    memory, params, points = official_cost(args.output)
    if not args.no_copy:
        SUBMISSION_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, SUBMISSION_MODEL)
    print(f"verified: {passed}/{total}")
    print(f"official cost: {memory} bytes + {params} params = {memory + params}")
    print(f"official score: {points:.6f}")
    print(f"model: {args.output}")
    if not args.no_copy:
        print(f"submission: {SUBMISSION_MODEL}")


if __name__ == "__main__":
    main()
