#!/usr/bin/env python3
"""Build and verify the compact ONNX solution for NeuroGolf task 015."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import shutil
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = pathlib.Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "tasks" / "task015"
TASK_DATA = ROOT / "kaggle_tasks_data" / "task015.json"
TASK_MODEL = TASK_DIR / "task015.onnx"
SUBMISSION_MODEL = ROOT / "submission" / "task015.onnx"

sys.path.insert(0, str(ROOT / "utils"))
import neurogolf_utils as ng  # noqa: E402


def build_model() -> onnx.ModelProto:
    """Return a one-node factored linear transform implementing ``twinkle``."""
    # E embeds/extracts the fixed 9x9 generator grid in the 30x30 canvas.
    coordinate = np.zeros((30, 9), dtype=np.float32)
    coordinate[:9, :] = np.eye(9, dtype=np.float32)

    # Local spatial operator 0 is identity.  Operator 1 is the symmetric
    # one-cell shift.  The generator keeps twinklers off the boundary, so no
    # out-of-grid special case is needed.
    spatial = np.zeros((2, 9, 9), dtype=np.float32)
    spatial[0] = np.eye(9, dtype=np.float32)
    for source in range(9):
        if source > 0:
            spatial[1, source - 1, source] = 1.0
        if source < 8:
            spatial[1, source + 1, source] = 1.0

    # Seven rank-one channel terms: five preserve the only input colors used
    # by generate(), one paints/suppresses the rook arms, and one does the
    # same for the bishop diagonals.
    output_factor = np.zeros((10, 7), dtype=np.float32)
    input_factor = np.zeros((10, 7), dtype=np.float32)
    spatial_terms = np.zeros((7, 2, 2), dtype=np.float32)
    for term, channel in enumerate((0, 1, 2, 6, 8)):
        output_factor[channel, term] = 1.0
        input_factor[channel, term] = 1.0
        spatial_terms[term, 0, 0] = 1.0

    output_factor[7, 5] = 1.0
    output_factor[0, 5] = -1.0
    input_factor[1, 5] = 1.0
    spatial_terms[5, 1, 0] = 1.0
    spatial_terms[5, 0, 1] = 1.0

    output_factor[4, 6] = 1.0
    output_factor[0, 6] = -1.0
    input_factor[2, 6] = 1.0
    spatial_terms[6, 1, 1] = 1.0

    initializers = [
        numpy_helper.from_array(coordinate, "safe_name_0"),
        numpy_helper.from_array(output_factor, "safe_name_1"),
        numpy_helper.from_array(input_factor, "safe_name_2"),
        numpy_helper.from_array(spatial_terms, "safe_name_3"),
        numpy_helper.from_array(spatial, "safe_name_4"),
    ]

    tensor_shape = [1, 10, 30, 30]
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Einsum",
                [
                    "input",
                    "safe_name_0", "safe_name_0",
                    "safe_name_0", "safe_name_0",
                    "safe_name_1", "safe_name_2", "safe_name_3",
                    "safe_name_4", "safe_name_4",
                ],
                ["output"],
                equation="nirc,rx,cy,du,ev,ot,it,tab,aux,bvy->node",
            )
        ],
        name="twinkle",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, tensor_shape)],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, tensor_shape)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def verify(model_path: pathlib.Path) -> tuple[int, int, int, int, float]:
    """Run every packaged example and return official cost information."""
    with TASK_DATA.open() as stream:
        examples = json.load(stream)

    sanitized = ng.sanitize_model(onnx.load(model_path))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected the model")

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    with tempfile.TemporaryDirectory(prefix="task015-profile-") as profile_dir:
        options.profile_file_prefix = str(pathlib.Path(profile_dir) / "profile")
        session = ort.InferenceSession(sanitized.SerializeToString(), options)

        total = passed = 0
        for subset in ("train", "test", "arc-gen"):
            for example in examples[subset]:
                benchmark = ng.convert_to_numpy(example)
                if benchmark is None:
                    continue
                actual = ng.run_network(session, benchmark["input"])
                if not np.array_equal(actual, benchmark["output"]):
                    raise AssertionError(f"failed packaged {subset} example {total}")
                passed += 1
                total += 1

        trace_path = session.end_profiling()
        memory, params = ng.score_network(sanitized, trace_path)

    if memory is None or params is None:
        raise RuntimeError("official scorer rejected the model")
    cost = memory + params
    points = max(1.0, 25.0 - math.log(max(1.0, cost)))
    return passed, total, memory, params, points


def verify_structured(model_path: pathlib.Path) -> int:
    """Check every valid single-star placement, including all four colors."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    checked = 0
    for color in (1, 2, 6, 8):
        margin = 1 if color in (1, 2) else 0
        for row in range(margin, 9 - margin):
            for col in range(margin, 9 - margin):
                grid = [[0] * 9 for _ in range(9)]
                grid[row][col] = color
                expected = [line[:] for line in grid]
                offsets = ()
                paint = 0
                if color == 1:
                    offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
                    paint = 7
                elif color == 2:
                    offsets = ((-1, -1), (-1, 1), (1, -1), (1, 1))
                    paint = 4
                for dr, dc in offsets:
                    expected[row + dr][col + dc] = paint

                example = {"input": grid, "output": expected}
                benchmark = ng.convert_to_numpy(example)
                actual = ng.run_network(session, benchmark["input"])
                if not np.array_equal(actual, benchmark["output"]):
                    raise AssertionError(
                        f"failed structured color={color}, row={row}, col={col}"
                    )
                checked += 1
    return checked


def main() -> None:
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_MODEL.parent.mkdir(parents=True, exist_ok=True)

    model = build_model()
    onnx.save(model, TASK_MODEL)
    shutil.copyfile(TASK_MODEL, SUBMISSION_MODEL)

    passed, total, memory, params, points = verify(TASK_MODEL)
    structured = verify_structured(TASK_MODEL)
    if TASK_MODEL.read_bytes() != SUBMISSION_MODEL.read_bytes():
        raise AssertionError("task and submission model copies differ")

    digest = hashlib.sha256(TASK_MODEL.read_bytes()).hexdigest()
    print(f"packaged examples: {passed}/{total}")
    print(f"structured examples: {structured}/{structured}")
    print(f"memory: {memory} bytes")
    print(f"parameters: {params}")
    print(f"objective: {memory + params}")
    print(f"points: {points:.12f}")
    print(f"model size: {TASK_MODEL.stat().st_size} bytes")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
