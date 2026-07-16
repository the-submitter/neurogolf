#!/usr/bin/env python3
"""Build and verify the optimized task007 (diagstripes / ARC 05269061) model."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[2]
TASK_DATA = ROOT / "kaggle_tasks_data" / "task007.json"
MODEL_PATH = Path(__file__).with_name("task007.onnx")
SUBMISSION_PATH = ROOT / "submission" / "task007.onnx"
GRID_SHAPE = (1, 10, 30, 30)


def tensor(name: str, values: np.ndarray) -> onnx.TensorProto:
    values = np.asarray(values, dtype=np.float32)
    return helper.make_tensor(name, TensorProto.FLOAT, values.shape, values.ravel())


def build_model() -> onnx.ModelProto:
    # The three residue classes use vectors e0=(1, 0), e1=(0, 1), and
    # e2=(-1, -1).  Equal vectors have positive dot products; unequal vectors
    # have dot products 0 or -1, exactly matching the scorer's > 0 threshold.
    coordinate_vectors = np.zeros((30, 2), dtype=np.float32)
    vectors = ((1.0, 0.0), (0.0, 1.0), (-1.0, -1.0))
    for coordinate in range(7):
        coordinate_vectors[coordinate] = vectors[coordinate % 3]

    # Bilinear multiplication in R[x]/(x^2+x+1).  It maps the coordinate
    # vectors for a and b to the vector for (a+b) mod 3.  Reusing this tensor
    # on both sides lets one Einsum compare input and output diagonals.
    angle_add = np.zeros((2, 2, 2), dtype=np.float32)
    angle_add[0, 0, 0] = 1.0
    angle_add[0, 1, 1] = 1.0
    angle_add[1, 0, 1] = 1.0
    angle_add[1, 1] = (-1.0, -1.0)

    # Sequential safe names make the official sanitizer idempotent.
    initializers = [
        tensor("safe_name_0", np.array([0.0] + [1.0] * 9)),
        tensor("safe_name_1", coordinate_vectors),
        tensor("safe_name_2", angle_add),
    ]

    node = helper.make_node(
        "Einsum",
        [
            "input",
            "safe_name_0",
            "safe_name_1",
            "safe_name_1",
            "safe_name_1",
            "safe_name_1",
            "safe_name_2",
            "safe_name_2",
        ],
        ["output"],
        equation="nkij,k,ia,jb,ru,cv,abq,uvq->nkrc",
        name="output",
    )
    graph = helper.make_graph(
        [node],
        "task007_diagstripes",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, GRID_SHAPE)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-task007",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def one_hot(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros(GRID_SHAPE, dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            result[0, color, row, col] = 1.0
    return result


def verify(model: onnx.ModelProto) -> tuple[int, int]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    examples = json.loads(TASK_DATA.read_text())
    passed = total = 0
    for subset in examples.values():
        for example in subset:
            actual = session.run(["output"], {"input": one_hot(example["input"])})[0]
            expected = one_hot(example["output"])
            passed += int(np.array_equal(actual > 0.0, expected > 0.0))
            total += 1
    return passed, total


if __name__ == "__main__":
    model = build_model()
    passed, total = verify(model)
    if passed != total:
        raise SystemExit(f"Verification failed: {passed}/{total} examples passed")

    onnx.save(model, MODEL_PATH)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MODEL_PATH, SUBMISSION_PATH)
    # There are no scored intermediate outputs in this one-node graph.
    params = sum(
        math.prod(initializer.dims) for initializer in model.graph.initializer
    )
    score = max(1.0, 25.0 - math.log(max(1.0, params)))
    print(f"Verified {passed}/{total} examples")
    print(f"Saved {MODEL_PATH.relative_to(ROOT)} ({MODEL_PATH.stat().st_size} bytes)")
    print(f"Copied {SUBMISSION_PATH.relative_to(ROOT)}")
    print(f"Official cost: 0 bytes + {params} params = {params}")
    print(f"Official score: {score:.6f}")
