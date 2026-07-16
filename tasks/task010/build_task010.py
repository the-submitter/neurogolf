#!/usr/bin/env python3
"""Build and exhaustively verify the NeuroGolf task010 ONNX model."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[2]
TASK_DATA = ROOT / "kaggle_tasks_data" / "task010.json"
WORK_MODEL = Path(__file__).with_name("task010.onnx")
SUBMISSION_MODEL = ROOT / "submission" / "task010.onnx"
SHAPE = (1, 10, 30, 30)


def tensor(name: str, dtype: int, dims: list[int], values: list[int | float]):
    return helper.make_tensor(name, dtype, dims, values)


def build_model() -> onnx.ModelProto:
    """Return a compact shared height-ranking and routing graph.

    A rank-4 spatial factorization is the identity at the four possible bar
    columns and a partition of unity everywhere else.  The first Einsum uses
    it to read the four heights.  TopK ranks them, Gather permutes the same
    spatial factors into rank order, and the last Einsum routes pixels.

    The channel code is .5-(output-desired)^2.  For gray input the desired
    color is rank+1; for black input it is zero.  This has an exact three-term
    factorization.  Because the spatial rows sum to one, the black score is
    preserved without a separate background/sentinel route.  NeuroGolf's
    strict >0 threshold makes the sign code exactly one-hot.
    """
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    # The four rows select the four possible bar columns. Non-bar columns can
    # use any partition of unity; assigning them to row zero is simplest.
    spatial_left = np.eye(4, dtype=np.float32)
    spatial_right = np.zeros((4, 30), dtype=np.float32)
    spatial_right[0, :] = 1.0
    for bar, column in enumerate((1, 3, 5, 7)):
        spatial_right[:, column] = 0.0
        spatial_right[bar, column] = 1.0

    # For black input g=0 and the desired color is zero.  For gray input g=1
    # and it is the rank color q.  Expanding .5-(o-g*q)^2 gives three products.
    rank_colors = np.asarray([1, 2, 3, 4], dtype=np.float32)
    output_colors = np.arange(10, dtype=np.float32)
    # Share a two-column [constant, is-gray] input basis between both Einsums.
    # term_mix expands it to the three factors needed by the channel code.
    input_basis = np.zeros((10, 2), dtype=np.float32)
    input_basis[:, 0] = 1.0
    input_basis[5, 1] = 1.0
    term_mix = np.asarray([[1, 0, 0], [0, 1, 1]], dtype=np.float32)
    rank_factor = np.stack(
        [np.ones(4), rank_colors, rank_colors**2], axis=1
    ).astype(np.float32)
    output_factor = np.stack(
        [0.5 - output_colors**2, 2 * output_colors, -np.ones(10)], axis=0
    ).astype(np.float32)

    initializers = [
        tensor("topk_k", TensorProto.INT64, [1], [4]),
        tensor("height_mix", TensorProto.FLOAT, [2], [0, 1]),
        tensor("spatial_left", TensorProto.FLOAT, [4, 4], spatial_left.reshape(-1).tolist()),
        tensor("spatial_right", TensorProto.FLOAT, [4, 30], spatial_right.reshape(-1).tolist()),
        tensor("input_basis", TensorProto.FLOAT, [10, 2], input_basis.reshape(-1).tolist()),
        tensor("term_mix", TensorProto.FLOAT, [2, 3], term_mix.reshape(-1).tolist()),
        tensor("rank_factor", TensorProto.FLOAT, [4, 3], rank_factor.reshape(-1).tolist()),
        tensor("output_factor", TensorProto.FLOAT, [3, 10], output_factor.reshape(-1).tolist()),
    ]

    nodes = [
        helper.make_node(
            "Einsum",
            ["input", "input_basis", "height_mix", "spatial_left", "spatial_right"],
            ["rank_input"],
            equation="bihw,iz,z,kt,tw->k",
        ),
        helper.make_node(
            "TopK", ["rank_input", "topk_k"], ["sorted_heights", "rank_indices"], axis=0
        ),
        helper.make_node(
            "Gather", ["spatial_left", "rank_indices"], ["ranked_spatial_left"], axis=0
        ),
        helper.make_node(
            "Einsum",
            [
                "input", "ranked_spatial_left", "spatial_right",
                "input_basis", "term_mix", "rank_factor", "output_factor",
            ],
            ["output"],
            equation="bihw,pt,tw,iz,za,pa,ao->bohw",
        ),
    ]

    value_info = [
        helper.make_tensor_value_info("rank_input", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("sorted_heights", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("rank_indices", TensorProto.INT64, [4]),
        helper.make_tensor_value_info("ranked_spatial_left", TensorProto.FLOAT, [4, 4]),
    ]
    graph = helper.make_graph(
        nodes,
        "task010_bar_height_ranking",
        [input_info],
        [output_info],
        initializer=initializers,
        value_info=value_info,
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 12)],
        producer_name="neurogolf-task010",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def one_hot(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros(SHAPE, dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            result[0, color, row, col] = 1.0
    return result


def expected_case(heights: tuple[int, ...], order: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    source = np.zeros(SHAPE, dtype=np.float32)
    target = np.zeros(SHAPE, dtype=np.float32)
    source[0, 0, :9, :9] = 1.0
    target[0, 0, :9, :9] = 1.0
    for rank, height in enumerate(heights):
        col = order[rank] * 2 + 1
        source[0, 0, 9 - height : 9, col] = 0.0
        source[0, 5, 9 - height : 9, col] = 1.0
        target[0, 0, 9 - height : 9, col] = 0.0
        target[0, rank + 1, 9 - height : 9, col] = 1.0
    return source, target


def verify(path: Path, exhaustive: bool) -> tuple[int, int]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(path.read_bytes(), options, providers=["CPUExecutionProvider"])

    examples = json.loads(TASK_DATA.read_text())
    passed = total = 0
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[split]):
            total += 1
            actual = session.run(["output"], {"input": one_hot(example["input"])})[0] > 0
            expected = one_hot(example["output"]).astype(bool)
            if not np.array_equal(actual, expected):
                raise AssertionError(f"failed {split}[{index}]")
            passed += 1

    if exhaustive:
        for chosen in itertools.combinations(range(1, 10), 4):
            heights = tuple(sorted(chosen, reverse=True))
            for order in itertools.permutations(range(4)):
                source, target = expected_case(heights, order)
                actual = session.run(["output"], {"input": source})[0] > 0
                if not np.array_equal(actual, target.astype(bool)):
                    raise AssertionError(f"failed generated heights={heights}, order={order}")
        print("exhaustive generator verification: 3024/3024")
    return passed, total


def official_cost(path: Path) -> tuple[int, int, float]:
    sys.path.insert(0, str(ROOT))
    from utils.neurogolf_utils import sanitize_model, score_network

    model = sanitize_model(onnx.load(path))
    if model is None:
        raise RuntimeError("official sanitizer rejected model")
    with tempfile.TemporaryDirectory(prefix="task010-profile-") as profile_dir:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.profile_file_prefix = str(Path(profile_dir) / "trace")
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        examples = json.loads(TASK_DATA.read_text())
        for split in ("train", "test", "arc-gen"):
            for example in examples[split]:
                session.run(["output"], {"input": one_hot(example["input"])})
        memory, params = score_network(model, session.end_profiling())
    if memory is None or params is None:
        raise RuntimeError("official scorer rejected model")
    return memory, params, max(1.0, 25.0 - math.log(max(1.0, memory + params)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--no-exhaustive", action="store_true")
    args = parser.parse_args()

    WORK_MODEL.parent.mkdir(parents=True, exist_ok=True)
    SUBMISSION_MODEL.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), WORK_MODEL)
    passed, total = verify(WORK_MODEL, not args.no_exhaustive)
    memory, params, points = official_cost(WORK_MODEL)
    if not args.no_copy:
        shutil.copy2(WORK_MODEL, SUBMISSION_MODEL)

    print(f"packaged verification: {passed}/{total}")
    print(f"official cost: {memory} bytes + {params} params = {memory + params}")
    print(f"estimated points: {points:.6f}")
    print(f"model: {WORK_MODEL} ({WORK_MODEL.stat().st_size} bytes)")
    if not args.no_copy:
        print(f"submission: {SUBMISSION_MODEL}")


if __name__ == "__main__":
    main()
