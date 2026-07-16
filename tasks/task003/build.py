#!/usr/bin/env python3
"""Build and verify the optimized ONNX model for NeuroGolf task 003."""

from __future__ import annotations

import copy
import itertools
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
DATA_PATH = REPO_ROOT / "kaggle_tasks_data" / "task003.json"
MODEL_PATH = TASK_DIR / "task003.onnx"
SUBMISSION_PATH = REPO_ROOT / "submission" / "task003.onnx"

sys.path.insert(0, str(REPO_ROOT / "utils"))
import neurogolf_utils  # noqa: E402


def build_model() -> onnx.ModelProto:
    """Return a single-Einsum recoloring and row-continuation solution.

    Channel vectors convert black/blue into a signed scalar and that scalar into
    black/red logits. Supplying the same 6x30 row-factor initializer twice forms
    a rank-four Gram matrix whose signs reproduce rows 0--8; zero columns after
    row eight suppress all padded output.
    """

    input_colors = np.zeros(10, dtype=np.float32)
    input_colors[0], input_colors[1] = -1, 1
    output_colors = np.zeros(10, dtype=np.float32)
    output_colors[0], output_colors[2] = -1, 1

    row_factors = np.zeros((4, 30), dtype=np.float32)
    # A low-rank shared factor found by optimize_rank_torch.py, then rounded to
    # small integers. Its minimum exhaustive signed margin is seven.
    row_factors[:, :9] = [
        [0, -3, -2, 3, -2, -8, 1, 1, -2],
        [-5, 3, 2, 1, -6, 0, 2, 1, -3],
        [4, -2, 8, 2, -2, 0, 6, -2, 4],
        [0, 5, 2, 6, 2, -1, 5, 9, -1],
    ]

    graph_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 10, 30, 30]
    )
    graph_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 10, 30, 30]
    )
    initializers = [
        helper.make_tensor("input_colors", TensorProto.FLOAT, [10], input_colors),
        helper.make_tensor("output_colors", TensorProto.FLOAT, [10], output_colors),
        helper.make_tensor("row_factors", TensorProto.FLOAT, [4, 30], row_factors.ravel()),
    ]
    einsum = helper.make_node(
        "Einsum",
        ["input", "input_colors", "output_colors", "row_factors", "row_factors"],
        ["output"],
        name="output",
        equation="birc,i,o,kr,kh->bohc",
    )
    graph = helper.make_graph(
        [einsum], "task003", [graph_input], [graph_output], initializers
    )
    return helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 12)],
    )


def legal_generator_examples() -> list[dict[str, list[list[int]]]]:
    """Enumerate the complete finite domain of generate() defaults."""

    examples = []
    for steps, count, flip in [(2, 3, 0), (2, 3, 1), (3, 4, 0), (3, 5, 0)]:
        for pixels in itertools.combinations(range(3 * steps), count):
            source = np.zeros((6, 3), dtype=np.int64)
            target = np.zeros((9, 3), dtype=np.int64)
            flipped = 0
            for offset in range(0, 9, steps):
                for pixel in pixels:
                    row, col = divmod(pixel, 3)
                    col = col if not flipped else 2 - col
                    if offset + row < 6:
                        source[offset + row, col] = 1
                    if offset + row < 9:
                        target[offset + row, col] = 2
                if flip:
                    flipped = 1 - flipped
            examples.append({"input": source.tolist(), "output": target.tolist()})
    assert len(examples) == 292
    return examples


def verify(model: onnx.ModelProto) -> tuple[int, int, float, int, int]:
    """Verify every supplied example and return official memory/parameter cost."""

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    assert [d.dim_value for d in inferred.graph.output[0].type.tensor_type.shape.dim] == [
        1,
        10,
        30,
        30,
    ]

    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(TASK_DIR / "task003_profile")
    sanitized = neurogolf_utils.sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    session = onnxruntime.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )

    with DATA_PATH.open() as handle:
        task = json.load(handle)
    supplied_examples = task["train"] + task["test"] + task["arc-gen"]
    exhaustive_examples = legal_generator_examples()
    examples = supplied_examples + exhaustive_examples
    failures = []
    minimum_true_logit = math.inf
    maximum_false_logit = -math.inf
    for index, example in enumerate(examples):
        benchmark = neurogolf_utils.convert_to_numpy(example)
        actual_logits = session.run(["output"], {"input": benchmark["input"]})[0]
        expected = benchmark["output"].astype(bool)
        actual = actual_logits > 0.0
        if not np.array_equal(actual, expected):
            failures.append(index)
        minimum_true_logit = min(minimum_true_logit, float(actual_logits[expected].min()))
        maximum_false_logit = max(maximum_false_logit, float(actual_logits[~expected].max()))

    trace_path = session.end_profiling()
    memory, parameters = neurogolf_utils.score_network(sanitized, trace_path)
    Path(trace_path).unlink(missing_ok=True)
    assert not failures, f"failed example indices: {failures[:10]}"
    assert minimum_true_logit > 0.0
    assert maximum_false_logit <= 0.0
    assert memory is not None and parameters is not None
    return (
        memory,
        parameters,
        25.0 - math.log(max(1.0, memory + parameters)),
        len(supplied_examples),
        len(exhaustive_examples),
    )


def main() -> None:
    model = build_model()
    memory, parameters, points, supplied_count, exhaustive_count = verify(model)
    onnx.save(model, MODEL_PATH)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MODEL_PATH, SUBMISSION_PATH)
    print(f"verified {supplied_count}/{supplied_count} supplied examples")
    print(f"verified {exhaustive_count}/{exhaustive_count} legal generator configurations")
    print(f"official cost: {memory} bytes + {parameters} parameters = {memory + parameters}")
    print(f"task score: {points:.6f}")
    print(f"model size: {MODEL_PATH.stat().st_size} bytes")
    print(f"wrote {MODEL_PATH}")
    print(f"copied {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
