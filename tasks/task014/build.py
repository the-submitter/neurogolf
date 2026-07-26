#!/usr/bin/env python3
"""Build and verify NeuroGolf task014 / ARC-GEN 0b148d64."""

from __future__ import annotations

import copy
import hashlib
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
from onnx import TensorProto, helper, numpy_helper


TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parents[1]
DATA_PATH = ROOT / "kaggle_tasks_data" / "task014.json"
GENERATOR_PATH = ROOT / "ARC-GEN" / "tasks" / "task_0b148d64.py"
MODEL_PATH = TASK_DIR / "task014.onnx"
SUBMISSION_PATH = ROOT / "submission" / "task014.onnx"
GRID_SHAPE = [1, 10, 30, 30]

sys.path.insert(0, str(ROOT / "utils"))
import neurogolf_utils  # noqa: E402


def _initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value), name)


def _value(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def build_model() -> onnx.ModelProto:
    """Return a generator-general rare-color crop graph."""

    initializers = [
        # Channel zero is deliberately ineligible; every foreground channel
        # is eligible exactly when its count is positive.
        _initializer(
            "eligibility_threshold",
            np.asarray([1000.0] + [0.0] * 9, dtype=np.float32),
        ),
        _initializer("large_count", np.asarray(1000.0, dtype=np.float32)),
        _initializer("zero_float", np.asarray(0.0, dtype=np.float32)),
        _initializer("one_i64", np.asarray([1], dtype=np.int64)),
        _initializer("thirty_i64", np.asarray([30], dtype=np.int64)),
        _initializer("slice_axes", np.asarray([1, 2, 3], dtype=np.int64)),
        _initializer("zero_i64", np.asarray([0], dtype=np.int64)),
        _initializer("x_scale_base", np.asarray(0.5, dtype=np.float32)),
        _initializer("quant_zero", np.asarray(0, dtype=np.uint8)),
        _initializer("conv_x_zero", np.asarray(1, dtype=np.uint8)),
        # Quantized states are crop background=0, outside=1, rare color=2.
        _initializer("outside_one", np.asarray(1, dtype=np.uint8)),
        _initializer(
            "base_weight",
            np.asarray([-1] + [0] * 9, dtype=np.int8).reshape(10, 1, 1, 1),
        ),
        _initializer("rare_update", np.ones((1, 1, 1, 1), dtype=np.int8)),
        _initializer("w_scale", np.asarray(2.0, dtype=np.float32)),
        _initializer("w_zero", np.asarray(0, dtype=np.int8)),
        _initializer("y_scale", np.asarray(1.0, dtype=np.float32)),
        _initializer("y_zero", np.asarray(0, dtype=np.uint8)),
        _initializer("onehot_depth", np.asarray(10, dtype=np.int64)),
        _initializer("onehot_values", np.asarray([0.0, 1.0], dtype=np.float32)),
    ]

    nodes = [
        helper.make_node(
            "ReduceSum", ["input"], ["counts"], axes=[2, 3], keepdims=0
        ),
        helper.make_node(
            "Greater", ["counts", "eligibility_threshold"], ["eligible"]
        ),
        helper.make_node(
            "Where", ["eligible", "counts", "large_count"], ["candidate_counts"]
        ),
        helper.make_node(
            "ArgMin", ["candidate_counts"], ["rare"], axis=1, keepdims=0
        ),
        helper.make_node(
            "OneHot",
            ["rare", "onehot_depth", "onehot_values"],
            ["rare_selector_row"],
            axis=1,
        ),
        # Find the exact occupied row and column bounds of the rare color.
        helper.make_node(
            "Einsum", ["input", "rare_selector_row"], ["rare_row_counts"], equation="nchw,nc->nh"
        ),
        helper.make_node(
            "Greater", ["rare_row_counts", "zero_float"], ["rare_rows_bool"]
        ),
        helper.make_node("Cast", ["rare_rows_bool"], ["rare_rows"], to=TensorProto.UINT8),
        helper.make_node(
            "ArgMax",
            ["rare_rows"],
            ["row_start"],
            axis=1,
            keepdims=0,
            select_last_index=0,
        ),
        helper.make_node(
            "ArgMax",
            ["rare_rows"],
            ["row_last"],
            axis=1,
            keepdims=0,
            select_last_index=1,
        ),
        helper.make_node(
            "Einsum", ["input", "rare_selector_row"], ["rare_col_counts"], equation="nchw,nc->nw"
        ),
        helper.make_node(
            "Greater", ["rare_col_counts", "zero_float"], ["rare_cols_bool"]
        ),
        helper.make_node("Cast", ["rare_cols_bool"], ["rare_cols"], to=TensorProto.UINT8),
        helper.make_node(
            "ArgMax",
            ["rare_cols"],
            ["col_start"],
            axis=1,
            keepdims=0,
            select_last_index=0,
        ),
        helper.make_node(
            "ArgMax",
            ["rare_cols"],
            ["col_last"],
            axis=1,
            keepdims=0,
            select_last_index=1,
        ),
        helper.make_node("Add", ["rare", "one_i64"], ["rare_end"]),
        helper.make_node("Add", ["row_last", "one_i64"], ["row_end"]),
        helper.make_node("Add", ["col_last", "one_i64"], ["col_end"]),
        helper.make_node(
            "Concat", ["rare", "row_start", "col_start"], ["crop_starts"], axis=0
        ),
        helper.make_node(
            "Concat", ["rare_end", "row_end", "col_end"], ["crop_ends"], axis=0
        ),
        # Slice channel and space together. The actual tensor is at most one
        # generator quadrant rather than a full 30x30 channel plane.
        helper.make_node(
            "Slice",
            ["input", "crop_starts", "crop_ends", "slice_axes"],
            ["crop4"],
        ),
        helper.make_node(
            "QuantizeLinear", ["crop4", "x_scale_base", "quant_zero"], ["crop_quant"]
        ),
        helper.make_node("Sub", ["row_end", "row_start"], ["crop_height"]),
        helper.make_node("Sub", ["col_end", "col_start"], ["crop_width"]),
        helper.make_node("Sub", ["thirty_i64", "crop_height"], ["pad_rows"]),
        helper.make_node("Sub", ["thirty_i64", "crop_width"], ["pad_cols"]),
        helper.make_node(
            "Concat",
            [
                "zero_i64", "zero_i64", "zero_i64", "zero_i64",
                "zero_i64", "zero_i64", "pad_rows", "pad_cols",
            ],
            ["pads"],
            axis=0,
        ),
        helper.make_node(
            "Pad", ["crop_quant", "pads", "outside_one"], ["padded_crop"], mode="constant"
        ),
        helper.make_node("Unsqueeze", ["rare"], ["rare_weight_index"], axes=[1, 2, 3]),
        helper.make_node(
            "ScatterElements",
            ["base_weight", "rare_weight_index", "rare_update"],
            ["output_weight"],
            axis=0,
        ),
        helper.make_node(
            "QLinearConv",
            [
                "padded_crop", "x_scale_base", "conv_x_zero",
                "output_weight", "w_scale", "w_zero",
                "y_scale", "y_zero",
            ],
            ["output"],
            kernel_shape=[1, 1],
        ),
    ]

    # Dynamic Slice dimensions cannot be inferred statically. Positive
    # one-cell placeholders satisfy the official static-shape audit; runtime
    # profiling replaces their memory by the maximum actual crop area.
    value_info = [
        _value("crop4", TensorProto.FLOAT, [1, 1, 1, 1]),
        _value("crop_quant", TensorProto.UINT8, [1, 1, 1, 1]),
        _value("padded_crop", TensorProto.UINT8, [1, 1, 30, 30]),
    ]

    graph = helper.make_graph(
        nodes,
        "task014_minstatic",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, GRID_SHAPE)],
        initializers,
        value_info=value_info,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-task014",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def _session(model: onnx.ModelProto, profiling: bool = False) -> ort.InferenceSession:
    sanitized = neurogolf_utils.sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_profiling = profiling
    if profiling:
        options.profile_file_prefix = str(TASK_DIR / "profile")
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def _check_example(session: ort.InferenceSession, example: dict, label: str) -> None:
    benchmark = neurogolf_utils.convert_to_numpy(example)
    assert benchmark is not None
    logits = session.run(["output"], {"input": benchmark["input"]})[0]
    actual = logits > 0.0
    expected = benchmark["output"].astype(bool)
    if not np.array_equal(actual, expected):
        mismatch = np.argwhere(actual != expected)[0].tolist()
        raise AssertionError(f"{label} failed at {mismatch}")


def verify(model: onnx.ModelProto) -> tuple[int, int, int, float]:
    """Verify packaged examples and return official cost statistics."""

    session = _session(model, profiling=True)
    examples_by_subset = json.loads(DATA_PATH.read_text())
    examples = [example for subset in examples_by_subset.values() for example in subset]
    for index, example in enumerate(examples):
        _check_example(session, example, f"packaged example {index}")

    trace_path = session.end_profiling()
    sanitized = neurogolf_utils.sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    memory, parameters = neurogolf_utils.score_network(sanitized, trace_path)
    Path(trace_path).unlink(missing_ok=True)
    assert memory is not None and parameters is not None
    score = max(1.0, 25.0 - math.log(max(1.0, memory + parameters)))
    return len(examples), memory, parameters, score


def stress_verify(model: onnx.ModelProto, trials: int = 2_000) -> int:
    """Check fresh samples from the default ARC-GEN distribution."""

    sys.path.insert(0, str(ROOT / "ARC-GEN"))
    spec = importlib.util.spec_from_file_location("task014_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    session = _session(model)
    random.seed(14_026)
    for index in range(trials):
        _check_example(session, generator.generate(), f"fresh example {index}")
    return trials


def main() -> None:
    model = build_model()
    packaged, memory, parameters, score = verify(model)
    stressed = stress_verify(model)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, MODEL_PATH)
    shutil.copyfile(MODEL_PATH, SUBMISSION_PATH)
    assert MODEL_PATH.read_bytes() == SUBMISSION_PATH.read_bytes()
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    print(f"validated {packaged} packaged + {stressed} fresh examples")
    print(f"memory={memory} params={parameters} cost={memory + parameters}")
    print(f"score={score:.12f}")
    print(f"size={MODEL_PATH.stat().st_size} sha256={digest}")
    print(f"saved {MODEL_PATH}")
    print(f"saved {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
