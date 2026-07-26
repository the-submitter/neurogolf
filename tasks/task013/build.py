#!/usr/bin/env python3
"""Build and verify NeuroGolf task013 / ARC-GEN 0a938d79."""

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
DATA_PATH = ROOT / "kaggle_tasks_data" / "task013.json"
GENERATOR_PATH = ROOT / "ARC-GEN" / "tasks" / "task_0a938d79.py"
MODEL_PATH = TASK_DIR / "task013.onnx"
SUBMISSION_PATH = ROOT / "submission" / "task013.onnx"
GRID_SHAPE = (1, 10, 30, 30)

sys.path.insert(0, str(ROOT / "utils"))
import neurogolf_utils  # noqa: E402


def _initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value), name)


def _value(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def build_model() -> onnx.ModelProto:
    """Return a compact, generator-general implementation of the columns rule."""

    initializers = [
        _initializer("foreground", np.asarray([0.0] + [1.0] * 9, dtype=np.float32)),
        _initializer("position", np.arange(30, dtype=np.int8)[None, :]),
        _initializer("zero_i8", np.asarray(0, dtype=np.int8)),
        _initializer("one_i8", np.asarray(1, dtype=np.int8)),
        _initializer("axis_one", np.asarray([1], dtype=np.int64)),
        _initializer("horizontal_row", np.asarray([True, False])[:, None, None]),
        _initializer("vertical_row", np.asarray([False, True])[:, None, None]),
        # Coordinate 13 is outside every 6..12 short side and inside every
        # 20..30 long side.  Together with coordinate zero it gates the two
        # orientations inside the terminal contraction without intermediates.
        _initializer(
            "probe_coordinate",
            np.eye(30, dtype=np.float32)[[0, 13]],
        ),
        _initializer("probe_row_route", np.eye(2, dtype=np.float32)),
        _initializer("probe_col_route", np.eye(2, dtype=np.float32)[::-1]),
        # Matching signs select background for component zero and foreground
        # for the two colored components. Cross-group logits are negative.
        _initializer("component_sign", np.asarray([1.0, -1.0, -1.0], np.float32)),
        _initializer("channel_sign", np.asarray([1.0] + [-1.0] * 9, np.float32)),
    ]

    nodes = [
        # Color-valued seed coordinates on the two possible long axes.
        helper.make_node(
            "Einsum", ["input", "foreground"], ["h_color"], equation="nchw,c->nw"
        ),
        helper.make_node(
            "Einsum", ["input", "foreground"], ["v_color"], equation="nchw,c->nh"
        ),
        # The first and last nonzero coordinate are the two seed positions.
        helper.make_node(
            "ArgMax",
            ["h_color"],
            ["h_first"],
            axis=1,
            keepdims=1,
            select_last_index=0,
        ),
        helper.make_node(
            "ArgMax",
            ["v_color"],
            ["v_first"],
            axis=1,
            keepdims=1,
            select_last_index=0,
        ),
        helper.make_node("Concat", ["h_first", "v_first"], ["first"], axis=0),
        helper.make_node(
            "ArgMax",
            ["h_color"],
            ["h_second"],
            axis=1,
            keepdims=1,
            select_last_index=1,
        ),
        helper.make_node(
            "ArgMax",
            ["v_color"],
            ["v_second"],
            axis=1,
            keepdims=1,
            select_last_index=1,
        ),
        helper.make_node("Concat", ["h_second", "v_second"], ["second"], axis=0),
        helper.make_node("Cast", ["first"], ["first_i8"], to=TensorProto.INT8),
        helper.make_node("Cast", ["second"], ["second_i8"], to=TensorProto.INT8),
        helper.make_node("Sub", ["second_i8", "first_i8"], ["raw_spacing"]),
        # The unused orientation can collapse both seeds to one coordinate;
        # clamp that branch to avoid a zero divisor before it is gated away.
        helper.make_node("Max", ["raw_spacing", "one_i8"], ["spacing"]),
        helper.make_node("Add", ["spacing", "spacing"], ["period"]),
        helper.make_node("Sub", ["position", "first_i8"], ["offset"]),
        helper.make_node("Mod", ["offset", "period"], ["phase"], fmod=0),
        helper.make_node("GreaterOrEqual", ["offset", "zero_i8"], ["forward"]),
        helper.make_node("Equal", ["phase", "zero_i8"], ["phase_first"]),
        helper.make_node("Equal", ["phase", "spacing"], ["phase_second"]),
        helper.make_node("And", ["phase_first", "forward"], ["first_lines"]),
        helper.make_node("And", ["phase_second", "forward"], ["second_lines"]),
        helper.make_node("Or", ["first_lines", "second_lines"], ["colored_lines"]),
        helper.make_node("Not", ["colored_lines"], ["not_colored"]),
        helper.make_node("Unsqueeze", ["not_colored", "axis_one"], ["background_component"]),
        helper.make_node("Unsqueeze", ["first_lines", "axis_one"], ["first_component"]),
        helper.make_node("Unsqueeze", ["second_lines", "axis_one"], ["second_component"]),
        helper.make_node(
            "Concat",
            ["background_component", "first_component", "second_component"],
            ["component_bool"],
            axis=1,
        ),
        # The input-occupancy factor in the final contraction suppresses all
        # padded cells, so the short axis can be represented by scalar true.
        helper.make_node(
            "Or", ["horizontal_row", "component_bool"], ["row_factor_bool"]
        ),
        helper.make_node(
            "Or", ["vertical_row", "component_bool"], ["col_factor_bool"]
        ),
        helper.make_node("Cast", ["row_factor_bool"], ["row_factor"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["col_factor_bool"], ["col_factor"], to=TensorProto.FLOAT),
        helper.make_node(
            "Einsum",
            [
                "input", "probe_coordinate", "probe_coordinate",
                "probe_row_route", "probe_col_route",
                "input", "row_factor", "col_factor", "component_sign", "channel_sign",
                "input", "row_factor", "col_factor",
            ],
            ["output"],
            equation=(
                "nqij,ai,bj,oa,ob,ncuv,oku,okv,k,c,"
                "nshw,okh,okw->nchw"
            ),
        ),
    ]

    # Explicit value_info makes scorer-facing static shapes auditable and also
    # catches accidental broadcasting changes during future graph surgery.
    value_info = [
        _value("h_color", TensorProto.FLOAT, [1, 30]),
        _value("v_color", TensorProto.FLOAT, [1, 30]),
    ]

    graph = helper.make_graph(
        nodes,
        "task013_columns",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, GRID_SHAPE)],
        initializers,
        value_info=value_info,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-task013",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 14)],
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


def verify(model: onnx.ModelProto) -> tuple[int, int, int, int, float, float, float]:
    """Verify every packaged example and return official cost statistics."""

    session = _session(model, profiling=True)
    examples_by_subset = json.loads(DATA_PATH.read_text())
    examples = [example for subset in examples_by_subset.values() for example in subset]
    min_true = math.inf
    max_false = -math.inf
    for index, example in enumerate(examples):
        benchmark = neurogolf_utils.convert_to_numpy(example)
        logits = session.run(["output"], {"input": benchmark["input"]})[0]
        expected = benchmark["output"].astype(bool)
        actual = logits > 0.0
        if not np.array_equal(actual, expected):
            mismatch = np.argwhere(actual != expected)[0].tolist()
            raise AssertionError(f"example {index} failed at {mismatch}")
        min_true = min(min_true, float(logits[expected].min()))
        max_false = max(max_false, float(logits[~expected].max()))

    trace_path = session.end_profiling()
    sanitized = neurogolf_utils.sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    memory, parameters = neurogolf_utils.score_network(sanitized, trace_path)
    Path(trace_path).unlink(missing_ok=True)
    assert memory is not None and parameters is not None
    score = max(1.0, 25.0 - math.log(max(1.0, memory + parameters)))
    return len(examples), len(examples), memory, parameters, score, min_true, max_false


def stress_verify(model: onnx.ModelProto, trials: int = 2_000) -> int:
    """Check fresh samples spanning the default generator distribution."""

    sys.path.insert(0, str(ROOT / "ARC-GEN"))
    spec = importlib.util.spec_from_file_location("task013_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    session = _session(model)
    random.seed(13_026)
    coverage: set[tuple[int, int, int, tuple[int, int]]] = set()
    for index in range(trials):
        example = generator.generate()
        benchmark = neurogolf_utils.convert_to_numpy(example)
        logits = session.run(["output"], {"input": benchmark["input"]})[0]
        if not np.array_equal(logits > 0.0, benchmark["output"].astype(bool)):
            raise AssertionError(f"randomized generator example {index} failed")
        grid = np.asarray(example["input"])
        positions = np.argwhere(grid != 0)
        long_positions = positions[:, 0] if grid.shape[0] > grid.shape[1] else positions[:, 1]
        spacing = int(abs(long_positions[1] - long_positions[0]))
        coverage.add((grid.shape[0] > grid.shape[1], spacing, len(set(grid.flat)) - 1,
                      tuple(sorted((int(positions[0, 0] in (0, grid.shape[0] - 1)),
                                    int(positions[1, 0] in (0, grid.shape[0] - 1)))))))
    assert {item[0] for item in coverage} == {False, True}
    assert {item[1] for item in coverage} == {2, 3, 4, 5, 6}
    return trials


def main() -> None:
    model = build_model()
    passed, total, memory, parameters, score, min_true, max_false = verify(model)
    stress_passed = stress_verify(model)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, MODEL_PATH)
    shutil.copyfile(MODEL_PATH, SUBMISSION_PATH)
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    print(f"verified: {passed}/{total}")
    print(f"randomized generator verification: {stress_passed}/{stress_passed}")
    print(f"official cost: {memory} bytes + {parameters} params = {memory + parameters}")
    print(f"official score: {score:.12f}")
    print(f"logit margins: true >= {min_true:.8g}, false <= {max_false:.8g}")
    print(f"model size: {MODEL_PATH.stat().st_size} bytes")
    print(f"sha256: {digest}")
    print(f"wrote {MODEL_PATH.relative_to(ROOT)}")
    print(f"copied {SUBMISSION_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
