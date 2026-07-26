#!/usr/bin/env python3
"""Build and verify NeuroGolf task011 / ARC-GEN 09629e4f."""

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
DATA_PATH = ROOT / "kaggle_tasks_data" / "task011.json"
GENERATOR_PATH = ROOT / "ARC-GEN" / "tasks" / "task_09629e4f.py"
MODEL_PATH = TASK_DIR / "task011.onnx"
SUBMISSION_PATH = ROOT / "submission" / "task011.onnx"
GRID_SHAPE = (1, 10, 30, 30)

sys.path.insert(0, str(ROOT / "utils"))
import neurogolf_utils  # noqa: E402


def _initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name)


def build_model() -> onnx.ModelProto:
    """Return the compact two-Einsum block selector and router."""

    # For a content coordinate 4*a+p, ``outer`` selects block a and ``inner``
    # selects local coordinate p.  Their fourth columns select the two gray
    # separator coordinates.  Reusing these two tables on both spatial axes
    # and at every occurrence is the main parameter saving.
    outer_map = np.zeros((30, 4), dtype=np.float32)
    inner_map = np.zeros((30, 4), dtype=np.float32)
    for block in range(3):
        for local in range(3):
            coordinate = 4 * block + local
            outer_map[coordinate, block] = 1.0
            inner_map[coordinate, local] = 1.0
    outer_map[(3, 7), 3] = 1.0
    inner_map[(3, 7), 3] = 1.0

    # On content cells, a four-color block scores 5*5 - 4*4 = 9 and every
    # five-color block scores 4*5 - 5*4 = 0.  Gray gets a positive weight so
    # the fourth selector row/column supplies the output separator lines.
    count_color = np.asarray(
        [5, -4, -4, -4, -4, 1, -4, -4, -4, -4], dtype=np.float32
    )

    initializers = [
        _initializer("safe_name_0", outer_map),
        _initializer("safe_name_1", inner_map),
        _initializer("safe_name_2", count_color),
    ]

    nodes = [
        # Shape [1,4,4].  The ordinary 3x3 corner is the exact selected-block
        # indicator.  Its last row/column are positive gray-line selectors.
        helper.make_node(
            "Einsum",
            ["input", "safe_name_2", "safe_name_0", "safe_name_0"],
            ["safe_name_3"],
            equation="njuv,j,ua,vb->nab",
        ),
        # Swap the selected block's outer and inner coordinates.  The special
        # fourth coordinate naturally propagates horizontal/vertical gray
        # separators, while zeros after coordinate ten suppress all padding.
        helper.make_node(
            "Einsum",
            [
                "input", "safe_name_3",
                "safe_name_0", "safe_name_1", "safe_name_0",
                "safe_name_0", "safe_name_1", "safe_name_0",
            ],
            ["output"],
            equation="noxy,nab,xa,xp,rp,yb,yq,sq->nors",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "task011_four",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, GRID_SHAPE)],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-task011",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def verify(model: onnx.ModelProto) -> tuple[int, int, int, int, float, float, float]:
    """Verify packaged examples and return counts, cost, and logit margins."""

    sanitized = neurogolf_utils.sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(TASK_DIR / "profile")
    session = ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )

    examples_by_subset = json.loads(DATA_PATH.read_text())
    examples = [example for subset in examples_by_subset.values() for example in subset]
    passed = 0
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
        passed += 1
        min_true = min(min_true, float(logits[expected].min()))
        max_false = max(max_false, float(logits[~expected].max()))

    trace_path = session.end_profiling()
    memory, parameters = neurogolf_utils.score_network(sanitized, trace_path)
    Path(trace_path).unlink(missing_ok=True)
    assert memory is not None and parameters is not None
    score = max(1.0, 25.0 - math.log(max(1.0, memory + parameters)))
    return passed, len(examples), memory, parameters, score, min_true, max_false


def stress_verify(model: onnx.ModelProto, trials: int = 2_000) -> int:
    """Check deterministic fresh samples from the default ARC-GEN generator."""

    sys.path.insert(0, str(ROOT / "ARC-GEN"))
    spec = importlib.util.spec_from_file_location("task011_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    sanitized = neurogolf_utils.sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    session = ort.InferenceSession(
        sanitized.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    random.seed(11_026)
    chosen_blocks: set[tuple[int, int]] = set()
    for index in range(trials):
        example = generator.generate()
        benchmark = neurogolf_utils.convert_to_numpy(example)
        logits = session.run(["output"], {"input": benchmark["input"]})[0]
        if not np.array_equal(logits > 0.0, benchmark["output"].astype(bool)):
            raise AssertionError(f"randomized generator example {index} failed")

        grid = np.asarray(example["input"])
        counts = np.asarray(
            [
                [np.count_nonzero(grid[4 * row : 4 * row + 3, 4 * col : 4 * col + 3])
                 for col in range(3)]
                for row in range(3)
            ]
        )
        chosen_blocks.add(tuple(np.argwhere(counts == 4)[0]))

    assert len(chosen_blocks) == 9
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
