#!/usr/bin/env python3
"""Build and verify NeuroGolf task012 / ARC-GEN 0962bcdd."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from scipy.optimize import linprog


TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parents[1]
DATA_PATH = ROOT / "kaggle_tasks_data" / "task012.json"
GENERATOR_PATH = ROOT / "ARC-GEN" / "tasks" / "task_0962bcdd.py"
MODEL_PATH = TASK_DIR / "task012.onnx"
SUBMISSION_PATH = ROOT / "submission" / "task012.onnx"
GRID_SHAPE = (1, 10, 30, 30)
KERNEL_SHAPE = (8, 9)
PADS = (3, 4, 4, 4)
MARGIN = 0.03

sys.path.insert(0, str(ROOT / "utils"))
import neurogolf_utils  # noqa: E402


def one_hot(grid: list[list[int]]) -> np.ndarray:
    """Encode an ARC grid in the scorer's unbatched 10x30x30 layout."""

    result = np.zeros(GRID_SHAPE[1:], dtype=np.uint8)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            result[color, row, col] = 1
    return result


def load_examples() -> list[dict[str, list[list[int]]]]:
    subsets = json.loads(DATA_PATH.read_text())
    return [example for subset in subsets.values() for example in subset]


def collect_patches(
    examples: list[dict[str, list[list[int]]]], channels: Iterable[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the unique labeled same-channel 8x9 neighborhoods."""

    labels_by_patch: dict[bytes, int] = {}
    for example in examples:
        source = one_hot(example["input"])
        target = one_hot(example["output"])
        padded = np.pad(source, ((0, 0), (3, 4), (4, 4)))
        for channel in channels:
            patches = np.lib.stride_tricks.sliding_window_view(
                padded[channel], KERNEL_SHAPE
            ).reshape(30 * 30, -1)
            labels = target[channel].reshape(-1)
            for patch, label in zip(patches, labels, strict=True):
                key = np.packbits(patch).tobytes()
                old_label = labels_by_patch.get(key)
                if old_label is not None and old_label != int(label):
                    raise AssertionError("the chosen receptive field is ambiguous")
                labels_by_patch[key] = int(label)

    # Every patch has 72 bits, exactly nine packed bytes.  Keeping only unique
    # rows makes the linear programs small and deterministic.
    packed = np.frombuffer(b"".join(labels_by_patch), dtype=np.uint8)
    patches = np.unpackbits(packed).reshape(len(labels_by_patch), 72).astype(float)
    labels = np.fromiter(labels_by_patch.values(), dtype=np.uint8)
    return patches, labels


def fit_kernel(
    examples: list[dict[str, list[list[int]]]], channels: Iterable[int]
) -> np.ndarray:
    """Fit a robust affine separator with a fixed -1 Conv bias."""

    patches, labels = collect_patches(examples, channels)
    positive = patches[labels == 1]
    negative = patches[labels == 0]

    # ONNX Runtime thresholds the terminal logits at zero.  With bias -1,
    # these inequalities put every true cell at >= +MARGIN and every false
    # cell at <= -MARGIN.
    constraints = np.concatenate((-positive, negative))
    bounds = np.concatenate(
        (
            np.full(len(positive), -(1.0 + MARGIN)),
            np.full(len(negative), 1.0 - MARGIN),
        )
    )
    solution = linprog(
        np.zeros(72),
        A_ub=constraints,
        b_ub=bounds,
        bounds=[(None, None)] * 72,
        method="highs",
    )
    if not solution.success:
        raise AssertionError(f"kernel fit failed: {solution.message}")
    return solution.x.astype(np.float32).reshape(KERNEL_SHAPE)


def build_model() -> onnx.ModelProto:
    """Build the single-node, zero-intermediate-memory depthwise Conv."""

    examples = load_examples()
    background_kernel = fit_kernel(examples, [0])
    foreground_kernel = fit_kernel(examples, range(1, 10))
    weights = np.stack(
        [background_kernel] + [foreground_kernel] * 9, axis=0
    )[:, None, :, :]
    bias = np.full(10, -1.0, dtype=np.float32)

    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["input", "W", "B"],
                ["output"],
                group=10,
                kernel_shape=list(KERNEL_SHAPE),
                pads=list(PADS),
            )
        ],
        "task012_supernova",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, GRID_SHAPE)],
        [numpy_helper.from_array(weights, "W"), numpy_helper.from_array(bias, "B")],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-task012",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def make_session(model: onnx.ModelProto, profiling: bool = False) -> ort.InferenceSession:
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


def check_example(
    session: ort.InferenceSession, example: dict[str, list[list[int]]]
) -> tuple[float, float]:
    benchmark = neurogolf_utils.convert_to_numpy(example)
    logits = session.run(["output"], {"input": benchmark["input"]})[0]
    expected = benchmark["output"].astype(bool)
    if not np.array_equal(logits > 0.0, expected):
        mismatch = np.argwhere((logits > 0.0) != expected)[0].tolist()
        raise AssertionError(f"example failed at {mismatch}")
    return float(logits[expected].min()), float(logits[~expected].max())


def verify_packaged(
    model: onnx.ModelProto,
) -> tuple[int, int, int, int, float, float, float]:
    """Verify all competition examples and measure the official cost."""

    session = make_session(model, profiling=True)
    examples = load_examples()
    margins = [check_example(session, example) for example in examples]
    trace_path = session.end_profiling()

    sanitized = neurogolf_utils.sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    memory, parameters = neurogolf_utils.score_network(sanitized, trace_path)
    Path(trace_path).unlink(missing_ok=True)
    assert memory is not None and parameters is not None
    score = max(1.0, 25.0 - math.log(max(1.0, memory + parameters)))
    return (
        len(examples),
        len(examples),
        memory,
        parameters,
        score,
        min(margin[0] for margin in margins),
        max(margin[1] for margin in margins),
    )


def verify_generator_space(model: onnx.ModelProto) -> int:
    """Exhaust the structural generator space (colors are channel-equivariant)."""

    sys.path.insert(0, str(ROOT / "ARC-GEN"))
    spec = importlib.util.spec_from_file_location("task012_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    session = make_session(model)
    checked = 0
    # The foreground kernel is identical for channels 1..9.  Swapping these
    # two representative colors covers both center-color and arm-color roles;
    # columns and gravity are then enumerated completely.
    for colors in ([1, 2], [2, 1]):
        for first_col in range(3, 10):
            for second_col in range(3, 10):
                for gravity in range(4):
                    example = generator.generate(
                        colors=colors,
                        cols=[first_col, second_col],
                        gravity=gravity,
                    )
                    check_example(session, example)
                    checked += 1
    return checked


def main() -> None:
    model = build_model()
    passed, total, memory, parameters, score, min_true, max_false = verify_packaged(
        model
    )
    exhaustive_cases = verify_generator_space(model)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, MODEL_PATH)
    shutil.copyfile(MODEL_PATH, SUBMISSION_PATH)
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    print(f"verified packaged examples: {passed}/{total}")
    print(f"verified exhaustive structural cases: {exhaustive_cases}/{exhaustive_cases}")
    print(f"official cost: {memory} bytes + {parameters} params = {memory + parameters}")
    print(f"official score: {score:.12f}")
    print(f"logit margins: true >= {min_true:.9g}, false <= {max_false:.9g}")
    print(f"model size: {MODEL_PATH.stat().st_size} bytes")
    print(f"sha256: {digest}")
    print(f"wrote {MODEL_PATH.relative_to(ROOT)}")
    print(f"copied {SUBMISSION_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
