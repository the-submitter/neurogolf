"""Build and verify the compact solver for NeuroGolf task 008 (05f2a901)."""

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(__file__).with_name("task008.onnx")
SUBMISSION_PATH = ROOT / "submission" / "task008.onnx"
TASK_DATA = ROOT / "kaggle_tasks_data" / "task008.json"
SHAPE = [1, 10, 30, 30]


def initializer(name, values, dtype=None):
    array = np.asarray(values, dtype=dtype)
    return numpy_helper.from_array(array, name=name)


def build_model():
    nodes = []

    def node(op_type, inputs, output, **attrs):
        nodes.append(helper.make_node(op_type, inputs, [output], name=output, **attrs))
        return output

    # Channel selectors let Einsum project only red (2) or cyan (8), without
    # creating a full-size channel slice.  The cyan coordinate weights turn a
    # 2x2 block at positions p,p+1 into its top/left coordinate exactly.
    red_selector = np.zeros(10, np.float32)
    red_selector[2] = 1
    cyan_selector = np.zeros(10, np.float32)
    cyan_selector[8] = 1
    coordinate_weights = np.arange(30, dtype=np.float32) / 4 - np.float32(0.125)

    inits = [
        initializer("red_selector", red_selector),
        initializer("cyan_selector", cyan_selector),
        initializer("coordinate_weights", coordinate_weights),
        initializer("zero_i", 0, np.int64),
        initializer("two_i", 2, np.int64),
        initializer("group_one", 1, np.int64),
        initializer("scalar_axes", [0, 1, 2], np.int64),
    ]

    # Red occupancy projections.  TopK(k=5) covers both possible orientations:
    # the generated rectangle is 2/3 x 4/5 before an optional transpose.
    red_rows = node("Einsum", ["input", "red_selector"], "red_rows", equation="bchw,c->bh")
    red_cols = node("Einsum", ["input", "red_selector"], "red_cols", equation="bchw,c->bw")
    # Generator validation guarantees that no complete rectangle row/column is
    # removed.  Thus Sign yields a contiguous occupancy interval, and the two
    # ArgMax tie modes return its first and last coordinates more cheaply than
    # sorting five candidates with TopK.
    red_row_mask = node("Sign", [red_rows], "red_row_mask")
    red_col_mask = node("Sign", [red_cols], "red_col_mask")
    rmin = node("ArgMax", [red_row_mask], "rmin", axis=1, keepdims=0,
                select_last_index=0)
    rmax = node("ArgMax", [red_row_mask], "rmax", axis=1, keepdims=0,
                select_last_index=1)
    cmin = node("ArgMax", [red_col_mask], "cmin", axis=1, keepdims=0,
                select_last_index=0)
    cmax = node("ArgMax", [red_col_mask], "cmax", axis=1, keepdims=0,
                select_last_index=1)

    # For a 2x2 cyan block, these contractions directly return its top-left
    # row and column.  Only scalar tensors are materialized.
    cyan_r_f = node(
        "Einsum", ["input", "cyan_selector", "coordinate_weights"],
        "cyan_r_f", equation="bchw,c,h->b",
    )
    cyan_c_f = node(
        "Einsum", ["input", "cyan_selector", "coordinate_weights"],
        "cyan_c_f", equation="bchw,c,w->b",
    )
    cyan_r = node("Cast", [cyan_r_f], "cyan_r", to=TensorProto.INT64)
    cyan_c = node("Cast", [cyan_c_f], "cyan_c", to=TensorProto.INT64)

    cyan_r2 = node("Add", [cyan_r, "two_i"], "cyan_r2")
    cyan_c2 = node("Add", [cyan_c, "two_i"], "cyan_c2")
    above = node("Less", [rmax, cyan_r], "above")
    below = node("Greater", [rmin, cyan_r2], "below")
    left = node("Less", [cmax, cyan_c], "left")
    vertical = node("Or", [above, below], "vertical")

    down = node("Sub", [cyan_r, rmax], "down_plus_one")
    down = node("Sub", [down, "group_one"], "down")
    up = node("Sub", [cyan_r2, rmin], "up")
    dr_below = node("Where", [below, up, "zero_i"], "dr_below")
    dr = node("Where", [above, down, dr_below], "dr")

    move_right = node("Sub", [cyan_c, cmax], "right_plus_one")
    move_right = node("Sub", [move_right, "group_one"], "move_right")
    move_left = node("Sub", [cyan_c2, cmin], "move_left")
    horizontal_move = node("Where", [left, move_right, move_left], "horizontal_move")
    dc = node("Where", [vertical, "zero_i", horizontal_move], "dc")

    # Always enumerate exactly 3x5 candidate cells.  For a horizontal move the
    # two offset matrices swap, representing the transposed 5x3 rectangle.
    short_offsets = np.broadcast_to(
        np.arange(3, dtype=np.int64)[None, :, None, None], (1, 3, 5, 1)
    ).copy()
    long_offsets = np.broadcast_to(
        np.arange(5, dtype=np.int64)[None, None, :, None], (1, 3, 5, 1)
    ).copy()
    inits.extend([
        initializer("short_offsets", short_offsets),
        initializer("long_offsets", long_offsets),
    ])
    roff = node("Where", [vertical, "short_offsets", "long_offsets"], "roff")
    coff = node("Where", [vertical, "long_offsets", "short_offsets"], "coff")
    # Broadcast four scalar origins over the candidate offsets.  This skips
    # four 3x5 coordinate tensors and their four equally-sized Unsqueeze
    # outputs: only the final grouped row/column tensors are materialized.
    dst_r0 = node("Add", [rmin, dr], "dst_r0")
    dst_c0 = node("Add", [cmin, dc], "dst_c0")
    src_r0_u = node("Unsqueeze", [rmin, "scalar_axes"], "src_r0_u")
    src_c0_u = node("Unsqueeze", [cmin, "scalar_axes"], "src_c0_u")
    dst_r0_u = node("Unsqueeze", [dst_r0, "scalar_axes"], "dst_r0_u")
    dst_c0_u = node("Unsqueeze", [dst_c0, "scalar_axes"], "dst_c0_u")
    row_bases = node(
        "Concat", [src_r0_u, src_r0_u, dst_r0_u, dst_r0_u], "row_bases", axis=0
    )
    col_bases = node(
        "Concat", [src_c0_u, src_c0_u, dst_c0_u, dst_c0_u], "col_bases", axis=0
    )
    group_rows = node("Add", [row_bases, roff], "group_rows")
    group_cols = node("Add", [col_bases, coff], "group_cols")

    prefix = np.zeros((4, 3, 5, 2), np.int64)
    prefix[0, :, :, 1] = 0  # source: add black
    prefix[1, :, :, 1] = 2  # source: remove red
    prefix[2, :, :, 1] = 0  # destination: remove black
    prefix[3, :, :, 1] = 2  # destination: add red
    inits.extend([
        initializer("index_prefix", prefix),
        initializer("delta_coefficients", np.asarray([1, -1, -1, 1], np.float32)[:, None, None]),
    ])
    indices = node("Concat", ["index_prefix", group_rows, group_cols], "indices", axis=3)

    # Read the source-red mask using the already-built scatter indices.  Holes
    # and unused candidates produce a zero delta, so no filtering tensor is needed.
    indexed_values = node("GatherND", ["input", indices], "indexed_values")
    # Contracting the indexed values directly avoids materializing a separate
    # 3x5 source mask.  Only group 1 (source red) contributes to each update.
    source_group = np.zeros(4, np.float32)
    source_group[1] = 1
    inits.append(initializer("source_group", source_group))
    updates = node(
        "Einsum", [indexed_values, "source_group", "delta_coefficients"],
        "updates", equation="gij,g,hij->hij",
    )
    node("ScatterND", ["input", indices, updates], "output", reduction="add")

    graph = helper.make_graph(
        nodes,
        "task008_magnets",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        inits,
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 16)],
        producer_name="neurogolf-task008",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def one_hot(grid):
    value = np.zeros(SHAPE, dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            value[0, color, row, col] = 1.0
    return value


def verify_model(path):
    data = json.loads(TASK_DATA.read_text())
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(path.read_bytes(), options,
                                   providers=["CPUExecutionProvider"])
    passed = total = 0
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data[split]):
            total += 1
            actual = session.run(["output"], {"input": one_hot(example["input"])})[0] > 0
            expected = one_hot(example["output"]).astype(bool)
            if not np.array_equal(actual, expected):
                raise AssertionError(f"failed {split}[{index}]")
            passed += 1
    return passed, total


def official_cost(path):
    sys.path.insert(0, str(ROOT))
    from utils.neurogolf_utils import sanitize_model, score_network

    model = sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("official sanitizer rejected model")
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(path.parent / "task008-profile")
    session = ort.InferenceSession(model.SerializeToString(), options,
                                   providers=["CPUExecutionProvider"])
    data = json.loads(TASK_DATA.read_text())
    for split in ("train", "test", "arc-gen"):
        for example in data[split]:
            session.run(["output"], {"input": one_hot(example["input"])})
    trace_path = Path(session.end_profiling())
    memory, params = score_network(model, trace_path)
    trace_path.unlink(missing_ok=True)
    if memory is None or params is None:
        raise RuntimeError("official scorer rejected model")
    points = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
    return memory, params, points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), MODEL_PATH)
    passed, total = verify_model(MODEL_PATH)
    memory, params, points = official_cost(MODEL_PATH)
    if not args.no_copy:
        shutil.copy2(MODEL_PATH, SUBMISSION_PATH)
    print(f"verified: {passed}/{total}")
    print(f"official cost: {memory} bytes + {params} params = {memory + params}")
    print(f"estimated points: {points:.6f}")
    print(f"model: {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")
    if not args.no_copy:
        print(f"submission: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
