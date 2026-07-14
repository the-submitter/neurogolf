from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from hypothesis import given
from hypothesis import strategies as st

from neurogolf.judge.official_adapter import OfficialAdapter
from neurogolf.onnx_lab.builder import GraphBuilder
from neurogolf.onnx_lab.constants import constant_node, initializer, scalar
from neurogolf.onnx_lab.conv import (
    color_map_1x1,
    conv2d,
    depthwise_conv2d,
    grouped_conv2d,
)
from neurogolf.onnx_lab.graph import NameScope, static_shape, tensor_info_map
from neurogolf.onnx_lab.inspect import fusion_opportunities, inspect_model
from neurogolf.onnx_lab.masks import (
    cast_mask,
    color_channel,
    color_mask,
    equal,
    greater,
    logical_and,
    logical_not,
    logical_or,
)
from neurogolf.onnx_lab.optimize import (
    eliminate_common_subexpressions,
    fold_constant_nodes,
    optimize_model,
    remove_dead_nodes,
    remove_identity_nodes,
    remove_redundant_casts,
    remove_redundant_transposes,
)
from neurogolf.onnx_lab.reductions import (
    argmax,
    argmin,
    color_counts,
    column_sums,
    einsum,
    expand,
    reduce_sum,
    row_sums,
)
from neurogolf.onnx_lab.rendering import render_color_mask, signed_logits
from neurogolf.onnx_lab.score_estimator import (
    estimate_score,
    official_score,
    parameter_elements,
    static_tensor_costs,
)
from neurogolf.onnx_lab.spatial import (
    flip_horizontal,
    flip_vertical,
    gather_axis,
    pad,
    shift,
    slice_tensor,
    transpose_hw,
)
from neurogolf.onnx_lab.templates.basic import identity_model, single_conv_model

GRID_SHAPE = (1, 10, 30, 30)


def _run(model: onnx.ModelProto, value: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return cast(np.ndarray, session.run(None, {"input": value})[0])


def _finish(
    builder: GraphBuilder,
    value: str,
    *,
    dtype: int = onnx.TensorProto.FLOAT,
    shape: tuple[int, ...] = GRID_SHAPE,
) -> onnx.ModelProto:
    builder.direct_output(value, dtype=dtype, shape=shape)
    return builder.build()


def test_name_scope_and_static_metadata() -> None:
    names = NameScope()
    assert names.unique("odd name/kernel_time") == "odd_name_kernel"
    assert names.unique("odd name/kernel_time") == "odd_name_kernel_1"
    with pytest.raises(ValueError, match="already reserved"):
        names.reserve(names.reserve("taken"))
    model = identity_model()
    metadata = tensor_info_map(model.graph)
    assert static_shape(metadata["input"]) == GRID_SHAPE


def test_builder_constants_and_direct_output(tmp_path: Path) -> None:
    builder = GraphBuilder.standard("constants")
    assert builder.opset == 12
    one = initializer(builder, np.ones(GRID_SHAPE, dtype=np.float32), name="one")
    two = constant_node(builder, np.asarray(2.0, dtype=np.float32), name="two")
    assert scalar(builder, 3.0, name="three")
    summed = str(
        builder.node("Add", ["input", one], dtype=onnx.TensorProto.FLOAT, shape=GRID_SHAPE)
    )
    multiplied = str(
        builder.node("Mul", [summed, two], dtype=onnx.TensorProto.FLOAT, shape=GRID_SHAPE)
    )
    model = _finish(builder, multiplied)
    actual = _run(model, np.zeros(GRID_SHAPE, dtype=np.float32))
    assert np.all(actual == 2.0)
    assert parameter_elements(model) == math.prod(GRID_SHAPE) + 2
    destination = tmp_path / "builder.onnx"
    assert builder.save(destination).SerializeToString() == model.SerializeToString()
    assert destination.is_file()
    with pytest.raises(ValueError, match="already declared"):
        builder.add_output("output", onnx.TensorProto.FLOAT, GRID_SHAPE)


def test_opset_12_einsum_is_available_and_opset_remains_configurable() -> None:
    value = np.arange(math.prod(GRID_SHAPE), dtype=np.float32).reshape(GRID_SHAPE)
    builder = GraphBuilder.standard("einsum")
    copied = einsum(builder, ["input"], "nchw->nchw", shape=GRID_SHAPE)
    assert np.array_equal(_run(_finish(builder, copied), value), value)
    with pytest.raises(ValueError, match="opset 12"):
        einsum(
            GraphBuilder.standard("legacy", opset=10),
            ["input"],
            "nchw->nchw",
            shape=GRID_SHAPE,
        )
    assert GraphBuilder.standard(opset=18).opset == 18


@given(row=st.integers(-2, 2), col=st.integers(-2, 2))
def test_shift_helper_moves_one_hot_values(row: int, col: int) -> None:
    builder = GraphBuilder.standard("shift")
    model = _finish(builder, shift(builder, "input", row, col))
    value = np.zeros(GRID_SHAPE, dtype=np.float32)
    value[0, 4, 10, 10] = 1
    actual = _run(model, value)
    assert actual[0, 4, 10 + row, 10 + col] == 1
    assert actual.sum() == 1


def test_conv_grouped_depthwise_and_color_mapping(official_adapter: OfficialAdapter) -> None:
    identity_weights = np.eye(10, dtype=np.float32).reshape(10, 10, 1, 1)
    builder = GraphBuilder.standard("conv")
    conv = conv2d(builder, "input", identity_weights)
    model = _finish(builder, conv)
    tensor = official_adapter.grid_to_tensor([[1, 2], [3, 4]])
    assert np.array_equal(_run(model, tensor), tensor)

    grouped_builder = GraphBuilder.standard("grouped")
    grouped = grouped_conv2d(
        grouped_builder,
        "input",
        np.ones((10, 1, 1, 1), dtype=np.float32),
        groups=10,
    )
    assert np.array_equal(_run(_finish(grouped_builder, grouped), tensor), tensor)

    depthwise_builder = GraphBuilder.standard("depthwise")
    depthwise = depthwise_conv2d(depthwise_builder, "input", np.ones((10, 1, 1), dtype=np.float32))
    assert np.array_equal(_run(_finish(depthwise_builder, depthwise), tensor), tensor)

    map_builder = GraphBuilder.standard("color_map")
    mapped = color_map_1x1(map_builder, "input", [0, 2, 1, 3, 4, 5, 6, 7, 8, 9])
    thresholded = (_run(_finish(map_builder, mapped), tensor) > 0).astype(float)
    assert official_adapter.tensor_to_grid(thresholded) == [[2, 1], [3, 4]]
    with pytest.raises(ValueError, match="ten target colors"):
        color_map_1x1(GraphBuilder.standard(), "input", [0, 1])


def test_masks_boolean_logic_and_rendering(official_adapter: OfficialAdapter) -> None:
    builder = GraphBuilder.standard("mask_render")
    channel = color_channel(builder, "input", 2)
    threshold = builder.initializer(np.asarray(0.5, dtype=np.float32))
    selected = greater(builder, channel, threshold, shape=(1, 1, 30, 30))
    same = equal(builder, selected, color_mask(builder, "input", 2), shape=(1, 1, 30, 30))
    conjunction = logical_and(builder, selected, same, shape=(1, 1, 30, 30))
    disjunction = logical_or(builder, conjunction, selected, shape=(1, 1, 30, 30))
    inverted = logical_not(builder, disjunction, shape=(1, 1, 30, 30))
    restored = logical_not(builder, inverted, shape=(1, 1, 30, 30))
    assert cast_mask(builder, restored, shape=(1, 1, 30, 30))
    logits = render_color_mask(builder, restored, foreground=7, background=0)
    signed_logits(builder, logits)
    model = builder.build()
    tensor = official_adapter.grid_to_tensor([[2, 0], [0, 2]])
    output = (_run(model, tensor) > 0).astype(float)
    rendered = official_adapter.tensor_to_grid(output)
    assert rendered[0][:2] == [7, 0]
    assert rendered[1][:2] == [0, 7]
    assert len(rendered) == len(rendered[0]) == 30
    with pytest.raises(ValueError, match="range"):
        color_channel(GraphBuilder.standard(), "input", 12)


def test_transpose_flip_gather_slice_and_pad() -> None:
    value = np.arange(math.prod(GRID_SHAPE), dtype=np.float32).reshape(GRID_SHAPE)

    builder = GraphBuilder.standard("transpose")
    assert np.array_equal(
        _run(_finish(builder, transpose_hw(builder, "input")), value), value.transpose(0, 1, 3, 2)
    )

    builder = GraphBuilder.standard("horizontal")
    assert np.array_equal(
        _run(_finish(builder, flip_horizontal(builder, "input")), value), value[:, :, :, ::-1]
    )

    builder = GraphBuilder.standard("vertical")
    assert np.array_equal(
        _run(_finish(builder, flip_vertical(builder, "input")), value), value[:, :, ::-1, :]
    )

    builder = GraphBuilder.standard("gather")
    gathered = gather_axis(builder, "input", [2, 0], axis=1, input_shape=GRID_SHAPE)
    gathered_model = _finish(builder, gathered, shape=(1, 2, 30, 30))
    assert np.array_equal(_run(gathered_model, value), value[:, [2, 0]])

    builder = GraphBuilder.standard("slice")
    sliced = slice_tensor(
        builder,
        "input",
        starts=[1, 2],
        ends=[5, 8],
        axes=[2, 3],
        steps=[2, 3],
        input_shape=GRID_SHAPE,
    )
    sliced_model = _finish(builder, sliced, shape=(1, 10, 2, 2))
    assert np.array_equal(_run(sliced_model, value), value[:, :, 1:5:2, 2:8:3])
    with pytest.raises(ValueError, match="positive steps"):
        slice_tensor(
            GraphBuilder.standard(),
            "input",
            starts=[2],
            ends=[0],
            axes=[2],
            steps=[-1],
            input_shape=GRID_SHAPE,
        )

    builder = GraphBuilder.standard("pad")
    padded = pad(builder, "input", pads=[0, 0, 1, 2, 0, 0, 3, 4], input_shape=GRID_SHAPE)
    padded_model = _finish(builder, padded, shape=(1, 10, 34, 36))
    assert _run(padded_model, value).shape == (1, 10, 34, 36)
    with pytest.raises(ValueError, match="begin/end"):
        pad(GraphBuilder.standard(), "input", pads=[1], input_shape=GRID_SHAPE)


@pytest.mark.parametrize(
    ("helper", "expected_shape"),
    [(row_sums, (1, 10, 30, 1)), (column_sums, (1, 10, 1, 30)), (color_counts, (1, 10, 1, 1))],
)
def test_reduction_shortcuts(helper, expected_shape: tuple[int, ...]) -> None:
    builder = GraphBuilder.standard("reduction")
    reduced = helper(builder, "input")
    model = _finish(builder, reduced, shape=expected_shape)
    actual = _run(model, np.ones(GRID_SHAPE, dtype=np.float32))
    assert actual.shape == expected_shape


def test_reduce_extrema_and_expand() -> None:
    value = np.zeros(GRID_SHAPE, dtype=np.float32)
    value[:, 3] = 4
    builder = GraphBuilder.standard("reduce_sum")
    reduced = reduce_sum(builder, "input", axes=[1], input_shape=GRID_SHAPE, keepdims=False)
    assert _run(_finish(builder, reduced, shape=(1, 30, 30)), value).shape == (1, 30, 30)

    builder = GraphBuilder.standard("argmax")
    maximum = argmax(builder, "input", axis=1, input_shape=GRID_SHAPE)
    max_result = _run(
        _finish(builder, maximum, dtype=onnx.TensorProto.INT64, shape=(1, 1, 30, 30)), value
    )
    assert np.all(max_result == 3)

    builder = GraphBuilder.standard("argmin")
    minimum = argmin(builder, "input", axis=1, input_shape=GRID_SHAPE)
    min_result = _run(
        _finish(builder, minimum, dtype=onnx.TensorProto.INT64, shape=(1, 1, 30, 30)), value
    )
    assert np.all(min_result == 0)

    builder = GraphBuilder.standard("expand")
    counts = color_counts(builder, "input")
    expanded = expand(builder, counts, GRID_SHAPE)
    assert _run(_finish(builder, expanded), value).shape == GRID_SHAPE


def test_score_estimation_inspection_and_official_call(
    tmp_path: Path, official_adapter: OfficialAdapter
) -> None:
    identity = identity_model()
    estimate = estimate_score(identity)
    assert estimate["objective"] == 0
    assert estimate["points"] == 25
    assert static_tensor_costs(identity) == []
    assert parameter_elements(identity) == 0
    inspected = inspect_model(identity)
    assert inspected["operators"] == {"Identity": 1}
    path = tmp_path / "identity.onnx"
    onnx.save(identity, path)
    assert inspect_model(path)["file_size_bytes"] == path.stat().st_size
    score = official_score(official_adapter, path, [[[0]]])
    assert score.objective == 0
    assert score.points == 25

    conv = single_conv_model()
    conv_estimate = estimate_score(conv)
    assert conv_estimate["parameter_elements"] == 100
    assert conv_estimate["objective"] == 100


def test_optimizer_rewrites_preserve_valid_graphs() -> None:
    builder = GraphBuilder.standard("rewrites")
    cast = str(
        builder.node(
            "Cast",
            ["input"],
            to=onnx.TensorProto.FLOAT,
            dtype=onnx.TensorProto.FLOAT,
            shape=GRID_SHAPE,
        )
    )
    first = transpose_hw(builder, cast)
    second = transpose_hw(builder, first)
    identity = str(
        builder.node("Identity", [second], dtype=onnx.TensorProto.FLOAT, shape=GRID_SHAPE)
    )
    builder.node("Relu", ["input"], dtype=onnx.TensorProto.FLOAT, shape=GRID_SHAPE, name="dead")
    model = _finish(builder, identity)
    optimized = optimize_model(model)
    assert [node.op_type for node in optimized.graph.node] == ["Identity"]
    onnx.checker.check_model(optimized, full_check=True)

    assert len(remove_redundant_casts(model).graph.node) < len(model.graph.node)
    assert len(remove_redundant_transposes(model).graph.node) < len(model.graph.node)
    assert len(remove_identity_nodes(model).graph.node) < len(model.graph.node)
    assert len(remove_dead_nodes(model).graph.node) < len(model.graph.node)


def test_constant_folding_cse_and_fusion_detection() -> None:
    builder = GraphBuilder.standard("fold_cse")
    constant = constant_node(builder, np.asarray(1.0, dtype=np.float32))
    added = str(
        builder.node("Add", ["input", constant], dtype=onnx.TensorProto.FLOAT, shape=GRID_SHAPE)
    )
    duplicate = str(
        builder.node("Add", ["input", constant], dtype=onnx.TensorProto.FLOAT, shape=GRID_SHAPE)
    )
    combined = str(
        builder.node("Mul", [added, duplicate], dtype=onnx.TensorProto.FLOAT, shape=GRID_SHAPE)
    )
    model = _finish(builder, combined)
    folded = fold_constant_nodes(model)
    assert all(node.op_type != "Constant" for node in folded.graph.node)
    assert parameter_elements(folded) == 1
    common = eliminate_common_subexpressions(folded)
    assert sum(node.op_type == "Add" for node in common.graph.node) == 1
    onnx.checker.check_model(common, full_check=True)
    fusion_builder = GraphBuilder.standard("fusion")
    one = fusion_builder.initializer(np.asarray(1.0, dtype=np.float32))
    added = str(
        fusion_builder.node("Add", ["input", one], dtype=onnx.TensorProto.FLOAT, shape=GRID_SHAPE)
    )
    compared = str(
        fusion_builder.node("Greater", [added, one], dtype=onnx.TensorProto.BOOL, shape=GRID_SHAPE)
    )
    fusion_model = _finish(fusion_builder, compared, dtype=onnx.TensorProto.BOOL, shape=GRID_SHAPE)
    assert fusion_opportunities(fusion_model)[0]["pattern"] == "Add->Greater"
