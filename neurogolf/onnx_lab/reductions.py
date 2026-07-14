"""Row/column/color reductions, extrema, and broadcasting helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import onnx

from neurogolf.onnx_lab.builder import GraphBuilder


def einsum(
    builder: GraphBuilder,
    values: Sequence[str],
    equation: str,
    *,
    shape: Sequence[int],
    dtype: int = onnx.TensorProto.FLOAT,
    output: str | None = None,
) -> str:
    """Add a statically described standard-domain Einsum node (opset 12+)."""

    if builder.opset < 12:
        raise ValueError("Einsum requires opset 12 or newer")
    if not values:
        raise ValueError("Einsum requires at least one input")
    if not equation.strip():
        raise ValueError("Einsum equation cannot be empty")
    return str(
        builder.node(
            "Einsum",
            values,
            equation=equation,
            output=output,
            dtype=dtype,
            shape=shape,
        )
    )


def reduce_sum(
    builder: GraphBuilder,
    value: str,
    *,
    axes: Sequence[int],
    input_shape: Sequence[int],
    keepdims: bool = True,
    dtype: int = onnx.TensorProto.FLOAT,
) -> str:
    out_shape = [int(item) for item in input_shape]
    for axis in sorted((axis % len(out_shape) for axis in axes), reverse=True):
        if keepdims:
            out_shape[axis] = 1
        else:
            del out_shape[axis]
    return str(
        builder.node(
            "ReduceSum",
            [value],
            axes=list(axes),
            keepdims=int(keepdims),
            dtype=dtype,
            shape=out_shape,
        )
    )


def row_sums(builder: GraphBuilder, value: str, *, channels: int = 10) -> str:
    return reduce_sum(builder, value, axes=[3], input_shape=(1, channels, 30, 30))


def column_sums(builder: GraphBuilder, value: str, *, channels: int = 10) -> str:
    return reduce_sum(builder, value, axes=[2], input_shape=(1, channels, 30, 30))


def color_counts(builder: GraphBuilder, value: str, *, channels: int = 10) -> str:
    return reduce_sum(builder, value, axes=[2, 3], input_shape=(1, channels, 30, 30))


def argmax(
    builder: GraphBuilder,
    value: str,
    *,
    axis: int,
    input_shape: Sequence[int],
    keepdims: bool = True,
) -> str:
    out_shape = [int(item) for item in input_shape]
    if keepdims:
        out_shape[axis] = 1
    else:
        del out_shape[axis]
    return str(
        builder.node(
            "ArgMax",
            [value],
            axis=axis,
            keepdims=int(keepdims),
            dtype=onnx.TensorProto.INT64,
            shape=out_shape,
        )
    )


def argmin(
    builder: GraphBuilder,
    value: str,
    *,
    axis: int,
    input_shape: Sequence[int],
    keepdims: bool = True,
) -> str:
    out_shape = [int(item) for item in input_shape]
    if keepdims:
        out_shape[axis] = 1
    else:
        del out_shape[axis]
    return str(
        builder.node(
            "ArgMin",
            [value],
            axis=axis,
            keepdims=int(keepdims),
            dtype=onnx.TensorProto.INT64,
            shape=out_shape,
        )
    )


def expand(
    builder: GraphBuilder,
    value: str,
    shape: Sequence[int],
    *,
    dtype: int = onnx.TensorProto.FLOAT,
) -> str:
    shape_name = builder.initializer(np.asarray(shape, dtype=np.int64), name="expand_shape")
    return str(builder.node("Expand", [value, shape_name], dtype=dtype, shape=shape))
