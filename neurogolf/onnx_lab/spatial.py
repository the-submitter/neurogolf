"""Static spatial shifts, slices, gathers, transposes, and flips."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import onnx

from neurogolf.onnx_lab.builder import GraphBuilder
from neurogolf.onnx_lab.conv import depthwise_conv2d


def shift(
    builder: GraphBuilder,
    value: str,
    row_delta: int,
    col_delta: int,
    *,
    channels: int = 10,
    output: str | None = None,
) -> str:
    """Move source pixels by `(row_delta, col_delta)`, zero-filling boundaries."""

    radius = max(abs(row_delta), abs(col_delta))
    size = 2 * radius + 1
    kernels = np.zeros((channels, 1, size, size), dtype=np.float32)
    kernels[:, 0, radius - row_delta, radius - col_delta] = 1.0
    return depthwise_conv2d(
        builder,
        value,
        kernels,
        pads=(radius, radius, radius, radius),
        input_shape=(1, channels, 30, 30),
        output=output,
    )


def transpose_hw(
    builder: GraphBuilder, value: str, *, shape: Sequence[int] = (1, 10, 30, 30)
) -> str:
    out_shape = (int(shape[0]), int(shape[1]), int(shape[3]), int(shape[2]))
    return str(
        builder.node(
            "Transpose",
            [value],
            perm=[0, 1, 3, 2],
            dtype=onnx.TensorProto.FLOAT,
            shape=out_shape,
        )
    )


def gather_axis(
    builder: GraphBuilder,
    value: str,
    indices: Sequence[int],
    *,
    axis: int,
    input_shape: Sequence[int],
    dtype: int = onnx.TensorProto.FLOAT,
) -> str:
    index = builder.initializer(np.asarray(indices, dtype=np.int64), name="gather_indices")
    out_shape = list(int(item) for item in input_shape)
    out_shape[axis] = len(indices)
    return str(builder.node("Gather", [value, index], axis=axis, dtype=dtype, shape=out_shape))


def flip_horizontal(builder: GraphBuilder, value: str, *, width: int = 30) -> str:
    return gather_axis(
        builder,
        value,
        list(range(width - 1, -1, -1)),
        axis=3,
        input_shape=(1, 10, 30, width),
    )


def flip_vertical(builder: GraphBuilder, value: str, *, height: int = 30) -> str:
    return gather_axis(
        builder,
        value,
        list(range(height - 1, -1, -1)),
        axis=2,
        input_shape=(1, 10, height, 30),
    )


def slice_tensor(
    builder: GraphBuilder,
    value: str,
    *,
    starts: Sequence[int],
    ends: Sequence[int],
    axes: Sequence[int],
    input_shape: Sequence[int],
    steps: Sequence[int] | None = None,
    dtype: int = onnx.TensorProto.FLOAT,
) -> str:
    step_values = list(steps or [1] * len(starts))
    if any(step <= 0 for step in step_values):
        raise ValueError("Slice helper currently requires positive steps")
    inputs = [
        value,
        builder.initializer(np.asarray(starts, dtype=np.int64), name="slice_starts"),
        builder.initializer(np.asarray(ends, dtype=np.int64), name="slice_ends"),
        builder.initializer(np.asarray(axes, dtype=np.int64), name="slice_axes"),
        builder.initializer(np.asarray(step_values, dtype=np.int64), name="slice_steps"),
    ]
    out_shape = list(int(item) for item in input_shape)
    for start, end, axis, step_value in zip(starts, ends, axes, step_values, strict=True):
        out_shape[axis] = max(0, (end - start + step_value - 1) // step_value)
    return str(builder.node("Slice", inputs, dtype=dtype, shape=out_shape))


def pad(
    builder: GraphBuilder,
    value: str,
    *,
    pads: Sequence[int],
    input_shape: Sequence[int],
    constant_value: float = 0.0,
    dtype: int = onnx.TensorProto.FLOAT,
) -> str:
    rank = len(input_shape)
    if len(pads) != rank * 2:
        raise ValueError("Pad list must contain begin/end values for every axis")
    out_shape = [
        int(input_shape[index]) + int(pads[index]) + int(pads[index + rank])
        for index in range(rank)
    ]
    if builder.opset >= 11:
        inputs = [
            value,
            builder.initializer(np.asarray(pads, dtype=np.int64), name="pad_widths"),
        ]
        if constant_value != 0.0:
            value_dtype = onnx.helper.tensor_dtype_to_np_dtype(dtype)
            inputs.append(
                builder.initializer(
                    np.asarray(constant_value, dtype=value_dtype), name="pad_constant"
                )
            )
        return str(
            builder.node(
                "Pad",
                inputs,
                mode="constant",
                dtype=dtype,
                shape=out_shape,
            )
        )
    return str(
        builder.node(
            "Pad",
            [value],
            mode="constant",
            pads=list(pads),
            value=float(constant_value),
            dtype=dtype,
            shape=out_shape,
        )
    )
