"""Bool-mask and channel selection helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import onnx

from neurogolf.onnx_lab.builder import GraphBuilder


def color_channel(
    builder: GraphBuilder, value: str, color: int, *, shape: Sequence[int] = (1, 10, 30, 30)
) -> str:
    if not 0 <= color <= 9:
        raise ValueError("Color must be in range 0..9")
    index = builder.initializer(np.asarray([color], dtype=np.int64), name="color_index")
    output_shape = (int(shape[0]), 1, int(shape[2]), int(shape[3]))
    return str(
        builder.node(
            "Gather",
            [value, index],
            axis=1,
            dtype=onnx.TensorProto.FLOAT,
            shape=output_shape,
        )
    )


def greater(builder: GraphBuilder, left: str, right: str, *, shape: Sequence[int]) -> str:
    return str(builder.node("Greater", [left, right], dtype=onnx.TensorProto.BOOL, shape=shape))


def equal(builder: GraphBuilder, left: str, right: str, *, shape: Sequence[int]) -> str:
    return str(builder.node("Equal", [left, right], dtype=onnx.TensorProto.BOOL, shape=shape))


def color_mask(builder: GraphBuilder, value: str, color: int) -> str:
    channel = color_channel(builder, value, color)
    threshold = builder.initializer(np.asarray(0.5, dtype=np.float32), name="threshold")
    return greater(builder, channel, threshold, shape=(1, 1, 30, 30))


def logical_and(builder: GraphBuilder, left: str, right: str, *, shape: Sequence[int]) -> str:
    return str(builder.node("And", [left, right], dtype=onnx.TensorProto.BOOL, shape=shape))


def logical_or(builder: GraphBuilder, left: str, right: str, *, shape: Sequence[int]) -> str:
    return str(builder.node("Or", [left, right], dtype=onnx.TensorProto.BOOL, shape=shape))


def logical_not(builder: GraphBuilder, value: str, *, shape: Sequence[int]) -> str:
    return str(builder.node("Not", [value], dtype=onnx.TensorProto.BOOL, shape=shape))


def cast_mask(builder: GraphBuilder, value: str, *, shape: Sequence[int]) -> str:
    return str(
        builder.node(
            "Cast",
            [value],
            to=onnx.TensorProto.FLOAT,
            dtype=onnx.TensorProto.FLOAT,
            shape=shape,
        )
    )
