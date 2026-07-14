"""Conv, grouped/depthwise Conv, and 1x1 color mapping helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import onnx

from neurogolf.onnx_lab.builder import GraphBuilder


def conv2d(
    builder: GraphBuilder,
    value: str,
    weights: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    pads: Sequence[int] = (0, 0, 0, 0),
    strides: Sequence[int] = (1, 1),
    dilations: Sequence[int] = (1, 1),
    group: int = 1,
    input_shape: Sequence[int] = (1, 10, 30, 30),
    output: str | None = None,
) -> str:
    weight_array = np.asarray(weights, dtype=np.float32)
    if weight_array.ndim != 4:
        raise ValueError("Conv weights must have [out_channels, in/group, kH, kW] shape")
    weight_name = builder.initializer(weight_array, name="conv_weight")
    inputs = [value, weight_name]
    if bias is not None:
        bias_array = np.asarray(bias, dtype=np.float32)
        if bias_array.shape != (weight_array.shape[0],):
            raise ValueError("Conv bias length must equal output channels")
        inputs.append(builder.initializer(bias_array, name="conv_bias"))
    batch, _channels, height, width = (int(item) for item in input_shape)
    kernel_height, kernel_width = weight_array.shape[2:]
    out_height = (
        height + int(pads[0]) + int(pads[2]) - int(dilations[0]) * (kernel_height - 1) - 1
    ) // int(strides[0]) + 1
    out_width = (
        width + int(pads[1]) + int(pads[3]) - int(dilations[1]) * (kernel_width - 1) - 1
    ) // int(strides[1]) + 1
    return str(
        builder.node(
            "Conv",
            inputs,
            output=output,
            dtype=onnx.TensorProto.FLOAT,
            shape=(batch, weight_array.shape[0], out_height, out_width),
            kernel_shape=[kernel_height, kernel_width],
            pads=list(pads),
            strides=list(strides),
            dilations=list(dilations),
            group=group,
        )
    )


def grouped_conv2d(
    builder: GraphBuilder,
    value: str,
    weights: np.ndarray,
    *,
    groups: int,
    **kwargs: object,
) -> str:
    # Kept flexible for parity with `conv2d`; cast-free validation occurs there.
    return conv2d(builder, value, weights, group=groups, **kwargs)  # type: ignore[arg-type]


def depthwise_conv2d(
    builder: GraphBuilder,
    value: str,
    kernels: np.ndarray,
    *,
    pads: Sequence[int] = (0, 0, 0, 0),
    input_shape: Sequence[int] = (1, 10, 30, 30),
    output: str | None = None,
) -> str:
    kernel_array = np.asarray(kernels, dtype=np.float32)
    if kernel_array.ndim == 3:
        kernel_array = kernel_array[:, None, :, :]
    if kernel_array.ndim != 4 or kernel_array.shape[1] != 1:
        raise ValueError("Depthwise kernels must have [channels, 1, kH, kW] shape")
    return conv2d(
        builder,
        value,
        kernel_array,
        pads=pads,
        group=kernel_array.shape[0],
        input_shape=input_shape,
        output=output,
    )


def color_map_1x1(
    builder: GraphBuilder,
    value: str,
    mapping: Mapping[int, int] | Sequence[int],
    *,
    output: str | None = None,
) -> str:
    table = (
        [mapping[index] for index in range(10)] if isinstance(mapping, Mapping) else list(mapping)
    )
    if len(table) != 10 or any(not 0 <= color <= 9 for color in table):
        raise ValueError("Color map must define ten target colors in range 0..9")
    weights = np.full((10, 10, 1, 1), -1.0, dtype=np.float32)
    for source, target in enumerate(table):
        weights[target, source, 0, 0] = 1.0
    return conv2d(builder, value, weights, output=output)
