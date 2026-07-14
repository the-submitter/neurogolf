"""Generic legal models used for infrastructure tests, never task solutions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import onnx

from neurogolf.onnx_lab.builder import GraphBuilder
from neurogolf.onnx_lab.conv import conv2d


def identity_model() -> onnx.ModelProto:
    builder = GraphBuilder.standard("identity_fixture")
    builder.node(
        "Identity",
        ["input"],
        output="output",
        dtype=onnx.TensorProto.FLOAT,
        shape=(1, 10, 30, 30),
    )
    builder.add_output("output", onnx.TensorProto.FLOAT, (1, 10, 30, 30))
    return builder.build()


def single_conv_model(
    weight_fn: Callable[[int, int, tuple[int, int]], float] | None = None,
    *,
    kernel_size: int = 1,
) -> onnx.ModelProto:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    center = kernel_size // 2
    weights = np.zeros((10, 10, kernel_size, kernel_size), dtype=np.float32)
    for out_channel in range(10):
        for in_channel in range(10):
            for row in range(kernel_size):
                for col in range(kernel_size):
                    if weight_fn is None:
                        weights[out_channel, in_channel, row, col] = float(
                            out_channel == in_channel and row == center and col == center
                        )
                    else:
                        weights[out_channel, in_channel, row, col] = weight_fn(
                            out_channel, in_channel, (row - center, col - center)
                        )
    builder = GraphBuilder.standard("single_conv_fixture")
    conv2d(
        builder,
        "input",
        weights,
        pads=(center, center, center, center),
        output="output",
    )
    builder.add_output("output", onnx.TensorProto.FLOAT, (1, 10, 30, 30))
    return builder.build()
