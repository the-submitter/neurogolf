"""Build the task 001 dynamic Kronecker-stencil network."""

from pathlib import Path

import numpy as np
import onnx

from neurogolf.onnx_lab.builder import GraphBuilder


def build() -> onnx.ModelProto:
    """Return a compact static graph for ARC task 007bbfb7."""

    # Opset 9 keeps Slice coordinates as unscored attributes. Both
    # ConvTranspose schemas used below are valid from opset 1 onward.
    builder = GraphBuilder.standard("task001_dynamic_kronecker", opset=9)

    background = str(
        builder.node(
            "Slice",
            ["input"],
            starts=[0, 0, 0, 0],
            ends=[1, 1, 3, 3],
            dtype=onnx.TensorProto.FLOAT,
            shape=(1, 1, 3, 3),
        )
    )

    # The background channel is 0 on foreground cells and 1 elsewhere.
    # Encoding it as -2/-1 makes its dynamic transposed-convolution outer
    # product 4/2/1 for foreground-foreground/mixed/background-background.
    minus_two = builder.initializer(
        np.asarray([-2.0], dtype=np.float32), name="minus_two"
    )
    encoded_stencil = str(
        builder.node(
            "Add",
            [background, minus_two],
            dtype=onnx.TensorProto.FLOAT,
            shape=(1, 1, 3, 3),
        )
    )
    spatial_bias = builder.initializer(
        np.asarray([-2.5], dtype=np.float32), name="spatial_bias"
    )
    spatial_logits = str(
        builder.node(
            "ConvTranspose",
            [encoded_stencil, encoded_stencil, spatial_bias],
            kernel_shape=[3, 3],
            strides=[3, 3],
            dtype=onnx.TensorProto.FLOAT,
            shape=(1, 1, 9, 9),
        )
    )

    # Reduce the input to the channels that are present. Negating only the
    # background channel makes the 9x9 scalar logits serve both roles:
    # positive spatial logits select the active color, while negative logits
    # select background. A full stencil has no background presence and an
    # empty stencil has no foreground presence, so both boundaries follow the
    # same rule without special cases.
    presence = str(
        builder.node(
            "ReduceMax",
            ["input"],
            axes=[2, 3],
            keepdims=1,
            dtype=onnx.TensorProto.FLOAT,
            shape=(1, 10, 1, 1),
        )
    )
    color_signs = builder.initializer(
        np.asarray([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        .reshape(1, 10, 1, 1),
        name="color_signs",
    )
    signed_colors = str(
        builder.node(
            "Mul",
            [presence, color_signs],
            dtype=onnx.TensorProto.FLOAT,
            shape=(1, 10, 1, 1),
        )
    )

    # Use the signed color vector as a dynamic 1x1 kernel. Negative trailing
    # pads extend the natural 9x9 result to 30x30 with exact zeros, which stay
    # absent under the official strict > 0 threshold.
    builder.node(
        "ConvTranspose",
        [spatial_logits, signed_colors],
        output="output",
        kernel_shape=[1, 1],
        pads=[0, 0, -21, -21],
        dtype=onnx.TensorProto.FLOAT,
        shape=(1, 10, 30, 30),
    )
    builder.add_output("output", onnx.TensorProto.FLOAT, (1, 10, 30, 30))
    return builder.build()


if __name__ == "__main__":
    destination = Path(__file__).with_name("candidate.onnx")
    model = build()
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, destination)
    print(destination)
