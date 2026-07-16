"""Build the optimized ONNX solution for NeuroGolf task001 / ARC 007bbfb7."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(__file__).with_name("task001.onnx")
SUBMISSION_PATH = ROOT / "submission" / "task001.onnx"


def tensor(name: str, data_type: int, dims: list[int], values: list[float | int]):
    """Create a compact initializer with an explicit ONNX element type."""

    return helper.make_tensor(name, data_type, dims, values)


def build() -> onnx.ModelProto:
    """Return a compact exact graph for the 3x3 Kronecker-fractal rule."""

    nodes = [
        # Extract the logical 3x3 background plane. Background is one and the
        # task foreground is zero in this plane.
        helper.make_node("Slice", ["input", "starts", "ends"], ["background"]),
        # A boolean comparison obtains the foreground mask in only nine bytes;
        # casting it gives the float16 data and dynamic kernel used below.
        helper.make_node("Less", ["background", "half"], ["foreground_mask"]),
        helper.make_node(
            "Cast", ["foreground_mask"], ["stencil"], to=TensorProto.FLOAT16
        ),
        # Stride three arranges the nine scaled kernel copies as a 9x9
        # Kronecker square. Bias -0.5 makes foreground pairs positive and all
        # other logical output cells negative.
        helper.make_node(
            "ConvTranspose",
            ["stencil", "stencil", "spatial_bias"],
            ["spatial_logits"],
            kernel_shape=[3, 3],
            strides=[3, 3],
        ),
        # Reduce every color plane and flip only background's sign. The x/y
        # singleton axes make the result a dynamic 1x1 ConvTranspose kernel.
        helper.make_node(
            "Einsum",
            ["input", "color_signs"],
            ["signed_colors_fp32"],
            equation="bcij,cxy->bcxy",
        ),
        helper.make_node(
            "Cast", ["signed_colors_fp32"], ["signed_colors"], to=TensorProto.FLOAT16
        ),
        # Negative trailing pads extend the logical 9x9 result to the required
        # 30x30 output with zeros, which remain absent under scorer > 0.
        helper.make_node(
            "ConvTranspose",
            ["spatial_logits", "signed_colors"],
            ["output"],
            kernel_shape=[1, 1],
            pads=[0, 0, -21, -21],
        ),
    ]

    initializers = [
        tensor("starts", TensorProto.INT64, [4], [0, 0, 0, 0]),
        tensor("ends", TensorProto.INT64, [4], [1, 1, 3, 3]),
        tensor("half", TensorProto.FLOAT, [1], [0.5]),
        tensor("spatial_bias", TensorProto.FLOAT16, [1], [-0.5]),
        tensor(
            "color_signs",
            TensorProto.FLOAT,
            [10, 1, 1],
            [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task001_fractal",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    model = build()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, MODEL_PATH)
    onnx.save(model, SUBMISSION_PATH)
    print(MODEL_PATH)
    print(SUBMISSION_PATH)


if __name__ == "__main__":
    main()
