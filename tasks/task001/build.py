"""Build the optimized ONNX solution for NeuroGolf task001 / ARC 007bbfb7."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(__file__).with_name("task001.onnx")
SUBMISSION_PATH = ROOT / "submission" / "task001.onnx"


def build() -> onnx.ModelProto:
    """Return the zero-intermediate, 129-parameter terminal contraction."""

    # For r < 9, row r is the homogeneous base-3 coordinate
    # [1, floor(r / 3), r mod 3]. Rows in the padded tail stay zero.
    coordinates = np.zeros((30, 3), dtype=np.float32)
    for output_coordinate in range(9):
        coordinates[output_coordinate] = [
            1.0,
            output_coordinate // 3,
            output_coordinate % 3,
        ]

    # Select either the outer or inner base-3 digit and produce the three
    # linear forms [1, x, 1+x]. Squaring these forms spans every quadratic.
    basis = np.zeros((2, 3, 3), dtype=np.float32)
    basis[0] = np.asarray([[1, 0, 1], [0, 1, 1], [0, 0, 0]])
    basis[1] = np.asarray([[1, 0, 1], [0, 0, 0], [0, 1, 1]])

    # Decode [1, x^2, (1+x)^2] into the three Lagrange indicators
    # [x == 0, x == 1, x == 2].
    power_to_lagrange = np.asarray(
        [[1.0, 0.0, 0.0], [-1.5, 2.0, -0.5], [0.5, -1.0, 0.5]],
        dtype=np.float64,
    )
    square_to_power = np.asarray(
        [[1.0, 0.0, 0.0], [-0.5, -0.5, 0.5], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    decoder = (square_to_power.T @ power_to_lagrange).astype(np.float32)

    initializers = [
        # Background must remain positive unless both selected stencil cells
        # are foreground; foreground channels require the opposite sign.
        numpy_helper.from_array(
            np.asarray([2.0] + [-1.0] * 9, dtype=np.float32), "channel_sign"
        ),
        numpy_helper.from_array(coordinates, "coordinate"),
        numpy_helper.from_array(basis, "basis"),
        numpy_helper.from_array(decoder, "decoder"),
        # Source coordinates p and q use their inner base-3 digit.
        numpy_helper.from_array(np.asarray([0.0, 1.0], dtype=np.float32), "inner"),
    ]

    # The four coordinate groups synthesize exact one-hot relations for:
    # output row r, source row p, output column s, and source column q.
    # Sharing u between r and s selects the outer pair together or the inner
    # pair together. All other labels contract, so the only node is terminal.
    equation = (
        "ncxy,ndpq,c,d,"
        "rh,rk,uhA,ukA,Ai,"
        "pl,pm,vlB,vmB,Bi,v,"
        "sj,sz,ujC,uzC,Cf,"
        "qe,qo,weD,woD,Df,w"
        "->ncrs"
    )
    node = helper.make_node(
        "Einsum",
        [
            "input", "input", "channel_sign", "channel_sign",
            "coordinate", "coordinate", "basis", "basis", "decoder",
            "coordinate", "coordinate", "basis", "basis", "decoder", "inner",
            "coordinate", "coordinate", "basis", "basis", "decoder",
            "coordinate", "coordinate", "basis", "basis", "decoder", "inner",
        ],
        ["output"],
        equation=equation,
    )

    graph = helper.make_graph(
        [node],
        "task001_fractal_terminal_einsum",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
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
    print(f"saved {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")
    print(f"saved {SUBMISSION_PATH} ({SUBMISSION_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
