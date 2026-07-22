"""Prototype the sparse, single-Einsum task001 graph."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent


def sparse(name: str, dims: list[int], coordinates: list[list[int]], values: list[float]):
    value_tensor = numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)
    index_tensor = numpy_helper.from_array(np.asarray(coordinates, dtype=np.int64))
    return helper.make_sparse_tensor(value_tensor, index_tensor, dims)


def build() -> onnx.ModelProto:
    # q=0 is the constant/background term; q=1..9 are color indicators.
    basis_coordinates = [[color, 0] for color in range(10)]
    basis_coordinates += [[color, color] for color in range(1, 10)]
    basis = sparse(
        "safe_name_0",
        [10, 10],
        basis_coordinates,
        [1.0] * len(basis_coordinates),
    )

    response_coordinates = [[0, 0]]
    response_values = [1.0]
    for color in range(1, 10):
        response_coordinates.extend([[0, color], [color, color]])
        response_values.extend([-1.0, 1.0])
    response = sparse(
        "safe_name_1", [10, 10], response_coordinates, response_values
    )

    position_coordinates = []
    for outer in range(3):
        for inner in range(3):
            position_coordinates.append([outer, inner, 3 * outer + inner])
    position = sparse(
        "safe_name_2", [30, 30, 30], position_coordinates, [1.0] * 9
    )

    node = helper.make_node(
        "Einsum",
        [
            "input",
            "input",
            "safe_name_0",
            "safe_name_0",
            "safe_name_1",
            "safe_name_2",
            "safe_name_2",
        ],
        ["output"],
        equation="bdij,bekl,dq,eq,cq,ikr,jls->bcrs",
    )
    graph = helper.make_graph(
        [node],
        "task001_sparse_einsum",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    )
    graph.sparse_initializer.extend([basis, response, position])
    model = helper.make_model(
        graph, ir_version=10, opset_imports=[helper.make_opsetid("", 12)]
    )
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    path = HERE / "candidate_sparse.onnx"
    onnx.save(build(), path)
    print(path)
