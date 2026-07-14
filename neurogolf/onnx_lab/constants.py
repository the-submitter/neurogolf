"""Initializer and Constant-node creation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import onnx

from neurogolf.onnx_lab.builder import GraphBuilder


def initializer(
    builder: GraphBuilder,
    value: np.ndarray | Sequence[Any] | int | float | bool,
    *,
    name: str | None = None,
    dtype: np.dtype[Any] | type[Any] | None = None,
) -> str:
    return builder.initializer(value, name=name, dtype=dtype)


def constant_node(
    builder: GraphBuilder,
    value: np.ndarray | Sequence[Any] | int | float | bool,
    *,
    name: str | None = None,
) -> str:
    array = np.asarray(value)
    tensor = onnx.numpy_helper.from_array(array)
    return str(
        builder.node(
            "Constant",
            [],
            name=name or "constant_node",
            dtype=tensor.data_type,
            shape=array.shape,
            value=tensor,
        )
    )


def scalar(builder: GraphBuilder, value: int | float | bool, *, name: str = "scalar") -> str:
    return builder.initializer(np.asarray(value), name=name)
