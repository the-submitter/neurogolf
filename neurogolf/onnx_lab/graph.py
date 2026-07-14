"""Graph naming, tensor specifications, and static-shape helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import onnx


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: int
    shape: tuple[int, ...]

    def value_info(self) -> onnx.ValueInfoProto:
        return onnx.helper.make_tensor_value_info(self.name, self.dtype, list(self.shape))


class NameScope:
    """Generate deterministic unique names safe for official sanitization."""

    def __init__(self) -> None:
        self._used: set[str] = {"input", "output"}
        self._counters: dict[str, int] = {}

    @staticmethod
    def clean(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "value"
        return cleaned.replace("kernel_time", "kernel")

    def reserve(self, value: str) -> str:
        if value in self._used and value not in {"input", "output"}:
            raise ValueError(f"ONNX name is already reserved: {value}")
        self._used.add(value)
        return value

    def unique(self, prefix: str) -> str:
        base = self.clean(prefix)
        index = self._counters.get(base, 0)
        while True:
            name = base if index == 0 else f"{base}_{index}"
            index += 1
            if name not in self._used:
                self._used.add(name)
                self._counters[base] = index
                return name


def static_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dimensions: list[int] = []
    for dimension in tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            return None
        dimensions.append(dimension.dim_value)
    return tuple(dimensions)


def tensor_info_map(graph: onnx.GraphProto) -> dict[str, onnx.ValueInfoProto]:
    return {item.name: item for item in (*graph.input, *graph.value_info, *graph.output)}


def replace_inputs(nodes: Iterable[onnx.NodeProto], old: str, new: str) -> None:
    for node in nodes:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new
