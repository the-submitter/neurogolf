"""A compact static-shape builder that remains fully bypassable via `onnx.helper`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from neurogolf.onnx_lab.graph import NameScope, TensorSpec

GRID_SHAPE = (1, 10, 30, 30)
DEFAULT_OPSET = 12


class GraphBuilder:
    """Build static standard-domain graphs while recording every intermediate shape."""

    def __init__(
        self,
        name: str = "neurogolf",
        *,
        opset: int = DEFAULT_OPSET,
        ir_version: int = 10,
    ):
        if opset < 1:
            raise ValueError("opset must be positive")
        self.name = name
        self.opset = opset
        self.ir_version = ir_version
        self.names = NameScope()
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.inputs: list[TensorSpec] = []
        self.outputs: list[TensorSpec] = []
        self.value_info: dict[str, TensorSpec] = {}
        self._initializer_names: set[str] = set()

    @classmethod
    def standard(
        cls,
        name: str = "neurogolf",
        *,
        opset: int = DEFAULT_OPSET,
        ir_version: int = 10,
    ) -> GraphBuilder:
        builder = cls(name, opset=opset, ir_version=ir_version)
        builder.add_input("input", onnx.TensorProto.FLOAT, GRID_SHAPE)
        return builder

    def add_input(self, name: str, dtype: int, shape: Sequence[int]) -> str:
        if any(item.name == name for item in self.inputs):
            raise ValueError(f"Graph input is already declared: {name}")
        if name != "input":
            self.names.reserve(name)
        spec = TensorSpec(name, dtype, tuple(int(item) for item in shape))
        self.inputs.append(spec)
        self.value_info[name] = spec
        return name

    def add_output(self, name: str, dtype: int, shape: Sequence[int]) -> str:
        if any(item.name == name for item in self.outputs):
            raise ValueError(f"Graph output is already declared: {name}")
        if name != "output":
            self.names.reserve(name)
        spec = TensorSpec(name, dtype, tuple(int(item) for item in shape))
        self.outputs.append(spec)
        self.value_info[name] = spec
        return name

    def initializer(
        self,
        value: np.ndarray | Sequence[Any] | int | float | bool,
        *,
        name: str | None = None,
        dtype: np.dtype[Any] | type[Any] | None = None,
    ) -> str:
        array = np.asarray(value, dtype=dtype)
        tensor_name = self.names.unique(name or "constant")
        tensor = onnx.numpy_helper.from_array(array, tensor_name)
        self.initializers.append(tensor)
        self._initializer_names.add(tensor_name)
        return tensor_name

    def node(
        self,
        op_type: str,
        inputs: Sequence[str],
        *,
        output: str | None = None,
        outputs: int = 1,
        name: str | None = None,
        dtype: int | None = None,
        shape: Sequence[int] | None = None,
        output_specs: Sequence[tuple[int, Sequence[int]]] | None = None,
        **attributes: Any,
    ) -> str | tuple[str, ...]:
        if output is not None and outputs != 1:
            raise ValueError("An explicit output name is only valid for one-output nodes")
        if output is not None:
            output_names = [output]
            if output not in {"output"}:
                self.names.reserve(output)
        else:
            output_names = [self.names.unique(f"{op_type.lower()}_out") for _ in range(outputs)]
        node_name = self.names.unique(name or op_type.lower())
        node = onnx.helper.make_node(
            op_type,
            list(inputs),
            output_names,
            name=node_name,
            **attributes,
        )
        self.nodes.append(node)
        specs = output_specs
        if specs is None and dtype is not None and shape is not None:
            specs = [(dtype, shape)]
        if specs is not None:
            if len(specs) != len(output_names):
                raise ValueError("Output metadata count does not match node outputs")
            for tensor_name, (tensor_dtype, tensor_shape) in zip(output_names, specs, strict=True):
                self.value_info[tensor_name] = TensorSpec(
                    tensor_name, tensor_dtype, tuple(int(item) for item in tensor_shape)
                )
        return output_names[0] if outputs == 1 else tuple(output_names)

    def direct_output(self, value: str, *, dtype: int, shape: Sequence[int]) -> str:
        if value == "output":
            self.add_output("output", dtype, shape)
            return value
        self.node("Identity", [value], output="output", dtype=dtype, shape=shape)
        self.add_output("output", dtype, shape)
        return "output"

    def build(self, *, check: bool = True) -> onnx.ModelProto:
        if not self.inputs:
            raise ValueError("Graph has no input")
        if not self.outputs:
            raise ValueError("Graph has no output; call add_output() or direct_output()")
        io_names = {item.name for item in (*self.inputs, *self.outputs)}
        intermediate = [
            spec.value_info()
            for name, spec in self.value_info.items()
            if name not in io_names and name not in self._initializer_names
        ]
        graph = onnx.helper.make_graph(
            self.nodes,
            self.name,
            [item.value_info() for item in self.inputs],
            [item.value_info() for item in self.outputs],
            self.initializers,
            value_info=intermediate,
        )
        model = onnx.helper.make_model(
            graph,
            ir_version=self.ir_version,
            opset_imports=[onnx.helper.make_opsetid("", self.opset)],
        )
        if check:
            onnx.checker.check_model(model, full_check=True)
        return model

    def save(self, path: Path, *, check: bool = True) -> onnx.ModelProto:
        model = self.build(check=check)
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, path)
        return model
