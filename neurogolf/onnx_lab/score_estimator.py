"""Local static cost estimate; the official profiler remains final truth."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from neurogolf.judge.official_adapter import OfficialAdapter, OfficialScore
from neurogolf.onnx_lab.graph import tensor_info_map
from neurogolf.tasks.models import Grid


def parameter_elements(model: onnx.ModelProto) -> int | None:
    total = 0
    for initializer in (*model.graph.initializer, *model.graph.sparse_initializer):
        tensor = (
            initializer.values if isinstance(initializer, onnx.SparseTensorProto) else initializer
        )
        if any(dimension <= 0 for dimension in tensor.dims):
            return None
        total += math.prod(tensor.dims)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                if any(dimension <= 0 for dimension in attribute.t.dims):
                    return None
                total += math.prod(attribute.t.dims)
            elif attribute.name == "sparse_value":
                values = attribute.sparse_tensor.values
                if any(dimension <= 0 for dimension in values.dims):
                    return None
                total += math.prod(values.dims)
            elif attribute.name == "value_floats":
                total += len(attribute.floats)
            elif attribute.name == "value_ints":
                total += len(attribute.ints)
            elif attribute.name == "value_strings":
                total += len(attribute.strings)
    return total


def static_tensor_costs(model: onnx.ModelProto) -> list[dict[str, Any]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    info = tensor_info_map(inferred.graph)
    initializer_names = {item.name for item in inferred.graph.initializer}
    producers = {
        output: {"node": node.name, "op_type": node.op_type}
        for node in inferred.graph.node
        for output in node.output
        if output
    }
    costs: list[dict[str, Any]] = []
    names = set(producers) | set(info)
    for name in sorted(names):
        if name in {"input", "output"} or name in initializer_names or name not in info:
            continue
        tensor_type = info[name].type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int] = []
        valid = True
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
                valid = False
                break
            dims.append(dimension.dim_value)
        if not valid:
            continue
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        byte_count = math.prod(dims) * np.dtype(dtype).itemsize
        costs.append(
            {
                "tensor": name,
                "shape": dims,
                "dtype": str(np.dtype(dtype)),
                "bytes": byte_count,
                **producers.get(name, {}),
            }
        )
    return costs


def estimate_score(model: onnx.ModelProto) -> dict[str, Any]:
    tensors = static_tensor_costs(model)
    memory = sum(item["bytes"] for item in tensors)
    params = parameter_elements(model)
    if params is None:
        return {"measurable": False, "reason": "invalid parameter dimensions", "tensors": tensors}
    objective = memory + params
    return {
        "measurable": True,
        "intermediate_tensor_bytes": memory,
        "parameter_elements": params,
        "objective": objective,
        "points": max(1.0, 25.0 - math.log(max(1.0, objective))),
        "tensors": tensors,
        "note": "Static estimate only; use the official profiler for final truth.",
    }


def official_score(
    adapter: OfficialAdapter, model_or_path: onnx.ModelProto | Path, grids: list[Grid]
) -> OfficialScore:
    return adapter.score_model(model_or_path, grids)
