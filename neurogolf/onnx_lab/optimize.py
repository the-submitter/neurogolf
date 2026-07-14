"""Conservative graph rewrites aimed at official memory/parameter cost."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

import onnx

from neurogolf.onnx_lab.graph import replace_inputs


def _replace_all(model: onnx.ModelProto, old: str, new: str) -> None:
    replace_inputs(model.graph.node, old, new)
    for output in model.graph.output:
        if output.name == old:
            output.name = new


def fold_constant_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    keep: list[onnx.NodeProto] = []
    for node in result.graph.node:
        value_attributes = [attribute for attribute in node.attribute if attribute.name == "value"]
        if (
            node.op_type == "Constant"
            and len(node.output) == 1
            and node.output[0] != "output"
            and len(value_attributes) == 1
        ):
            tensor = copy.deepcopy(value_attributes[0].t)
            tensor.name = node.output[0]
            result.graph.initializer.append(tensor)
        else:
            keep.append(node)
    del result.graph.node[:]
    result.graph.node.extend(keep)
    return result


def remove_identity_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    removable: list[onnx.NodeProto] = []
    for node in result.graph.node:
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            continue
        if node.output[0] == "output":
            continue
        _replace_all(result, node.output[0], node.input[0])
        removable.append(node)
    keep = [node for node in result.graph.node if node not in removable]
    del result.graph.node[:]
    result.graph.node.extend(keep)
    return result


def remove_dead_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    nodes = list(result.graph.node)
    by_output = {
        output: index for index, node in enumerate(nodes) for output in node.output if output
    }
    needed_values = {output.name for output in result.graph.output}
    needed_nodes: set[int] = set()
    stack = list(needed_values)
    while stack:
        value = stack.pop()
        index = by_output.get(value)
        if index is None or index in needed_nodes:
            continue
        needed_nodes.add(index)
        stack.extend(nodes[index].input)
    keep = [node for index, node in enumerate(nodes) if index in needed_nodes]
    used_inputs = {value for node in keep for value in node.input}
    initializers = [item for item in result.graph.initializer if item.name in used_inputs]
    del result.graph.node[:]
    result.graph.node.extend(keep)
    del result.graph.initializer[:]
    result.graph.initializer.extend(initializers)
    produced = {value for node in keep for value in node.output}
    info = [
        item
        for item in result.graph.value_info
        if item.name in produced or item.name in used_inputs
    ]
    del result.graph.value_info[:]
    result.graph.value_info.extend(info)
    return result


def _permutation(node: onnx.NodeProto, rank: int) -> list[int]:
    attribute = next((item for item in node.attribute if item.name == "perm"), None)
    return list(attribute.ints) if attribute is not None else list(range(rank - 1, -1, -1))


def remove_redundant_transposes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Eliminate adjacent transpose pairs whose composed permutation is identity."""

    result = copy.deepcopy(model)
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in result.graph.node:
        for value in node.input:
            consumers[value].append(node)
    remove_names: set[str] = set()
    for first in result.graph.node:
        if first.op_type != "Transpose" or len(first.input) != 1 or len(first.output) != 1:
            continue
        downstream = consumers.get(first.output[0], [])
        if len(downstream) != 1:
            continue
        second = downstream[0]
        if second.op_type != "Transpose" or len(second.output) != 1:
            continue
        rank = len(_permutation(first, 4))
        first_perm = _permutation(first, rank)
        second_perm = _permutation(second, rank)
        composed = [first_perm[index] for index in second_perm]
        if composed != list(range(rank)):
            continue
        if second.output[0] == "output":
            second.op_type = "Identity"
            del second.attribute[:]
            second.input[:] = [first.input[0]]
        else:
            _replace_all(result, second.output[0], first.input[0])
            remove_names.add(second.name)
        remove_names.add(first.name)
    keep = [node for node in result.graph.node if node.name not in remove_names]
    del result.graph.node[:]
    result.graph.node.extend(keep)
    return result


def eliminate_common_subexpressions(model: onnx.ModelProto) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    seen: dict[tuple[Any, ...], str] = {}
    keep: list[onnx.NodeProto] = []
    for node in result.graph.node:
        if len(node.output) != 1 or node.op_type in {"RandomNormal", "RandomUniform"}:
            keep.append(node)
            continue
        attributes = tuple(attribute.SerializeToString() for attribute in node.attribute)
        key = (node.op_type, node.domain, tuple(node.input), attributes)
        previous = seen.get(key)
        if previous is None:
            seen[key] = node.output[0]
            keep.append(node)
        elif node.output[0] != "output":
            _replace_all(result, node.output[0], previous)
        else:
            keep.append(node)
    del result.graph.node[:]
    result.graph.node.extend(keep)
    return result


def remove_redundant_casts(model: onnx.ModelProto) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    try:
        inferred = onnx.shape_inference.infer_shapes(result, strict_mode=True)
    except Exception:
        return result
    types = {
        item.name: item.type.tensor_type.elem_type
        for item in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
        if item.type.HasField("tensor_type")
    }
    remove: list[onnx.NodeProto] = []
    for node in result.graph.node:
        if node.op_type != "Cast" or len(node.input) != 1 or len(node.output) != 1:
            continue
        target = next((attribute.i for attribute in node.attribute if attribute.name == "to"), None)
        if target is not None and types.get(node.input[0]) == target and node.output[0] != "output":
            _replace_all(result, node.output[0], node.input[0])
            remove.append(node)
    keep = [node for node in result.graph.node if node not in remove]
    del result.graph.node[:]
    result.graph.node.extend(keep)
    return result


def optimize_model(model: onnx.ModelProto) -> onnx.ModelProto:
    result = fold_constant_nodes(model)
    result = remove_redundant_casts(result)
    result = remove_redundant_transposes(result)
    result = remove_identity_nodes(result)
    result = eliminate_common_subexpressions(result)
    result = remove_dead_nodes(result)
    onnx.checker.check_model(result, full_check=True)
    return result
