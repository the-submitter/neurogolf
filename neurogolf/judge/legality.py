"""Official-first legality inspection with additional structured diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx

from neurogolf.judge.official_adapter import OfficialAdapter


@dataclass
class LegalityResult:
    legal: bool
    file_size_bytes: int | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "legal": self.legal,
            "file_size_bytes": self.file_size_bytes,
            "errors": self.errors,
            "warnings": self.warnings,
            "model_metadata": self.model_metadata,
        }


def inspect_legality(path: Path, adapter: OfficialAdapter) -> LegalityResult:
    errors: list[str] = []
    warnings: list[str] = []
    file_size = path.stat().st_size if path.is_file() else None
    official_valid, official_diagnostic = adapter.check_file(path)
    if not official_valid:
        errors.append(official_diagnostic or "Official file check failed")
        return LegalityResult(False, file_size, errors, warnings)
    try:
        model = onnx.load(path)
        onnx.checker.check_model(model, full_check=True)
    except Exception as error:
        errors.append(f"ONNX checker rejected model: {error}")
        return LegalityResult(False, file_size, errors, warnings)

    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        errors.append("Exactly one graph input and one graph output are required")
    if graph.input and graph.input[0].name != "input":
        errors.append("Graph input must be named 'input'")
    if graph.output and graph.output[0].name != "output":
        errors.append("Graph output must be named 'output'")
    if model.functions:
        errors.append("Model-local functions are prohibited")
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            errors.append(f"Custom opset domain is prohibited: {opset.domain}")
    excluded = set(adapter.excluded_ops)
    tensor_outputs: set[str] = set()
    for node in graph.node:
        if node.op_type.upper() in excluded or "Sequence" in node.op_type:
            errors.append(f"Prohibited operator: {node.op_type}")
        if any("kernel_time" in output for output in node.output):
            errors.append("Tensor names containing 'kernel_time' are prohibited")
        for output in node.output:
            if output and output in tensor_outputs:
                errors.append(f"Duplicate tensor output name: {output}")
            tensor_outputs.add(output)
        for attribute in node.attribute:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                errors.append(f"Subgraph attribute is prohibited on {node.op_type}")
    names = [item.name for item in (*graph.input, *graph.value_info, *graph.output)]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        errors.append(f"Duplicate graph type metadata names: {duplicates}")
    initializer_names = {item.name for item in graph.initializer}
    io_names = {item.name for item in (*graph.input, *graph.output)}
    if initializer_names & io_names:
        errors.append("Initializers may not reuse graph input/output names")
    try:
        inferred_graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
        inferred_items = (
            *inferred_graph.input,
            *inferred_graph.value_info,
            *inferred_graph.output,
        )
        inferred_map = {item.name: item for item in inferred_items}
        for item in inferred_items:
            if item.type.HasField("sequence_type"):
                errors.append(f"Sequence tensor metadata is prohibited: {item.name}")
                continue
            if not item.type.HasField("tensor_type"):
                continue
            tensor_type = item.type.tensor_type
            if not tensor_type.HasField("shape"):
                errors.append(f"Tensor lacks a static shape: {item.name}")
                continue
            for dimension in tensor_type.shape.dim:
                if (
                    dimension.HasField("dim_param")
                    or not dimension.HasField("dim_value")
                    or dimension.dim_value <= 0
                ):
                    errors.append(f"Tensor has a dynamic/nonpositive shape: {item.name}")
                    break
        for node in inferred_graph.node:
            for output in node.output:
                if output and output != "output" and output not in inferred_map:
                    errors.append(f"Node output lacks static type metadata: {output}")
    except Exception as error:
        errors.append(f"Strict ONNX shape inference failed: {error}")
    if adapter.calculate_params(model) is None:
        errors.append("Official parameter accounting rejected nonpositive dimensions")

    try:
        sanitized = adapter.sanitize(model)
        adapter.create_session(sanitized)
    except Exception as error:
        errors.append(str(error))

    metadata = {
        "ir_version": model.ir_version,
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
        "operator_types": sorted({node.op_type for node in graph.node}),
    }
    if model.ir_version != 10:
        warnings.append(f"Official starter uses IR version 10; candidate uses {model.ir_version}")
    return LegalityResult(not errors, file_size, errors, warnings, metadata)
