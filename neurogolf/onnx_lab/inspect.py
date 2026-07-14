"""Model structure and per-node/tensor scorer-cost inspection."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import onnx

from neurogolf.onnx_lab.score_estimator import estimate_score


def fusion_opportunities(model: onnx.ModelProto) -> list[dict[str, str]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    opportunities: list[dict[str, str]] = []
    for node in model.graph.node:
        if not node.output:
            continue
        downstream = consumers.get(node.output[0], [])
        if len(downstream) != 1:
            continue
        following = downstream[0]
        pair = (node.op_type, following.op_type)
        if pair in {
            ("Cast", "Where"),
            ("Transpose", "Transpose"),
            ("Add", "Greater"),
            ("Mul", "Add"),
        }:
            opportunities.append(
                {
                    "producer": node.name,
                    "consumer": following.name,
                    "pattern": f"{pair[0]}->{pair[1]}",
                }
            )
    return opportunities


def inspect_model(model_or_path: onnx.ModelProto | Path) -> dict[str, Any]:
    model = onnx.load(model_or_path) if isinstance(model_or_path, Path) else model_or_path
    onnx.checker.check_model(model, full_check=True)
    operators = Counter(node.op_type for node in model.graph.node)
    return {
        "ir_version": model.ir_version,
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "file_size_bytes": model_or_path.stat().st_size
        if isinstance(model_or_path, Path)
        else len(model.SerializeToString()),
        "node_count": len(model.graph.node),
        "operators": dict(sorted(operators.items())),
        "nodes": [
            {
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
            for node in model.graph.node
        ],
        "local_cost": estimate_score(model),
        "fusion_opportunities": fusion_opportunities(model),
    }
