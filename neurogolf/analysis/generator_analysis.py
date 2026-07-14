"""ARC-GEN source/signature/AST summaries and write-relationship hints."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from neurogolf.arcgen.importer import ImportedGenerator
from neurogolf.arcgen.tracing import source_ast_summary


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def analyze_generator(imported: ImportedGenerator) -> dict[str, Any]:
    path: Path = imported.metadata.source_path
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    writes: dict[str, int] = {}
    returns: list[list[str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                root = _root_name(target)
                if root in {"grid", "ingrid", "output"}:
                    writes[root] = writes.get(root, 0) + 1
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys = [
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            if keys:
                returns.append(keys)
    return {
        "source": source,
        "source_sha256": imported.metadata.source_sha256,
        "generate_signature": imported.metadata.generate_signature,
        "validate_signature": imported.metadata.validate_signature,
        "ast_summary": source_ast_summary(path),
        "write_relationships": writes,
        "returned_dictionary_keys": returns,
    }
