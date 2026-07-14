"""Absolute-path ARC-GEN imports with isolated names and actionable diagnostics."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from neurogolf.errors import GeneratorError
from neurogolf.tasks.models import TaskRecord

_IMPORT_LOCK = threading.RLock()


@dataclass(frozen=True)
class GeneratorMetadata:
    """Static provenance and signatures captured at import time."""

    task_id: str
    source_path: Path
    source_sha256: str
    module_name: str
    generate_signature: str
    validate_signature: str
    generate_doc: str | None
    unusual_signature: bool


@dataclass
class ImportedGenerator:
    """Loaded generator callables kept independently of `sys.modules`."""

    module: ModuleType
    generate: Callable[..., dict[str, Any]]
    validate: Callable[..., dict[str, Any]]
    metadata: GeneratorMetadata
    common_module: ModuleType


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GeneratorError(f"Could not create import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous_bytecode = sys.dont_write_bytecode
    try:
        # Canonical submodules are read-only source material; importing them must
        # not leave __pycache__ artifacts in the ARC-GEN working tree.
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    except Exception as error:
        raise GeneratorError(f"Failed to import {path}: {type(error).__name__}: {error}") from error
    finally:
        sys.dont_write_bytecode = previous_bytecode
        sys.modules.pop(name, None)
    return module


def _has_required_parameters(signature: inspect.Signature) -> bool:
    return any(
        parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    )


def import_generator(record: TaskRecord, arcgen_root: Path) -> ImportedGenerator:
    """Load one canonical task module without relying on the current directory."""

    root = arcgen_root.expanduser().resolve()
    source = record.arcgen_generator_path.expanduser().resolve()
    common_path = root / "common.py"
    if source.parent != root / "tasks":
        raise GeneratorError(f"Generator is outside canonical task directory: {source}")
    if not source.is_file() or not common_path.is_file():
        raise GeneratorError(f"Missing generator or ARC-GEN common module for {record.task_id}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    unique_name = f"_neurogolf_arcgen_{record.task_id}_{digest[:12]}"
    common_name = f"_neurogolf_arcgen_common_{digest[:12]}"

    with _IMPORT_LOCK:
        previous_common = sys.modules.get("common")
        previous_path = list(sys.path)
        try:
            common_module = _load_module(common_name, common_path)
            sys.modules["common"] = common_module
            sys.path.insert(0, str(root))
            module = _load_module(unique_name, source)
        finally:
            sys.path[:] = previous_path
            if previous_common is None:
                sys.modules.pop("common", None)
            else:
                sys.modules["common"] = previous_common

    generate = getattr(module, "generate", None)
    validate = getattr(module, "validate", None)
    if not callable(generate) or not callable(validate):
        raise GeneratorError(f"{source} must expose callable generate() and validate()")
    generate_signature = inspect.signature(generate)
    validate_signature = inspect.signature(validate)
    required_generate = _has_required_parameters(generate_signature)
    required_validate = _has_required_parameters(validate_signature)
    if required_generate or required_validate:
        raise GeneratorError(
            f"Unsupported signatures in {source}: generate{generate_signature}, "
            f"validate{validate_signature}; zero-argument invocation must be supported"
        )
    metadata = GeneratorMetadata(
        task_id=record.task_id,
        source_path=source,
        source_sha256=digest,
        module_name=unique_name,
        generate_signature=str(generate_signature),
        validate_signature=str(validate_signature),
        generate_doc=inspect.getdoc(generate),
        unusual_signature=bool(
            any(
                parameter.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL)
                for parameter in generate_signature.parameters.values()
            )
        ),
    )
    generate_call = cast(Callable[..., dict[str, Any]], generate)
    validate_call = cast(Callable[..., dict[str, Any]], validate)
    return ImportedGenerator(module, generate_call, validate_call, metadata, common_module)


def import_lock() -> threading.RLock:
    """Expose the shared lock so seeded calls can protect global random state."""

    return _IMPORT_LOCK
