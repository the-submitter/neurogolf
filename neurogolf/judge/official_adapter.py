"""Thin, parity-tested adapter around the immutable official competition utility."""

from __future__ import annotations

import importlib.util
import io
import math
import os
import sys
import tempfile
import threading
import time
from collections.abc import Iterable
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from neurogolf.config.models import NeuroGolfSettings
from neurogolf.errors import CandidateError, IntegrityError
from neurogolf.tasks.integrity import load_manifest, sha256_file
from neurogolf.tasks.models import ArcExample, Grid, validate_grid

_MODULE_CACHE: dict[tuple[Path, str], ModuleType] = {}
_MODULE_LOCK = threading.Lock()


@dataclass(frozen=True)
class OfficialScore:
    memory_bytes: int
    parameter_elements: int
    objective: int
    points: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "memory_bytes": self.memory_bytes,
            "parameter_elements": self.parameter_elements,
            "objective": self.objective,
            "points": self.points,
        }


def _copy_model(model: onnx.ModelProto) -> onnx.ModelProto:
    copied = onnx.ModelProto()
    copied.CopyFrom(model)
    return copied


def _load_module(path: Path, digest: str) -> ModuleType:
    key = (path.resolve(), digest)
    with _MODULE_LOCK:
        if key in _MODULE_CACHE:
            return _MODULE_CACHE[key]
        name = f"_neurogolf_official_{digest[:16]}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise IntegrityError(f"Could not import official utility from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            raise IntegrityError(
                f"Official utility import failed: {type(error).__name__}: {error}"
            ) from error
        finally:
            sys.modules.pop(name, None)
        _MODULE_CACHE[key] = module
        return module


class OfficialAdapter:
    """Expose official behavior with typed errors and no source modification."""

    def __init__(self, settings: NeuroGolfSettings, *, verify_hash: bool = True) -> None:
        self.settings = settings
        self.path = settings.paths.official_utils
        self.sha256 = sha256_file(self.path)
        if verify_hash:
            manifest = load_manifest(settings)
            relative = str(self.path.relative_to(settings.paths.repository_root))
            expected = manifest.get("files", {}).get(relative)
            if expected != self.sha256:
                raise IntegrityError(
                    f"Official utility hash mismatch: expected {expected}, found {self.sha256}"
                )
        if "MPLCONFIGDIR" not in os.environ:
            cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            matplotlib_cache = cache_root / "matplotlib"
            try:
                matplotlib_cache.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            else:
                os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
        self.module = _load_module(self.path, self.sha256)

    @property
    def excluded_ops(self) -> tuple[str, ...]:
        return tuple(self.module._EXCLUDED_OP_TYPES)  # noqa: SLF001

    @property
    def file_size_limit(self) -> int:
        return int(self.module._FILESIZE_LIMIT_IN_BYTES)  # noqa: SLF001

    def check_file(self, path: Path) -> tuple[bool, str]:
        output = io.StringIO()
        with redirect_stdout(output):
            valid = bool(self.module.check_network(str(path)))
        return valid, output.getvalue().strip()

    def convert_example(self, example: ArcExample | dict[str, Any]) -> dict[str, np.ndarray]:
        payload = example.model_dump() if isinstance(example, ArcExample) else example
        result = self.module.convert_to_numpy(payload)
        if result is None:
            raise CandidateError("Official conversion rejected a grid larger than 30x30")
        return result

    def grid_to_tensor(self, grid: Grid) -> np.ndarray:
        normalized = validate_grid(grid)
        # Calling the canonical conversion keeps indexing/dtype behavior exact.
        pair = {"input": normalized, "output": normalized}
        return self.convert_example(pair)["input"]

    def tensor_to_grid(self, tensor: np.ndarray) -> Grid:
        return self.module.convert_from_numpy(tensor)

    @staticmethod
    def threshold_logits(tensor: np.ndarray) -> np.ndarray:
        return (tensor > 0.0).astype(float)

    def sanitize(self, model: onnx.ModelProto) -> onnx.ModelProto:
        sanitized = self.module.sanitize_model(_copy_model(model))
        if sanitized is None:
            raise CandidateError("Official model sanitization rejected a tensor name")
        return sanitized

    def load_and_sanitize(self, path: Path) -> onnx.ModelProto:
        valid, diagnostic = self.check_file(path)
        if not valid:
            raise CandidateError(diagnostic or f"Official file check rejected {path}")
        try:
            model = onnx.load(path)
        except Exception as error:
            raise CandidateError(f"Could not load ONNX model {path}: {error}") from error
        return self.sanitize(model)

    @staticmethod
    def session_options(
        *, profiling: bool = False, profile_prefix: Path | None = None
    ) -> ort.SessionOptions:
        options = ort.SessionOptions()
        options.enable_profiling = profiling
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if profile_prefix is not None:
            options.profile_file_prefix = str(profile_prefix)
        return options

    def create_session(
        self,
        model: onnx.ModelProto,
        *,
        profiling: bool = False,
        profile_prefix: Path | None = None,
    ) -> ort.InferenceSession:
        try:
            return ort.InferenceSession(
                model.SerializeToString(),
                self.session_options(profiling=profiling, profile_prefix=profile_prefix),
                providers=["CPUExecutionProvider"],
            )
        except Exception as error:
            raise CandidateError(f"ONNX Runtime rejected the model: {error}") from error

    def run_session(self, session: ort.InferenceSession, tensor: np.ndarray) -> np.ndarray:
        try:
            return self.module.run_network(session, tensor)
        except Exception as error:
            raise CandidateError(f"Candidate runtime failed: {error}") from error

    def calculate_params(self, model: onnx.ModelProto) -> int | None:
        value = self.module.calculate_params(model)
        return None if value is None else int(value)

    def score_model(
        self,
        model_or_path: onnx.ModelProto | Path,
        grids: Iterable[Grid] = (),
        *,
        deadline: float | None = None,
    ) -> OfficialScore:
        sanitized = (
            self.load_and_sanitize(model_or_path)
            if isinstance(model_or_path, Path)
            else self.sanitize(model_or_path)
        )
        with tempfile.TemporaryDirectory(prefix="neurogolf-profile-") as temporary:
            prefix = Path(temporary) / "official"
            session = self.create_session(sanitized, profiling=True, profile_prefix=prefix)
            ran = False
            for grid in grids:
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("Official scoring deadline exceeded")
                self.run_session(session, self.grid_to_tensor(grid))
                ran = True
            if not ran:
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("Official scoring deadline exceeded")
                zero = np.zeros((1, 10, 30, 30), dtype=np.float32)
                self.run_session(session, zero)
            trace_path = session.end_profiling()
            with redirect_stdout(io.StringIO()):
                memory, params = self.module.score_network(sanitized, trace_path)
            if memory is None or params is None or memory < 0 or params < 0:
                raise CandidateError("Official scorer could not measure this model")
            objective = int(memory) + int(params)
            return OfficialScore(
                memory_bytes=int(memory),
                parameter_elements=int(params),
                objective=objective,
                points=max(1.0, 25.0 - math.log(max(1.0, objective))),
            )

    def describe_contract(self) -> dict[str, Any]:
        return {
            "official_utils_sha256": self.sha256,
            "shape": [1, 10, 30, 30],
            "dtype": "float32",
            "threshold": "> 0.0",
            "ir_version": int(self.module._IR_VERSION),  # noqa: SLF001
            "opsets": [
                {"domain": item.domain, "version": item.version}
                for item in self.module._OPSET_IMPORTS  # noqa: SLF001
            ],
            "excluded_ops": list(self.excluded_ops),
            "file_size_limit_bytes": self.module._FILESIZE_LIMIT_IN_BYTES,  # noqa: SLF001
            "graph_optimizations": "disabled",
            "objective": "intermediate_tensor_bytes + parameter_elements",
            "points": "max(1, 25 - log(max(1, objective)))",
        }
