from __future__ import annotations

import os
from pathlib import Path

import onnx
import pytest

from neurogolf.config import load_settings
from neurogolf.config.models import NeuroGolfSettings
from neurogolf.judge.official_adapter import OfficialAdapter

os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "matplotlib"))


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings(repository_root: Path) -> NeuroGolfSettings:
    return load_settings(repository_root=repository_root)


@pytest.fixture(scope="session")
def official_adapter(settings: NeuroGolfSettings) -> OfficialAdapter:
    return OfficialAdapter(settings, verify_hash=False)


@pytest.fixture
def save_model(tmp_path: Path):
    def save(model: onnx.ModelProto, name: str = "candidate.onnx") -> Path:
        path = tmp_path / name
        onnx.save(model, path)
        return path

    return save
