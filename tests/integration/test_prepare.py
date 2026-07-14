from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import onnx
import onnxruntime as ort
import pytest

from neurogolf.config import load_settings
from neurogolf.orchestrator.prepare import prepare_task


@pytest.mark.integration
def test_prepare_one_real_task_is_complete_neutral_and_resumable(
    repository_root: Path, tmp_path: Path
) -> None:
    settings = load_settings(
        repository_root=repository_root,
        overrides={
            "paths": {
                "task_workspace_root": tmp_path / "tasks",
                "champion_root": tmp_path / "champions",
            },
            "generation": {"development_seed_count": 2},
        },
    )
    result = prepare_task(1, settings)
    workspace = Path(result["workspace"])
    expected = {
        "AGENTS.md",
        "TASK_CONTEXT.md",
        "analysis/dossier.json",
        "analysis/report.md",
        "analysis/examples.txt",
        "analysis/examples.png",
        "devtests/generated_examples.json",
        "devtests/test_solution.py",
        "solution/build.py",
        "solution/explanation.md",
        "state/current_champion.json",
    }
    assert all((workspace / relative).is_file() for relative in expected)
    dossier = json.loads((workspace / "analysis/dossier.json").read_text())
    generated = json.loads((workspace / "devtests/generated_examples.json").read_text())
    assert dossier["task_num"] == 1
    assert dossier["arcgen_principle"] == "fractal"
    assert len(generated) == 2
    devtest = (workspace / "devtests/test_solution.py").read_text()
    assert "dataset.train" in devtest and "dataset.test" in devtest
    assert "dataset.arc_gen" in devtest and "generated_examples.json" in devtest
    assert "onnx.checker.check_model" in devtest
    assert "CandidateRunner" in devtest and "convert_example" in devtest
    task_agents = (workspace / "AGENTS.md").read_text()
    assert "analysis/examples.png" in task_agents
    assert "not universally opset-aware" in task_agents
    assert "direct `onnx.helper`" in task_agents
    context = json.loads(
        (workspace / "TASK_CONTEXT.md").read_text().split("```json\n", 1)[1].split("\n```", 1)[0]
    )
    assert context["arcgen_principle"] == "fractal"
    assert context["commands"]["gate"].endswith("--no-auto-promote")

    builder_path = workspace / "solution/build.py"
    namespace: dict[str, object] = {"__name__": "prepared_builder"}
    exec(compile(builder_path.read_text(), str(builder_path), "exec"), namespace)
    build = cast(Callable[[], onnx.ModelProto], namespace["build"])
    model = build()
    assert isinstance(model, onnx.ModelProto)
    onnx.checker.check_model(model, full_check=True)
    ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    explanation = (workspace / "solution/explanation.md").read_text()
    assert "No semantic solution has been attempted" in explanation

    builder_path.write_text("# user-owned task edit\n", encoding="utf-8")
    stale_dossier = json.loads((workspace / "analysis/dossier.json").read_text())
    stale_dossier.pop("arcgen_principle")
    (workspace / "analysis/dossier.json").write_text(json.dumps(stale_dossier), encoding="utf-8")
    prepare_task(1, settings)
    assert builder_path.read_text() == "# user-owned task edit\n"
    refreshed_dossier = json.loads((workspace / "analysis/dossier.json").read_text())
    assert refreshed_dossier["arcgen_principle"] == "fractal"
