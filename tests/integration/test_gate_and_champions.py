from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest

from neurogolf.config import load_settings
from neurogolf.config.models import NeuroGolfSettings
from neurogolf.errors import PromotionError
from neurogolf.judge.gate import AcceptanceGate
from neurogolf.judge.official_adapter import OfficialAdapter
from neurogolf.judge.promotion import promote_candidate, rollback_champion
from neurogolf.judge.registry import ChampionRegistry
from neurogolf.judge.reports import validate_report
from neurogolf.onnx_lab.builder import GraphBuilder
from neurogolf.onnx_lab.conv import color_map_1x1
from neurogolf.onnx_lab.templates.basic import identity_model
from neurogolf.tasks.integrity import sha256_file


class _UnverifiedOfficialAdapter(OfficialAdapter):
    def __init__(self, settings: NeuroGolfSettings) -> None:
        super().__init__(settings, verify_hash=False)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _synthetic_settings(repository_root: Path, tmp_path: Path) -> NeuroGolfSettings:
    task_id = "007bbfb7"
    arcgen = tmp_path / "arcgen"
    (arcgen / "tasks").mkdir(parents=True)
    (arcgen / "common.py").write_text("import random\n", encoding="utf-8")
    (arcgen / "tasks" / f"task_{task_id}.py").write_text(
        """from common import random


def generate():
    random.random()
    grid = [[0, 1], [1, 0]]
    return {"input": grid, "output": grid}


def validate():
    pair = generate()
    return {"train": [pair], "test": [pair]}
""",
        encoding="utf-8",
    )
    source = {
        "train": [{"input": [[0, 1], [1, 0]], "output": [[0, 1], [1, 0]]}],
        "test": [{"input": [[2]], "output": [[2]]}],
    }
    _write_json(
        arcgen / "external" / "ARC-AGI" / "data" / "training" / f"{task_id}.json",
        source,
    )
    data = {
        **source,
        "arc-gen": [{"input": [[3, 0]], "output": [[3, 0]]}],
    }
    data_root = tmp_path / "data"
    _write_json(data_root / "task001.json", data)
    task_map = tmp_path / "task_map.json"
    _write_json(task_map, {"task001": task_id})
    sealed = tmp_path / "sealed.json"
    _write_json(
        sealed,
        {
            "version": "fixture-v1",
            "algorithm": "affine-mod-v1",
            "start": 17,
            "stride": 31,
            "modulus": 2_147_483_647,
            "count": 1,
            "namespace": "fixture-sealed",
        },
    )
    return load_settings(
        repository_root=repository_root,
        overrides={
            "paths": {
                "arcgen_root": arcgen,
                "kaggle_data_root": data_root,
                "task_map": task_map,
                "task_workspace_root": tmp_path / "tasks",
                "champion_root": tmp_path / "champions",
                "sealed_manifest": sealed,
                "integrity_manifest": tmp_path / "integrity.json",
            },
            "generation": {
                "development_seed_count": 1,
                "regression_seed_count": 1,
                "gate_fuzz_count": 1,
                "boundary_seed_count": 1,
                "sealed_seed_count": 1,
                "max_attempts_per_seed": 2,
            },
            "gate": {"timeout_seconds": 60, "max_failure_examples": 3},
        },
    )


def _patch_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "neurogolf.judge.gate.verify_integrity",
        lambda _settings: {
            "valid": True,
            "judge_version": "fixture",
            "checked_files": 1,
            "failures": [],
        },
    )
    monkeypatch.setattr("neurogolf.judge.gate.OfficialAdapter", _UnverifiedOfficialAdapter)


def _save_identity(path: Path, *, label: str = "identity") -> Path:
    model = identity_model()
    model.graph.doc_string = label
    onnx.save(model, path)
    return path


@pytest.mark.integration
def test_gate_passes_identity_and_emits_schema_valid_failure_packet(
    repository_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _synthetic_settings(repository_root, tmp_path)
    _patch_gate(monkeypatch)
    candidate = _save_identity(tmp_path / "identity.onnx")
    passed_path = tmp_path / "passed.json"
    passed = AcceptanceGate(settings).run(
        task_num=1,
        candidate=candidate,
        output=passed_path,
        fuzz_nonce=123,
    )
    assert passed.status == "passed"
    assert passed.eligible_for_promotion
    assert set(passed.suites) == {
        "public",
        "provided_arc_gen",
        "regression",
        "fuzz",
        "boundary",
        "sealed",
    }
    assert all(suite.failed == 0 for suite in passed.suites.values())
    validate_report(json.loads(passed_path.read_text()), "gate_result.schema.json")
    assert not (settings.paths.arcgen_root / "__pycache__").exists()
    assert not (settings.paths.arcgen_root / "tasks/__pycache__").exists()

    builder = GraphBuilder.standard("wrong_color")
    mapped = color_map_1x1(builder, "input", [1, 0, 2, 3, 4, 5, 6, 7, 8, 9])
    builder.direct_output(mapped, dtype=onnx.TensorProto.FLOAT, shape=(1, 10, 30, 30))
    wrong_path = tmp_path / "wrong.onnx"
    onnx.save(builder.build(), wrong_path)
    failed_path = tmp_path / "failed.json"
    packet_path = tmp_path / "failure-packet.json"
    failed = AcceptanceGate(settings).run(
        task_num=1,
        candidate=wrong_path,
        output=failed_path,
        failure_output=packet_path,
        fuzz_nonce=123,
    )
    assert failed.status == "candidate_failure"
    assert not failed.eligible_for_promotion
    assert packet_path.is_file()
    packet = json.loads(packet_path.read_text())
    validate_report(packet, "failure_packet.schema.json")
    assert packet["failure_count"] >= 1
    assert packet["representative_failures"][0]["input"]
    assert packet["representative_failures"][0]["pixel_difference_count"] >= 1


@pytest.mark.integration
def test_promotion_rejection_history_and_rollback(
    repository_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _synthetic_settings(repository_root, tmp_path)
    _patch_gate(monkeypatch)
    monkeypatch.setattr(
        "neurogolf.judge.promotion.verify_integrity",
        lambda _settings: {"valid": True, "failures": []},
    )
    monkeypatch.setattr("neurogolf.judge.promotion.git_revision", lambda _path: "fixture-rev")

    first = _save_identity(tmp_path / "first.onnx", label="first")
    first_gate_path = tmp_path / "first-gate.json"
    first_gate = AcceptanceGate(settings).run(
        task_num=1, candidate=first, output=first_gate_path, fuzz_nonce=9
    )
    assert first_gate.passed
    first_record = promote_candidate(
        task_num=1,
        candidate=first,
        gate_result_path=first_gate_path,
        settings=settings,
    )
    assert first_record.candidate_sha256 == sha256_file(first)

    second = _save_identity(tmp_path / "second.onnx", label="second")
    with pytest.raises(PromotionError, match="hash does not match"):
        promote_candidate(
            task_num=1,
            candidate=second,
            gate_result_path=first_gate_path,
            settings=settings,
        )

    second_gate_path = tmp_path / "second-gate.json"
    second_gate = AcceptanceGate(settings).run(
        task_num=1, candidate=second, output=second_gate_path, fuzz_nonce=10
    )
    assert second_gate.passed
    second_record = promote_candidate(
        task_num=1,
        candidate=second,
        gate_result_path=second_gate_path,
        settings=settings,
    )
    registry = ChampionRegistry(settings.paths.champion_root)
    assert registry.current(1) == second_record
    assert Path(first_record.onnx_path).is_file()
    assert Path(second_record.onnx_path).is_file()

    rolled_back = rollback_champion(
        task_num=1,
        sha256=first_record.candidate_sha256,
        settings=settings,
        notes="integration rollback",
    )
    assert rolled_back.candidate_sha256 == first_record.candidate_sha256
    assert [item["action"] for item in registry.history(1)] == [
        "promote",
        "promote",
        "rollback",
    ]
    Path(second_record.onnx_path).write_bytes(b"tampered")
    with pytest.raises(PromotionError, match="hash mismatch"):
        rollback_champion(
            task_num=1,
            sha256=second_record.candidate_sha256,
            settings=settings,
        )
