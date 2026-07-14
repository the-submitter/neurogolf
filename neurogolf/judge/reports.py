"""Typed gate and suite reports plus JSON Schema validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class SuiteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    required: bool
    total: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    seeds: list[int] = Field(default_factory=list)


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    judge_version: str
    status: Literal["passed", "candidate_failure", "infrastructure_error"]
    eligible_for_promotion: bool
    task_num: int = Field(ge=1, le=400)
    task_id: str = Field(pattern=r"^[0-9a-f]{8}$")
    candidate_path: str
    candidate_sha256: str | None
    candidate_size_bytes: int | None
    integrity: dict[str, Any]
    legality: dict[str, Any]
    suites: dict[str, SuiteResult]
    score: dict[str, int | float] | None
    champion_comparison: dict[str, Any] | None
    failure_packet_path: str | None
    fuzz_nonce: int | None
    errors: list[str] = Field(default_factory=list)
    started_at: str
    completed_at: str
    duration_seconds: float = Field(ge=0)

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def schema_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "schemas" / name


def validate_report(payload: dict[str, Any], schema_name: str) -> None:
    schema = json.loads(schema_path(schema_name).read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)


def write_gate_result(result: GateResult, path: Path) -> Path:
    payload = result.model_dump(mode="json")
    validate_report(payload, "gate_result.schema.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
