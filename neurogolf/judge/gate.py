"""Integrity-protected external acceptance gate; never promotes candidates."""

from __future__ import annotations

import secrets
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from neurogolf.arcgen.generator import GeneratorOracle
from neurogolf.arcgen.sampling import boundary_biased_sample, seed_schedule
from neurogolf.config.models import NeuroGolfSettings
from neurogolf.errors import CandidateError
from neurogolf.judge.candidate_runner import CandidateRunner
from neurogolf.judge.counterexamples import Failure, build_failure_packet, write_failure_packet
from neurogolf.judge.legality import LegalityResult, inspect_legality
from neurogolf.judge.official_adapter import OfficialAdapter
from neurogolf.judge.registry import ChampionRegistry
from neurogolf.judge.reports import GateResult, SuiteResult, utc_now, write_gate_result
from neurogolf.judge.sealed import load_sealed_manifest
from neurogolf.tasks.integrity import JUDGE_VERSION, sha256_file, verify_integrity
from neurogolf.tasks.loader import load_kaggle_dataset
from neurogolf.tasks.mapping import TaskMap
from neurogolf.tasks.models import ArcExample, GeneratedExample


def _official_size(example: ArcExample) -> bool:
    return all(max(len(grid), len(grid[0])) <= 30 for grid in (example.input, example.output))


class AcceptanceGate:
    def __init__(self, settings: NeuroGolfSettings):
        self.settings = settings
        self.mapping = TaskMap(settings)

    def _suite(
        self,
        runner: CandidateRunner,
        name: str,
        examples: Iterable[tuple[ArcExample, int | None]],
        *,
        required: bool,
        deadline: float,
        failures: list[Failure],
    ) -> SuiteResult:
        passed = failed = skipped = 0
        seeds: list[int] = []
        for example, seed in examples:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Gate timeout during {name} suite")
            if not _official_size(example):
                skipped += 1
                continue
            comparison = runner.compare(example, seed=seed, family=name)
            if seed is not None:
                seeds.append(seed)
            if comparison.passed:
                passed += 1
            else:
                failed += 1
                failures.append((example, comparison))
        return SuiteResult(
            name=name,
            required=required,
            total=passed + failed,
            passed=passed,
            failed=failed,
            skipped=skipped,
            seeds=seeds,
        )

    @staticmethod
    def _generated(values: Iterable[GeneratedExample]) -> list[tuple[ArcExample, int]]:
        return [
            (ArcExample(input=value.input, output=value.output), value.seed) for value in values
        ]

    @staticmethod
    def _generate_seeds(
        oracle: GeneratorOracle,
        seeds: Iterable[int],
        *,
        family: str,
        deadline: float,
    ) -> list[GeneratedExample]:
        values: list[GeneratedExample] = []
        for seed in seeds:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Gate timeout during {family} generation")
            values.append(oracle.generate(seed))
        return values

    def run(
        self,
        *,
        task_num: int,
        candidate: Path,
        output: Path,
        failure_output: Path | None = None,
        fuzz_nonce: int | None = None,
    ) -> GateResult:
        started_monotonic = time.monotonic()
        started_at = utc_now()
        record = self.mapping.get(task_num)
        candidate_hash = sha256_file(candidate) if candidate.is_file() else None
        candidate_size = candidate.stat().st_size if candidate.is_file() else None
        integrity: dict[str, Any] = {}
        legality = LegalityResult(False, candidate_size, ["not evaluated"])
        suites: dict[str, SuiteResult] = {}
        score: dict[str, int | float] | None = None
        comparison: dict[str, Any] | None = None
        errors: list[str] = []
        failure_packet_path: str | None = None
        failures: list[Failure] = []
        nonce = fuzz_nonce if fuzz_nonce is not None else secrets.randbits(63)
        status = "infrastructure_error"
        eligible = False
        deadline = started_monotonic + self.settings.gate.timeout_seconds
        try:
            integrity = verify_integrity(self.settings)
            integrity["official_utils_sha256"] = sha256_file(self.settings.paths.official_utils)
            if not integrity["valid"]:
                raise RuntimeError(f"Protected integrity failed: {integrity['failures']}")
            adapter = OfficialAdapter(self.settings)
            legality = inspect_legality(candidate, adapter)
            if not legality.legal:
                status = "candidate_failure"
            else:
                runner = CandidateRunner(candidate, adapter)
                dataset = load_kaggle_dataset(record.kaggle_json_path)
                public_examples = [(item, None) for item in (*dataset.train, *dataset.test)]
                suites["public"] = self._suite(
                    runner,
                    "public",
                    public_examples,
                    required=self.settings.gate.require_public_pass,
                    deadline=deadline,
                    failures=failures,
                )
                suites["provided_arc_gen"] = self._suite(
                    runner,
                    "provided_arc_gen",
                    [(item, None) for item in dataset.arc_gen],
                    required=True,
                    deadline=deadline,
                    failures=failures,
                )
                oracle = GeneratorOracle(record, self.settings)
                regression_seeds = seed_schedule(
                    "regression-v1", task_num, 0, self.settings.generation.regression_seed_count
                )
                regression_values = self._generate_seeds(
                    oracle, regression_seeds, family="regression", deadline=deadline
                )
                suites["regression"] = self._suite(
                    runner,
                    "regression",
                    self._generated(regression_values),
                    required=self.settings.gate.require_regression_pass,
                    deadline=deadline,
                    failures=failures,
                )
                fuzz_seeds = seed_schedule(
                    "fresh-fuzz-v1", task_num, nonce, self.settings.generation.gate_fuzz_count
                )
                fuzz_values = self._generate_seeds(
                    oracle, fuzz_seeds, family="fuzz", deadline=deadline
                )
                suites["fuzz"] = self._suite(
                    runner,
                    "fuzz",
                    self._generated(fuzz_values),
                    required=self.settings.gate.require_fuzz_pass,
                    deadline=deadline,
                    failures=failures,
                )
                boundary_values = boundary_biased_sample(
                    oracle,
                    count=self.settings.generation.boundary_seed_count,
                    base_seed=nonce ^ 0x4E475F424F554E44,
                    deadline=deadline,
                )
                suites["boundary"] = self._suite(
                    runner,
                    "boundary",
                    self._generated(boundary_values),
                    required=True,
                    deadline=deadline,
                    failures=failures,
                )
                sealed_manifest = load_sealed_manifest(self.settings.paths.sealed_manifest)
                sealed_seeds = sealed_manifest.task_seeds(
                    record.task_id, self.settings.generation.sealed_seed_count
                )
                sealed_values = self._generate_seeds(
                    oracle, sealed_seeds, family="sealed", deadline=deadline
                )
                suites["sealed"] = self._suite(
                    runner,
                    "sealed",
                    self._generated(sealed_values),
                    required=self.settings.gate.require_sealed_pass,
                    deadline=deadline,
                    failures=failures,
                )
                all_inputs = [
                    item.input
                    for item in (*dataset.train, *dataset.test, *dataset.arc_gen)
                    if _official_size(item)
                ]
                all_inputs.extend(
                    value.input
                    for value in (
                        *regression_values,
                        *fuzz_values,
                        *boundary_values,
                        *sealed_values,
                    )
                )
                if time.monotonic() > deadline:
                    raise TimeoutError("Gate timeout before official scoring")
                score = adapter.score_model(candidate, all_inputs, deadline=deadline).to_dict()
                registry = ChampionRegistry(self.settings.paths.champion_root)
                champion = registry.current(task_num)
                comparison = {
                    "current_champion": champion.to_dict() if champion else None,
                    "candidate_objective": score["objective"],
                    "candidate_points": score["points"],
                    "score_non_regressing": champion is None
                    or int(score["objective"]) <= champion.objective,
                    "correctness_precedes_score": True,
                }
                required_pass = all(
                    suite.failed == 0 for suite in suites.values() if suite.required
                )
                status = "passed" if required_pass else "candidate_failure"
                eligible = status == "passed" and bool(comparison["score_non_regressing"])
                if failures and candidate_hash:
                    packet = build_failure_packet(
                        candidate_sha256=candidate_hash,
                        judge_version=JUDGE_VERSION,
                        task_num=task_num,
                        task_id=record.task_id,
                        failures=failures,
                        scorer_diagnostics={"legality": legality.to_dict(), "score": score},
                        max_examples=self.settings.gate.max_failure_examples,
                    )
                    packet_path = failure_output or output.with_name("failure_packet.json")
                    write_failure_packet(packet, packet_path)
                    failure_packet_path = str(packet_path)
        except CandidateError as error:
            errors.append(f"{type(error).__name__}: {error}")
            status = "candidate_failure"
            eligible = False
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            if status != "candidate_failure":
                status = "infrastructure_error"
            eligible = False
        result = GateResult(
            judge_version=JUDGE_VERSION,
            status=status,
            eligible_for_promotion=eligible,
            task_num=task_num,
            task_id=record.task_id,
            candidate_path=str(candidate),
            candidate_sha256=candidate_hash,
            candidate_size_bytes=candidate_size,
            integrity=integrity,
            legality=legality.to_dict(),
            suites=suites,
            score=score,
            champion_comparison=comparison,
            failure_packet_path=failure_packet_path,
            fuzz_nonce=nonce,
            errors=errors,
            started_at=started_at,
            completed_at=utc_now(),
            duration_seconds=time.monotonic() - started_monotonic,
        )
        write_gate_result(result, output)
        return result
