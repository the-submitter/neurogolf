"""Compose the deterministic task dossier consumed by later semantic workers."""

from __future__ import annotations

from typing import Any

from neurogolf.analysis.color_analysis import analyze_colors
from neurogolf.analysis.component_analysis import component_summary
from neurogolf.analysis.generator_analysis import analyze_generator
from neurogolf.analysis.local_rule_analysis import local_change_table, receptive_field_plausibility
from neurogolf.analysis.spatial_analysis import analyze_grid_spatial
from neurogolf.analysis.transform_analysis import analyze_transforms
from neurogolf.arcgen.generator import GeneratorOracle
from neurogolf.config.models import NeuroGolfSettings
from neurogolf.judge.official_adapter import OfficialAdapter
from neurogolf.tasks.loader import load_arc_agi_dataset, load_kaggle_dataset
from neurogolf.tasks.models import ArcExample, TaskRecord
from neurogolf.tasks.principles import task_principle

LEADERBOARD_TOTAL_POINTS = 8178.57
LEADERBOARD_TASK_COUNT = 400
LEADERBOARD_AVERAGE_POINTS = LEADERBOARD_TOTAL_POINTS / LEADERBOARD_TASK_COUNT
LEADERBOARD_APPROXIMATE_OBJECTIVE = 94.9713246265938


def build_dossier(
    record: TaskRecord,
    settings: NeuroGolfSettings,
    *,
    champion: dict[str, Any] | None = None,
    failure_summaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    kaggle = load_kaggle_dataset(record.kaggle_json_path)
    arc_agi = load_arc_agi_dataset(record.arc_agi_json_path)
    examples: list[tuple[str, int, ArcExample]] = []
    for family, values in (
        ("train", kaggle.train),
        ("test", kaggle.test),
        ("arc-gen", kaggle.arc_gen),
    ):
        examples.extend((family, index, example) for index, example in enumerate(values))
    oracle = GeneratorOracle(record, settings)
    per_example: list[dict[str, Any]] = []
    for family, index, example in examples:
        per_example.append(
            {
                "family": family,
                "index": index,
                "input": example.input,
                "output": example.output,
                "shape_relationship": {
                    "input": [len(example.input), len(example.input[0])],
                    "output": [len(example.output), len(example.output[0])],
                    "same": [len(example.input), len(example.input[0])]
                    == [len(example.output), len(example.output[0])],
                },
                "colors": analyze_colors(example),
                "spatial": {
                    "input": analyze_grid_spatial(example.input),
                    "output": analyze_grid_spatial(example.output),
                },
                "components": {
                    "input": component_summary(example.input),
                    "output": component_summary(example.output),
                },
                "transforms": analyze_transforms(example),
            }
        )
    all_examples = [example for _, _, example in examples]
    adapter = OfficialAdapter(settings, verify_hash=False)
    scorer_contract = adapter.describe_contract()
    scorer_contract.update(
        {
            "opset_policy": (
                "Any standard-domain opset accepted by the pinned ONNX checker and ONNX Runtime; "
                "operator-specific minimum versions still apply."
            ),
            "competitive_budget": {
                "leaderboard_total_points": LEADERBOARD_TOTAL_POINTS,
                "task_count": LEADERBOARD_TASK_COUNT,
                "average_points_per_task": LEADERBOARD_AVERAGE_POINTS,
                "approximate_objective": LEADERBOARD_APPROXIMATE_OBJECTIVE,
                "priority": "Correctness first; optimize toward or below this objective afterward.",
            },
        }
    )
    return {
        "schema_version": "1.0",
        "task_num": record.task_num,
        "task_id": record.task_id,
        "arcgen_principle": task_principle(
            settings.paths.repository_root, record.task_num, record.task_id
        ),
        "sources": {
            "kaggle_json": str(record.kaggle_json_path),
            "arcgen_generator": str(record.arcgen_generator_path),
            "arc_agi_json": str(record.arc_agi_json_path),
            "arc_agi_split": record.arc_agi_split,
        },
        "source_counts": {
            "kaggle_train": len(kaggle.train),
            "kaggle_test": len(kaggle.test),
            "kaggle_arc_gen": len(kaggle.arc_gen),
            "arc_agi_train": len(arc_agi.train),
            "arc_agi_test": len(arc_agi.test),
        },
        "examples": per_example,
        "local_rules": {
            "change_table": local_change_table(all_examples),
            "bounded_receptive_field": receptive_field_plausibility(all_examples),
        },
        "generator": analyze_generator(oracle.imported),
        "official_scorer": scorer_contract,
        "current_champion": champion,
        "existing_failure_summaries": failure_summaries or [],
    }
