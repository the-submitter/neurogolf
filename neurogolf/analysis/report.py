"""Write and schema-validate dossier JSON, Markdown, text, and PNG artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from neurogolf.tasks.loader import load_kaggle_dataset
from neurogolf.tasks.models import TaskRecord
from neurogolf.visualization.image import render_examples_image
from neurogolf.visualization.text import render_examples


def validate_dossier(dossier: dict[str, Any], schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(dossier, schema)


def dossier_markdown(dossier: dict[str, Any]) -> str:
    counts = dossier["source_counts"]
    shape_lines = []
    for example in dossier["examples"]:
        relationship = example["shape_relationship"]
        shape_lines.append(
            f"- {example['family']}[{example['index']}]: "
            f"{relationship['input']} → {relationship['output']}"
        )
    receptive = dossier["local_rules"]["bounded_receptive_field"]
    return "\n".join(
        [
            f"# Task {dossier['task_num']:03d} / `{dossier['task_id']}`",
            "",
            "This deterministic report describes evidence; it does not infer a task solution.",
            "",
            "## ARC-GEN principle",
            "",
            f"`{dossier['arcgen_principle']}`",
            "",
            "## Sources",
            "",
            *(f"- {name}: `{path}`" for name, path in dossier["sources"].items()),
            "",
            "## Example counts",
            "",
            *(f"- {name}: {value}" for name, value in counts.items()),
            "",
            "## Shape relationships",
            "",
            *shape_lines,
            "",
            "## Local receptive-field diagnostics",
            "",
            *(
                f"- radius {item['radius']}: plausible={item['plausible']}, "
                f"conflicts={item['conflicting_patches']}"
                for item in receptive
            ),
            "",
            "## Scorer contract",
            "",
            f"- objective: {dossier['official_scorer']['objective']}",
            f"- threshold: `{dossier['official_scorer']['threshold']}`",
            f"- opset policy: {dossier['official_scorer']['opset_policy']}",
            f"- excluded ops: {', '.join(dossier['official_scorer']['excluded_ops'])}",
            (
                "- competitive objective: approximately "
                f"{dossier['official_scorer']['competitive_budget']['approximate_objective']:.2f} "
                "after correctness"
            ),
            "",
        ]
    )


def write_analysis_artifacts(
    dossier: dict[str, Any],
    record: TaskRecord,
    output_dir: Path,
    schema_path: Path,
) -> dict[str, Path]:
    validate_dossier(dossier, schema_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_kaggle_dataset(record.kaggle_json_path)
    examples = [*dataset.train, *dataset.test, *dataset.arc_gen]
    paths = {
        "dossier": output_dir / "dossier.json",
        "report": output_dir / "report.md",
        "text": output_dir / "examples.txt",
        "image": output_dir / "examples.png",
    }
    paths["dossier"].write_text(
        json.dumps(dossier, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["report"].write_text(dossier_markdown(dossier), encoding="utf-8")
    paths["text"].write_text(render_examples(examples) + "\n", encoding="utf-8")
    render_examples_image(examples, paths["image"])
    return paths
