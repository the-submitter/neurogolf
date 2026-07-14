from __future__ import annotations

from neurogolf.analysis.color_analysis import analyze_colors
from neurogolf.analysis.component_analysis import component_summary
from neurogolf.analysis.dossier import build_dossier
from neurogolf.analysis.local_rule_analysis import receptive_field_plausibility
from neurogolf.analysis.report import validate_dossier
from neurogolf.analysis.spatial_analysis import analyze_grid_spatial
from neurogolf.analysis.transform_analysis import analyze_transforms
from neurogolf.tasks.mapping import TaskMap
from neurogolf.tasks.models import ArcExample


def test_color_spatial_component_transform_and_local_analysis():
    example = ArcExample(input=[[0, 1], [1, 0]], output=[[0, 1], [1, 0]])
    colors = analyze_colors(example)
    assert colors["preserved_colors"] == [0, 1]
    assert colors["changed_pixels"] == 0
    assert component_summary(example.input)["four_count"] == 2
    assert analyze_grid_spatial(example.input)["symmetry"]["main_diagonal"]
    assert "identity" in analyze_transforms(example)["d4_matches"]
    assert receptive_field_plausibility([example], max_radius=1)[0]["plausible"]


def test_real_dossier_is_schema_valid(settings):
    record = TaskMap(settings).get(1)
    dossier = build_dossier(record, settings)
    validate_dossier(
        dossier, settings.paths.repository_root / "neurogolf/schemas/dossier.schema.json"
    )
    assert dossier["task_id"] == record.task_id
    assert dossier["arcgen_principle"] == "fractal"
    assert dossier["examples"]
    assert "def generate" in dossier["generator"]["source"]
    assert dossier["official_scorer"]["threshold"] == "> 0.0"
    assert "Any standard-domain opset" in dossier["official_scorer"]["opset_policy"]
    assert dossier["official_scorer"]["competitive_budget"]["approximate_objective"] < 95
