from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_codex_tasks import Runner, _create_bundled_model_catalog


def runner_args(codex_bin: str) -> argparse.Namespace:
    return argparse.Namespace(
        codex_bin=codex_bin,
        profile="neurogolf-high",
        model="gpt-5.6-sol",
        reasoning="high",
    )


def test_exports_bundled_catalog_and_passes_it_to_worker(tmp_path: Path) -> None:
    fake_codex = tmp_path / "codex"
    fake_codex.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' "
        "'{\"models\":[{\"slug\":\"gpt-5.6-sol\","
        "\"supports_reasoning_summaries\":true}]}'\n"
    )
    fake_codex.chmod(0o755)

    catalog, warning = _create_bundled_model_catalog(
        str(fake_codex), tmp_path, "gpt-5.6-sol"
    )

    assert warning is None
    assert catalog is not None
    assert json.loads(catalog.read_text())["models"][0]["slug"] == "gpt-5.6-sol"

    runner = Runner(
        runner_args(str(fake_codex)),
        tmp_path,
        "Only solve task {{TASK_NUMBER}}.",
        catalog,
    )
    command, _, _ = runner.build_command("task006", 1)
    override = f"model_catalog_json={json.dumps(str(catalog))}"
    assert override in command
    assert command[-1] == "Only solve task 006."


def test_catalog_export_falls_back_when_selected_model_is_absent(tmp_path: Path) -> None:
    fake_codex = tmp_path / "codex"
    fake_codex.write_text("#!/bin/sh\nprintf '%s\\n' '{\"models\":[]}'\n")
    fake_codex.chmod(0o755)

    catalog, warning = _create_bundled_model_catalog(
        str(fake_codex), tmp_path, "gpt-5.6-sol"
    )

    assert catalog is None
    assert warning is not None
    assert "not bundled" in warning
