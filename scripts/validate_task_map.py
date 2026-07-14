"""Deeply validate all 400 canonical task mappings."""

import json

from neurogolf.config import load_settings
from neurogolf.tasks.mapping import TaskMap
from scripts.generate_task_principles import build_mapping, serialized

if __name__ == "__main__":
    settings = load_settings()
    result = TaskMap(settings).validate(deep=True)
    principles = build_mapping(settings.paths.repository_root)
    principles_current = (
        settings.paths.repository_root / "configs/task_principles.json"
    ).read_text(encoding="utf-8") == serialized(principles)
    payload = result.model_dump(mode="json")
    payload["task_principles_current"] = principles_current
    payload["task_principles_count"] = len(principles)
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if result.valid and principles_current else 1)
