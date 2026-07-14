"""Export runtime dependencies from pyproject.toml to requirements.txt."""

import tomllib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
dependencies = metadata["project"]["dependencies"]
(root / "requirements.txt").write_text("\n".join(dependencies) + "\n", encoding="utf-8")
print(f"Exported {len(dependencies)} dependencies")
