"""Resumably prepare every mapped task workspace without starting Codex."""

from neurogolf.config import load_settings
from neurogolf.orchestrator.prepare import prepare_all

if __name__ == "__main__":
    values = prepare_all(load_settings())
    print(f"Prepared {len(values)} task workspaces")
