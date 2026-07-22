#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
FULL=0

usage() {
  cat <<'EOF'
Usage: ./scripts/setup_demo.sh [--full]

Prepare the repository-local .venv for the quota-free CodexForge replay.
Use --full to install the ONNX, scoring, notebook, and visualization packages
from requirements.txt.
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --full) FULL=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "setup_demo.sh: unknown option: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

cd -- "${REPO_ROOT}"

case "$(uname -s)" in
  Linux)
    if grep -qi microsoft /proc/version 2>/dev/null; then
      PLATFORM="WSL"
    else
      PLATFORM="Ubuntu/Linux"
    fi
    ;;
  Darwin) PLATFORM="macOS" ;;
  *) echo "Unsupported platform: $(uname -s). Use Ubuntu/Linux, WSL, or macOS." >&2; exit 1 ;;
esac

for command_name in python3 git; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 1
  fi
done

echo "CodexForge setup (${PLATFORM})"

if command -v codex >/dev/null 2>&1; then
  echo "[ok] Codex CLI found (optional for replay)"
else
  echo "[note] Codex CLI not found; replay mode remains fully available"
fi

if [[ ! -f .gitmodules ]]; then
  echo "Missing .gitmodules; cannot verify ARC-GEN" >&2
  exit 1
fi

SUBMODULE_STATUS="$(git submodule status --recursive 2>/dev/null || true)"
if [[ ! -d ARC-GEN/tasks || ! -d ARC-GEN/external/ARC-AGI || "${SUBMODULE_STATUS}" == *$'\n-'* || "${SUBMODULE_STATUS}" == -* || "${SUBMODULE_STATUS}" == *$'\n+'* || "${SUBMODULE_STATUS}" == +* ]]; then
  echo "[setup] Initializing ARC-GEN recursive submodules"
  git submodule update --init --recursive
else
  echo "[ok] ARC-GEN recursive submodules are initialized"
fi

REQUIRED_FILES=(
  run_codex_tasks.py
  codex_quota_supervisor.py
  codex_task_prompt.md
  .codex/neurogolf-high.config.toml
  demo/codexforge_dashboard.py
  demo/build_replay_fixture.py
  demo/fixtures/codexforge_demo.json
  scripts/map_tasks.py
  scripts/generate_task_principles.py
  utils/task_principles.json
)
for required_file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required CodexForge file: ${required_file}" >&2
    exit 1
  fi
done

if [[ ! -x .venv/bin/python ]]; then
  echo "[setup] Creating repository-local .venv"
  python3 -m venv .venv
else
  echo "[ok] Repository-local .venv is ready"
fi

if (( FULL )); then
  echo "[setup] Installing full ONNX/scoring dependencies"
  .venv/bin/python -m pip install -r requirements.txt
else
  echo "[ok] Replay uses the Python standard library; heavy ONNX packages skipped"
fi

.venv/bin/python demo/build_replay_fixture.py --check >/dev/null
echo "[ok] Replay fixture matches committed task evidence"

cat <<'EOF'

Ready:
  ./scripts/launch_demo.sh

For live workers, first install the full environment and then launch explicitly:
  ./scripts/setup_demo.sh --full
  ./scripts/launch_demo.sh --live --tasks 11-12 --parallel 2

Live mode requires a Codex login and this manual profile installation:
  mkdir -p ~/.codex
  cp .codex/neurogolf-high.config.toml ~/.codex/neurogolf-high.config.toml

Setup never changes your Codex configuration automatically.
EOF
