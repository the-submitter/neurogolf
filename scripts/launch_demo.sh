#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd -- "${REPO_ROOT}"

if [[ ! -x .venv/bin/python || ! -f .venv/bin/activate ]]; then
  echo "CodexForge demo environment is missing. Run ./scripts/setup_demo.sh first." >&2
  exit 1
fi
if [[ ! -f demo/codexforge_dashboard.py ]]; then
  echo "Missing demo/codexforge_dashboard.py" >&2
  exit 1
fi

MODE="replay"
MODE_EXPLICIT=0
SHOW_HELP=0
for argument in "$@"; do
  case "${argument}" in
    --replay) MODE="replay"; MODE_EXPLICIT=1 ;;
    --live) MODE="live"; MODE_EXPLICIT=1 ;;
    --attach) MODE="attached"; MODE_EXPLICIT=1 ;;
    -h|--help) SHOW_HELP=1 ;;
  esac
done

if [[ "${MODE}" == "replay" && ! -f demo/fixtures/codexforge_demo.json ]]; then
  echo "Missing replay fixture. Run python3 demo/build_replay_fixture.py." >&2
  exit 1
fi

if [[ "${MODE}" == "live" ]]; then
  if ! command -v codex >/dev/null 2>&1; then
    echo "Live mode requires the Codex CLI. Replay mode does not." >&2
    exit 1
  fi
  if [[ ! -f "${CODEX_HOME:-${HOME}/.codex}/neurogolf-high.config.toml" ]]; then
    echo "Live mode requires the neurogolf-high profile." >&2
    echo "Install it manually: cp .codex/neurogolf-high.config.toml \"${CODEX_HOME:-${HOME}/.codex}/neurogolf-high.config.toml\"" >&2
    exit 1
  fi
fi

# Activate the repository-local environment without changing shell startup files.
source .venv/bin/activate

ARGS=("$@")
if (( ! MODE_EXPLICIT && ! SHOW_HELP )); then
  ARGS=(--replay "${ARGS[@]}")
fi

case "${MODE}" in
  replay) echo "CodexForge preflight: DEMO REPLAY (offline, quota-free)" ;;
  live) echo "CodexForge preflight: LIVE (Codex quota may be used)" ;;
  attached) echo "CodexForge preflight: ATTACHED (observation only)" ;;
esac

exec python demo/codexforge_dashboard.py "${ARGS[@]}"
