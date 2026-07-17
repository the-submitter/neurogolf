# NeuroGolf Codex task runner

This repository includes two independent Python 3 scripts:

- `run_codex_tasks.py` runs one Codex CLI process per NeuroGolf task, with at
  most 10 processes active by default.
- `codex_quota_supervisor.py` monitors the signed-in Codex account through a
  persistent `codex app-server` process and can redeem one reset credit when
  the Codex quota reaches zero.

Both scripts use only the Python standard library. They were checked against
`codex-cli 0.144.2`.

## Installation

Install Git, Python 3, and the Codex CLI, then sign in to Codex before setting
up this repository.

Clone the repository together with ARC-GEN and all of its nested submodules:

```bash
git clone --recurse-submodules https://github.com/the-submitter/neurogolf.git
cd neurogolf
```

Create the shared Python environment and install the scoring, ONNX, notebook,
and visualization dependencies:

```bash
python3 -m venv ~/.venv
~/.venv/bin/python -m pip install --upgrade pip
~/.venv/bin/python -m pip install -r requirements.txt
~/.venv/bin/python -c "import IPython, matplotlib, numpy, onnx, onnx_tool, onnxruntime"
```

Install the checked-in Codex profile reference as an active named profile:

```bash
mkdir -p ~/.codex
cp .codex/neurogolf-high.config.toml ~/.codex/neurogolf-high.config.toml
```

The Codex task prompt directs agents to use the shared `~/.venv` environment.
The versions are pinned to the official sample notebook where specified and to
the currently verified `~/.venv` versions for its display dependencies. The
runner and quota supervisor themselves remain standard-library-only scripts.

## Parallel task runner

Run all unfinished tasks, 10 at a time:

```bash
python3 run_codex_tasks.py
```

Preview commands without starting Codex:

```bash
python3 run_codex_tasks.py --tasks 1-20 -n 10 --dry-run
```

Run selected tasks or rerun a task whose submission already exists:

```bash
python3 run_codex_tasks.py --tasks 1,11,42-50 -n 6
python3 run_codex_tasks.py --tasks 11 --force
```

Each process is started with:

- profile `neurogolf-high`;
- model `gpt-5.6-sol` and reasoning level `high` explicitly set on the command
  line;
- the repository root as its working directory; and
- only that task's prompt, loaded from `codex_task_prompt.md` with
  `{{TASK_NUMBER}}` replaced by the three-digit task number.

The version-controlled `.codex/neurogolf-high.config.toml` is a reference copy
of this named profile. Codex does not load named profiles from a repository's
`.codex/` directory; the active copy must be installed as
`$CODEX_HOME/neurogolf-high.config.toml` (normally
`~/.codex/neurogolf-high.config.toml`). The `.gitignore` rules intentionally
track this reference file while ignoring any other project-local Codex state.

The prompt explicitly restricts each Codex process to `tasks/taskNNN/` and its
corresponding `submission/taskNNN.onnx`. The runner creates task workspaces as
needed. A successful submission causes that task to be skipped unless `--force`
is used. Per-task advisory locks prevent two runner processes from owning the
same task at once.

Repository preflight also requires `utils/neurogolf_utils.py`,
`utils/task_map.json`, `utils/task_principles.json`, and
`utils/the-2026-neurogolf-championship.py`. Task-to-generator lookup is loaded
from `utils/task_map.json`.

Outputs follow the requested layout:

```text
logs/task011.events.jsonl
logs/task011.stderr
results/task011.md
submission/task011.onnx
```

The first quota-limited retry uses `taskNNN.attempt2.*` log files so the
original evidence is preserved. Quota retries are bounded; this keeps the
runner usable without the supervisor. Change them with
`--rate-limit-retries` and `--retry-delay`. A complete machine-readable run
summary is written to `logs/codex-run-summary.json`. Ctrl-C terminates active
Codex process groups and records the partial summary.

## Quota supervisor

First verify the connection and current quota without allowing a reset:

```bash
python3 codex_quota_supervisor.py --once --dry-run
```

Then run the supervisor in one terminal and the task runner in another:

```bash
python3 codex_quota_supervisor.py --max-resets 3
python3 run_codex_tasks.py
```

The following capture shows tasks 001–010 running with `-n 10` while the
supervisor monitors the remaining Codex quota in a separate terminal:

![Ten parallel Codex task workers running while the quota supervisor monitors remaining usage](<assets/Screenshot from 2026-07-16 22-43-36.png>)

As the run continues, the monitored quota reaches 0% remaining. The supervisor
successfully consumes one reset credit, restores the quota to 100% remaining,
and continues monitoring with two reset credits left:

![Codex quota reaching zero and being successfully restored with one reset credit](<assets/Screenshot from 2026-07-16 22-58-05.png>)

The supervisor polls every 30 seconds by default. When the selected `codex`
rate-limit window reports 100% used (0% remaining), it confirms that a reset
credit is available and calls
`account/rateLimitResetCredit/consume` exactly once with a persisted UUID
idempotency key. If credit details are present, the credit expiring soonest is
used first. `--max-resets 3` limits this invocation history to the three resets
currently available; the default `0` means use any credits the server reports
as available.

Runtime state and diagnostics are stored at:

```text
logs/quota-supervisor-state.json
logs/quota-supervisor-state.json.lock
logs/quota-supervisor-app-server.stderr
```

Only one supervisor can hold the lock. Pending reset attempts are persisted
before the request is sent, so a restart reuses the same idempotency key rather
than risking a second redemption. The task runner has no dependency on the
supervisor and continues to work when it is absent; quota-limited tasks simply
use their bounded retries and are reported as failures if quota remains
unavailable.

`codex app-server` and the reset-credit method are experimental and may change
in future CLI releases. The supervisor launches its own persistent stdio app
server, so a separately managed app-server daemon is not required.

---

This code repository was developed using Codex GPT-5.6 Sol.
