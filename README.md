# CodexForge: Built by Codex for Codex

A Codex-built orchestration system that launches parallel Codex agents to solve, validate, and optimize 400 ARC-AGI visual-reasoning tasks from Kaggle NeuroGolf as executable ONNX programs.

## How this was built with Codex

This repository was developed as a human-directed collaboration with Codex.
The human author chose the product goal and competition strategy: isolate one
task per worker, optimize against the official memory-plus-parameter metric,
use 10 concurrent workers by default, preserve task-local evidence, and keep
quota resetting optional and independent from the runner. The author also
decided when completed tasks should be skipped or force-rerun and retained
control over credentials, reset-credit use, and final submissions.

Codex turned those decisions into the orchestration and demonstration system:
it implemented subprocess concurrency, task locks, bounded retries, signal
cleanup, per-task logs and summaries, stable-CLI selection, the persistent
quota monitor, the replay/live dashboard, tests, setup automation, and
documentation. Codex accelerated the repetitive engineering work—inspecting
the repository, tracing failures across scripts and CLI versions, applying
coordinated changes, and validating them—while the human author reviewed the
behavior and made the scope, safety, and product trade-offs.

GPT-5.6 Sol is also the worker model selected by the runner. With high
reasoning effort, each worker is prompted to inspect one task's examples and
ARC-GEN generator, infer the transformation, build and test candidate ONNX
graphs, compare their official cost and score, and document the best result.
This makes the model's contribution auditable through task-local READMEs,
event logs, result messages, and submission artifacts rather than presenting
the final models as unexplained outputs.

This repository includes two independent Python 3 scripts:

- `run_codex_tasks.py` runs one Codex CLI process per NeuroGolf task, with at
  most 10 processes active by default.
- `codex_quota_supervisor.py` monitors the signed-in Codex account through a
  persistent `codex app-server` process and can redeem one reset credit when
  the Codex quota reaches zero.

Both scripts use only the Python standard library. They prefer the stable Codex
standalone installation at `~/.codex/packages/standalone/current/bin/codex`
when `codex` on `PATH` resolves to a VS Code preview build. This avoids sharing
the preview executable with batch workers; `--codex-bin` can still select a
different executable explicitly. Before a real run, the task runner exports
the selected CLI's bundled model catalog to an ephemeral ignored file and
passes it through `model_catalog_json`. Workers therefore do not parse the
shared `~/.codex/models_cache.json`, which may simultaneously be written in a
different schema by the VS Code extension. The temporary catalog is removed
when the runner exits.

## 90-second demo

```bash
./scripts/setup_demo.sh
./scripts/launch_demo.sh
```

The default **DEMO REPLAY** is an offline, quota-free 78-second timeline built
from the committed READMEs and artifacts for tasks 001, 002, and 010. It does
not require Codex, a Codex login, Kaggle, AWS credentials, the ONNX stack, or a
network connection. Replay events are always visibly identified as replay;
the final validation counts, costs, scores, graph details, and artifact states
come from repository evidence rather than invented demo data.

The scripts support Ubuntu/Linux, WSL, and macOS with Python 3 and Git. To
launch two real workers after installing and signing in to Codex, run:

```bash
./scripts/launch_demo.sh --live --tasks 11-12 --parallel 2
```

To observe runner and supervisor processes that were started independently in
other terminals, use:

```bash
./scripts/launch_demo.sh --attach
```

Live mode never starts the quota supervisor unless `--with-supervisor` is
given. That explicit supervisor still runs with `--dry-run`; reset-credit
redemption requires the additional `--allow-reset-credit` flag. Replay and
attach modes never redeem reset credits.

For a live launch, the dashboard baselines existing logs and the prior run
summary, so a forced rerun begins queued/running instead of inheriting an old
completed state. Attach mode intentionally displays the latest historical
state before new events arrive.

```text
Task selection
      ↓
Parallel Codex workers
      ↓
Task-local builders and ONNX graphs
      ↓
Validation and official scoring
      ↓
Submission artifacts and run summary
```

## Installation

Install Git and Python 3 before setting up this repository. The Codex CLI and a
Codex login are needed only for live mode, not for the default replay.

Clone the repository together with ARC-GEN and all of its nested submodules:

```bash
git clone --recurse-submodules https://github.com/the-submitter/neurogolf.git
cd neurogolf
```

Prepare the repository-local replay environment. This is idempotent and skips
the heavyweight scoring and ONNX packages:

```bash
./scripts/setup_demo.sh
```

Install the full dependency set only when rebuilding or validating ONNX models:

```bash
./scripts/setup_demo.sh --full
```

### Makefile shortcuts

The [Makefile](Makefile) is optional. It contains two convenience targets and
does not compile or install anything:

```text
make demo       # ./scripts/launch_demo.sh
make demo-live  # ./scripts/launch_demo.sh --live --tasks 11-12 --parallel 2
```

Use `make demo` for the safe, offline replay. Use `make demo-live` only when
you deliberately want two real Codex workers and have installed and signed in
to the Codex CLI. If `make` is unavailable, run the equivalent shell commands
shown above; all setup, runner, supervisor, and dashboard scripts work without
the Makefile.

Replay uses only Python's standard library. The versions in `requirements.txt`
are pinned to the official sample notebook where specified and to the verified
environment for its display dependencies. The runner and quota supervisor also
remain standard-library-only scripts.

Live mode requires Codex. Install the checked-in profile reference manually as
an active named profile; the setup script deliberately does not alter user
configuration:

```bash
mkdir -p ~/.codex
cp .codex/neurogolf-high.config.toml ~/.codex/neurogolf-high.config.toml
```

The dashboard supports `q`, `p`, `r`, `+`, `-`, `l`, and `a` for quitting,
pausing/restarting or changing replay speed, and toggling activity/artifacts.
Use `--no-color` for terminals or recordings that do not support colour. The
fixture can be regenerated from current task evidence with:

```bash
python demo/build_replay_fixture.py
```

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

Older stderr logs may contain a `codex_models_manager` message about a missing
`supports_reasoning_summaries` field. This is a non-fatal schema mismatch
caused when stable and preview Codex builds share `~/.codex/models_cache.json`;
it is not a Python dependency problem. Deleting that cache is only temporary
because either Codex process can recreate it. The runner's ephemeral bundled
catalog bypasses the shared file, so new worker runs do not require cache
deletion. Existing stderr files remain historical until that task is rerun.

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

## License and third-party material

Original code and documentation in this repository are licensed under the
[Apache License 2.0](LICENSE). This choice is compatible with the license used
by the pinned Google ARC-GEN submodule, its nested ARC-AGI submodule, and the
copyright notice in Google's NeuroGolf utility.

That repository license does **not** relicense third-party material. ARC-GEN
and ARC-AGI retain their own Apache-2.0 licenses and copyright notices. The
NeuroGolf competition utility, starter notebook, task mappings, and task data
retain their respective owners' rights and remain subject to their source
notices and the applicable Kaggle competition rules and terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for sources, pinned revisions,
license locations, and the exact scope of this repository's license.

---

This code repository was developed using Codex GPT-5.6 Sol.
