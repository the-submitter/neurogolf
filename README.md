# CodexForge: Built by Codex for Codex

A Codex-built deterministic orchestration system that runs long-lived parallel
agents to solve, validate, and optimize 400 Kaggle NeuroGolf ARC-AGI tasks into
verified ONNX programs.

CodexForge is a deterministic control plane rather than a monolithic solving
agent. It assigns one Kaggle NeuroGolf ARC task to each isolated Codex worker,
manages the campaign lifecycle in Python, and preserves the evidence needed to
verify each resulting ONNX program.

## How this was built with Codex

This repository was developed as a human-directed collaboration with Codex.
The human author chose the product goal and competition strategy: isolate one
task per worker, optimize against the official memory-plus-parameter metric,
use 10 concurrent workers by default, preserve task-local evidence, and keep
quota resetting optional and independent from the runner. The author also
decided when completed tasks should be skipped or force-rerun and retained
control over credentials, reset-credit use, and final submissions.

Codex turned those decisions into a deterministic orchestration and
demonstration system: it implemented subprocess concurrency, task locks,
bounded retries and timeouts, signal cleanup, per-task logs and summaries,
artifact checks, stable-CLI selection, the persistent quota monitor, the
replay/live dashboard, tests, setup automation, and documentation. Codex
accelerated repository inspection, cross-script debugging, coordinated
implementation, and verification. The human author reviewed the behavior and
made the scope, safety, competition-strategy, and product trade-offs.

GPT-5.6 Sol is also the worker model selected by the runner. With high
reasoning effort, each worker is prompted to inspect one task's examples and
ARC-GEN generator, infer the transformation, build and test candidate ONNX
computational graphs, compare their official cost and score, debug failures,
optimize the implementation, and document the best result. This makes the
model's contribution auditable through task-local READMEs, event logs, result
messages, and submission artifacts rather than presenting the final ONNX
programs as unexplained outputs.

## Why a deterministic control plane as the orchestration system?

Codex workers may run for hours, while campaigns across 400 Kaggle NeuroGolf
ARC tasks may last days or weeks. A parent reasoning agent should not remain
active merely to poll workers.

- Deterministic Python handles concurrency, locking, process supervision,
  retries, timeouts, logs, artifact checks, summaries, and resumability.
- Codex workers handle ARC reasoning, generator inspection, ONNX synthesis,
  testing, debugging, and optimization.

**Use Codex for reasoning; use deterministic software for orchestration.**

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
the final validation counts, costs, scores, ONNX computational-graph details,
and artifact states come from repository evidence rather than invented demo
data.

The scripts support Ubuntu/Linux, WSL, and macOS with Python 3 and Git. Real
worker runs have additional Codex CLI, login, named-profile, and full Python
environment prerequisites. Complete the ordered **Live-mode prerequisites**
under Installation before using `--live`.

The launcher defaults to the offline replay even when `--tasks` is supplied.
Always pass `--live` to start real workers; for example, task001 and task006
require `./scripts/launch_demo.sh --live --tasks 1,6 --parallel 2`. Add
`--force` when their existing submissions should be optimized again. A replay
selection not present in the committed fixture is rejected instead of being
silently omitted.

To observe runner and supervisor processes that were started independently in
other terminals, use:

```bash
./scripts/launch_demo.sh --attach
```

Attach mode intentionally displays the latest historical
state before new events arrive.

```text
Campaign and task selection
            ↓
Deterministic Python control plane
(concurrency, locks, supervision, retries, timeouts, logs)
            ↓
Long-lived parallel Codex workers
(ARC reasoning, generator inspection, ONNX synthesis and debugging)
            ↓
Task-local validation and official scoring
            ↓
Verified ONNX programs, artifact checks, summaries, and resumable state
```

## Installation

Install Git and Python 3 before setting up this repository. The default replay
needs neither Codex nor a Codex login.

Clone the repository together with ARC-GEN and all of its nested submodules:

```bash
git clone --recurse-submodules https://github.com/the-submitter/neurogolf.git
cd neurogolf
```

### Live-mode prerequisites

> **Required before the first full live setup:** install the Codex CLI, sign in
> to Codex, and activate the repository's `neurogolf-high` named profile. The
> checked-in file is only a reference copy; Codex does not load named profiles
> directly from the project's `.codex/` directory.

After cloning the repository, verify the CLI/login and copy the profile into
`$CODEX_HOME` (normally `~/.codex`):

```bash
codex -V
codex login

mkdir -p ~/.codex
cp .codex/neurogolf-high.config.toml ~/.codex/neurogolf-high.config.toml
```

If `CODEX_HOME` is set to another directory, copy the profile there instead:

```bash
mkdir -p "$CODEX_HOME"
cp .codex/neurogolf-high.config.toml "$CODEX_HOME/neurogolf-high.config.toml"
```

Only after those live-mode prerequisites are ready, install the full Python
environment and launch real workers:

```bash
./scripts/setup_demo.sh --full
./scripts/launch_demo.sh --live --tasks 11-12 --parallel 2
```

Live mode never starts the quota supervisor unless `--with-supervisor` is
given. That explicit supervisor still runs with `--dry-run`; reset-credit
redemption requires the additional `--allow-reset-credit` flag. Replay and
attach modes never redeem reset credits.

For a live launch, the dashboard baselines existing logs and the prior run
summary, so a forced rerun begins queued/running instead of inheriting an old
completed state.

The profile and Codex login are not required for the offline replay setup
below.

Prepare the repository-local replay environment. This is idempotent and skips
the heavyweight scoring and ONNX packages:

```bash
./scripts/setup_demo.sh
```

Install the full dependency set before live worker runs and whenever rebuilding
or validating ONNX programs. For live mode, complete the profile installation
above first:

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

The dashboard supports `q`, `p`, `r`, `+`, `-`, `l`, and `a` for quitting,
pausing/restarting or changing replay speed, and toggling activity/artifacts.
Use `--no-color` for terminals or recordings that do not support colour.

### Turn completed live tasks into an offline replay

Live runs do not automatically rewrite replay fixtures. They update task
READMEs, ONNX files, results, and logs, but an explicit snapshot is required
after the selected workers finish successfully. This prevents an in-progress
or failed run from silently changing the committed demo.

If you only need to inspect the latest saved logs and final states,
`./scripts/launch_demo.sh --attach --tasks 1,6` can do that without building a
fixture. Use the fixture procedure below when you want the deterministic timed
**DEMO REPLAY** presentation.

For example, snapshot the completed task001 and task006 evidence into a
separate fixture and replay it without Codex or quota usage:

```bash
./.venv/bin/python demo/build_replay_fixture.py \
  --tasks 1,6 \
  --output demo/fixtures/task001-task006.json

./scripts/launch_demo.sh --replay \
  --fixture demo/fixtures/task001-task006.json
```

The builder automates extraction of task IDs, principles, documented metrics,
and existing README/ONNX/result/log artifact paths. It then creates a
deterministic authored timeline; it does not replay the original live process
timing or make network calls. Keep the custom fixture if it should be shared or
recorded, or delete it after a one-off local demo.

Validate that a saved custom fixture still matches current evidence with:

```bash
./.venv/bin/python demo/build_replay_fixture.py \
  --tasks 1,6 \
  --output demo/fixtures/task001-task006.json \
  --check
```

Running the builder without `--tasks` or `--output` regenerates the standard
committed task001/task002/task010 fixture used by `make demo` and by the default
launcher. `setup_demo.sh` checks that standard fixture for staleness but does
not rewrite it automatically:

```bash
./.venv/bin/python demo/build_replay_fixture.py
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

The two mappings are generated maintenance artifacts. Regenerate or validate
them from the repository root with:

```bash
./.venv/bin/python scripts/map_tasks.py --check
./.venv/bin/python scripts/generate_task_principles.py --check

# Omit --check to rewrite the corresponding JSON files in utils/.
```

`map_tasks.py` matches complete public NeuroGolf train/test pairs against
ARC-AGI without relying on example order, restricts candidates to tasks with
an ARC-GEN generator, and deliberately excludes generated `arc-gen` cases.
`generate_task_principles.py` joins that map with the principle comments in
`ARC-GEN/task_list.py`. Both scripts use only the Python standard library and
default to this repository's current `kaggle_tasks_data/`, `ARC-GEN/`, and
`utils/` layout.

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

## Future applications

The same control-plane pattern can support other bounded, independently
verifiable engineering campaigns, such as compiler optimization sweeps,
repository migration batches, benchmark pipelines, and artifact-optimization
queues. The transferable idea is the separation of deterministic lifecycle
management from task-specific Codex reasoning, backed by explicit validators
and durable artifacts. CodexForge does not claim general AGI or arbitrary
unseen ARC solving; its present scope is the 400 Kaggle NeuroGolf ARC tasks and
their verified ONNX programs.

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
