# NeuroGolf Codex-first solver infrastructure

This repository is the deterministic platform around later, independent Codex
workers for the [2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026/overview).
It maps all 400 competition tasks to ARC-AGI and ARC-GEN, prepares evidence-rich
task workspaces, builds and runs legal ONNX graphs, applies the official scorer,
generates structured counterexamples, and protects champion promotion. It does
not contain a general ARC solver or any task-specific solution.

## Architecture and trust boundary

The data path is deliberately one-way:

1. `configs/task_map.json` resolves a Kaggle task number to its immutable
   `data/taskNNN.json`, `ARC-GEN/tasks/task_<id>.py`, and ARC-AGI JSON.
2. `neurogolf.tasks` and `neurogolf.arcgen` validate/load those sources and
   produce normalized examples with seed provenance.
3. `neurogolf.analysis` writes a dossier plus terminal and PNG evidence into a
   task workspace.
4. A task worker edits only its builder, explanation, and development tests.
5. `neurogolf.judge` independently verifies integrity, legality, public,
   regression, fresh fuzz, boundary-biased, and sealed cases, then calls the
   official profiler.
6. The gate CLI or parent orchestrator automatically promotes an eligible exact
   passing hash into the SQLite registry and immutable history; the hash-bound
   transaction remains available explicitly for maintenance.

Canonical data, the ARC-GEN submodule, `utils/neurogolf_utils.py`, mapping,
scorer/gate configuration, sealed seeds, integrity manifest, effective judge
modules and report schemas, registry, and promotion logic are protected.
Task-local `solution/`, `devtests/`, analysis output, and state reports are
editable. See [AGENTS.md](AGENTS.md) for the complete policy.

## Setup

Python 3.12 or newer is required. The official competition versions are pinned
to NumPy 2.4.4, ONNX 1.21.0, ONNX Runtime 1.24.4, and onnx-tool 1.0.1.

```bash
source ~/.venv/bin/activate
python -m pip install -e '.[dev]'
python -m neurogolf.cli doctor
```

`requirements.txt` remains as a compatible runtime export. Re-export it after a
reviewed dependency update with `python scripts/export_requirements.py`.

Configuration precedence is deterministic: model defaults, then
`configs/solver.yaml`, then `configs/scorer.yaml`, then `NEUROGOLF_*`
environment variables/optional `.env`, then explicit programmatic CLI
overrides. Nested environment keys use `__`, for example
`NEUROGOLF_GATE__TIMEOUT_SECONDS=300`.

## Mapping and canonical data

The competition files use the format documented on the
[Kaggle data page](https://www.kaggle.com/competitions/neurogolf-2026/data):
`train`, `test`, and `arc-gen` example lists. Validate every mapped path, JSON
grid, source subset, generator import, and generator `validate()` result with:

```bash
python -m neurogolf.cli tasks validate-map
python scripts/generate_task_principles.py --check
python -m neurogolf.cli tasks list
python -m neurogolf.cli tasks show 137
python -m neurogolf.cli arcgen sample 137 --count 10 --seed 123
python -m neurogolf.cli arcgen validate 137 --count 100
```

Generator imports use absolute paths and isolated module names. Calls serialize
access to ARC-GEN's process-global RNG, save/restore its state, and record the
actual retry seed and source hash.

## Prepare and analyze a task

```bash
python -m neurogolf.cli task prepare 137
```

This resumably creates `tasks/task137/` with root/task guidance, a task context,
schema-valid `analysis/dossier.json`, Markdown/text/PNG views, reproducible
generator fixtures, editable dev tests that execute those fixtures alongside
all three Kaggle families, a neutral builder, explanation, and state
directories. The builder is explicitly an infrastructure-only identity
baseline, not a task solution. Use `task prepare-all` to prepare all workspaces;
existing editable files are preserved unless `--force` is supplied.

The dossier reports the ARC-GEN `task_list()` side-comment principle, shapes and
color transitions, histograms and bounding boxes, four/eight-connected
components, D4 and fixed-translation matches, periodicity and symmetries, local
neighborhood conflicts, bounded receptive-field evidence,
generator source/signature/AST/write hints, scorer constraints, champion state,
and prior failure summaries. Workers must inspect both the text rendering and
`analysis/examples.png`; the visual evidence is a first-class input to rule
inference. Text rendering supports numeric, symbolic, ANSI, indices,
side-by-side pairs, and expected/actual/diff output.

## Build, test, and inspect ONNX

Task workers implement `tasks/taskNNN/solution/build.py`, then run the commands
recorded in `TASK_CONTEXT.md`:

```bash
~/.venv/bin/python tasks/task137/solution/build.py
~/.venv/bin/pytest -q tasks/task137/devtests
~/.venv/bin/python -m neurogolf.cli onnx inspect \
  tasks/task137/solution/candidate.onnx --cost-breakdown
~/.venv/bin/python -m neurogolf.cli scorer verify \
  --task 137 tasks/task137/solution/candidate.onnx
```

The ONNX lab defaults helper-built models to opset 12 and accepts any positive
standard-domain opset supported by the pinned ONNX checker and ONNX Runtime.
`Einsum` is available for opset-12+ graphs. The lab also provides static-shape
graph/naming helpers, initializers and Constant nodes,
standard/grouped/depthwise Conv, 1×1 color maps, shifts,
transpose/flips, Slice/Gather/Pad, masks, row/column/color reductions,
ArgMin/ArgMax, Expand, and signed-logit rendering. It also checks/runs models,
reports per-tensor estimated cost, identifies fusion opportunities, and applies
conservative dead-node, Constant, Cast, Transpose, Identity, and common-
subexpression rewrites. Helpers are optional; direct `onnx.helper` code is valid.
They are not universally opset-aware: each helper may cover only part of an
operator's schema history. Inspect helper/schema compatibility for the selected
opset and validate with the full checker and ONNX Runtime; use a version-correct
direct node when necessary.

The local estimate uses:

```text
objective = intermediate_tensor_bytes + parameter_elements
points = max(1, 25 - log(max(1, objective)))
```

It is diagnostic only. The official utility and its profiler are final truth.
The current 8178.57-point leaderboard baseline averages about 20.4464 points per
task, which corresponds to an approximate objective of 94.97. Optimize toward
or below that budget only after the complete correctness gate passes.

## External gate and counterexamples

```bash
python -m neurogolf.cli gate run \
  --task 137 \
  --candidate tasks/task137/solution/candidate.onnx \
  --output tasks/task137/state/gate_result.json
```

The gate refuses invalid protected integrity, applies the official file/model
restrictions, runs fixed public/provided/regression cases, records a fresh fuzz
nonce, selects deterministic boundary extremes, runs the independent sealed
manifest, profiles official memory/parameters, and compares the candidate with
the current champion. It writes a schema-valid gate result and, on failure, a
schema-valid packet with diverse failure signatures, seeds, grids, shape/pixel/
one-hot/runtime diagnostics, and scorer context. By default, an eligible fully
passing and score-nonregressing exact hash is then auto-promoted through the
same protected promotion transaction. Use `--no-auto-promote` for isolated task
workers; their parent orchestrator performs the promotion after validating the
structured worker result. Set `NEUROGOLF_GATE__AUTO_PROMOTE_ELIGIBLE=false` to
disable auto-promotion globally.

An intentional protected-file update requires review before hashes can change:

```bash
python -m neurogolf.cli integrity verify
python -m neurogolf.cli integrity refresh --reviewed
```

Task workers submit suspected gate/test defects under `test_change_requests/`
instead of editing acceptance rules.

## Champion lifecycle

Only the exact hash named by a fully passing, integrity-valid gate result is
eligible. Automatic and explicit promotion both recheck judge/task identity,
every required suite and its expected case count, current protected hashes, and
the candidate digest.
Correctness precedes score; a worse official objective is rejected by default.
Promotion copies the artifact to immutable history and updates the
single-current-champion SQLite state transactionally. Rollback also verifies
protected integrity and the historical artifact digest.

```bash
python -m neurogolf.cli champions list
python -m neurogolf.cli champions show 137
python -m neurogolf.cli champions promote \
  --task 137 \
  --candidate tasks/task137/solution/candidate.onnx \
  --gate-result tasks/task137/state/gate_result.json \
  --builder tasks/task137/solution/build.py \
  --explanation tasks/task137/solution/explanation.md
python -m neurogolf.cli champions rollback --task 137 --sha <64-hex-sha>
```

Rollback changes registry status; it never overwrites a historical model.

## Codex orchestration

The outer orchestrator starts one noninteractive Codex process per task, rooted
in that task's distinct workspace. It uses the configured model/reasoning level,
`workspace-write`, approval policy `never` only inside that workspace, JSONL
events, a final-response schema, a timeout, bounded parallelism, resumable
session IDs, and atomic status files. Failures are isolated per task.

```bash
python -m neurogolf.cli solve task 137
python -m neurogolf.cli solve many 1 2 3 4
python -m neurogolf.cli solve pending
python -m neurogolf.cli solve failed
python -m neurogolf.cli solve golfed
python scripts/solve_all_tasks.py --max-parallel 4
```

`solve task` and `solve many` accept `--kind`, `--epoch`, and `--fresh`.
Later epochs resume the recorded session by default; `--fresh` deliberately
starts a separate session. Failed-task scheduling increments the saved epoch
and uses the repair prompt, while golfing starts a fresh epoch. Structured
worker output is schema-validated before it is persisted into status.

`scripts/solve_all_tasks.py` enumerates all 400 mapped tasks and runs the same
resumable prepare-then-solve workflow for each. Codex process concurrency comes
from `codex.max_parallel_tasks` (default 4) and can be overridden with
`--max-parallel`. Use `--dry-run` to validate the batch plan without preparing
workspaces or launching task workers. The configured `neurogolf-xhigh` Codex
profile supplies model, reasoning, sandbox, and approval defaults.

The profile is loaded from `~/.codex/neurogolf-xhigh.config.toml`:

```toml
model = "gpt-5.6-sol"
model_reasoning_effort = "xhigh"
model_verbosity = "low"
approval_policy = "never"
sandbox_mode = "workspace-write"
model_reasoning_summary = "concise"

[sandbox_workspace_write]
network_access = true
writable_roots = [
  "/home/rohit-raje/.cache",
  "/home/rohit-raje/.codex",
  "/home/rohit-raje/.m2",
]
```

The verbosity and reasoning-summary keys are optional presentation controls;
they do not change the selected reasoning effort. Edit the profile or disable it
in programmatic invocations to select another supported model/reasoning level.

For interactive Codex IDE work, invoke the repository skill with
`$solve-neurogolf-task` and one task number or ID. The repo-scoped skill lives at
`.agents/skills/solve-neurogolf-task/` and respects the model and reasoning level
selected in the IDE.

The solve, repair, golf, review, and test-change templates live under `prompts/`.
Bootstrap and task preparation do not invoke them.

## Repository verification

```bash
~/.venv/bin/python -m neurogolf.cli doctor
~/.venv/bin/python -m neurogolf.cli tasks validate-map
~/.venv/bin/python scripts/generate_task_principles.py --check
~/.venv/bin/python -m neurogolf.cli task prepare 1
~/.venv/bin/python -m neurogolf.cli arcgen sample 1 --count 3 --seed 123
~/.venv/bin/pytest -q
~/.venv/bin/ruff check neurogolf scripts tests
~/.venv/bin/pyright neurogolf scripts tests
```

Parity tests load the official utility rather than mocking its behavior.

## Troubleshooting and conventions

- Run commands from the repository root or let configuration root discovery
  find the nearest directory containing `configs/task_map.json` and `ARC-GEN/`.
- An integrity failure is an infrastructure failure, not a candidate failure.
  Review the diff; do not reflexively refresh hashes.
- A gate exit code of 2 means candidate rejection. Exit code 1 means
  infrastructure/configuration/integrity failure.
- Grids must be non-empty, rectangular, and use integer colors 0–9. Official
  verification skips examples larger than 30×30.
- ONNX output uses strict positive logits. Zero is negative; malformed one-hot
  cells appear as special colors 10 (none) or 11 (multiple).
- The gate timeout is enforced between generated examples and profiler runs; a
  single stuck native ONNX call or generator call cannot be preempted safely in
  process.
- Sealed seeds are independent and integrity-protected, but deliberately stable
  rather than secret. The local static cost estimate is diagnostic; only the
  official profiler decides final memory and points.
- Keep shared helpers semantic-neutral and tested. Keep task logic in its task
  workspace, and never share a mutable task directory between workers.
