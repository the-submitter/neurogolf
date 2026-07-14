---
name: solve-neurogolf-task
description: Solve exactly one NeuroGolf/ARC task with this repository's prepared workspace, visual and textual examples, ARC-GEN generator evidence, ONNX construction helpers, development tests, protected acceptance gate, and eligible-candidate auto-promotion. Use when a Codex IDE user asks to solve, repair, review, or golf one task by Kaggle task number, taskNNN key, or eight-character ARC ID. Do not use for multi-task batches, generic ARC solver development, or protected gate changes.
---

# Solve one NeuroGolf task

Work on exactly one task and retain the model and reasoning effort selected by
the user in the Codex IDE. Do not edit project/user model configuration. The
scripted CLI defaults to the `neurogolf-xhigh` profile, but this skill does not
require that profile.

## Establish the workspace

1. Read the root `AGENTS.md` and the closer task `AGENTS.md` completely.
2. Resolve the requested task with
   `~/.venv/bin/python -m neurogolf.cli tasks show <task>`.
3. Run `~/.venv/bin/python -m neurogolf.cli task prepare <task-num>` if its
   workspace is absent or incomplete. Do not use `--force` on user-edited work.
4. Work only in `tasks/taskNNN/`. Never start or modify another task.

## Analyze before building

Read `TASK_CONTEXT.md`, `analysis/dossier.json`, `analysis/report.md`, and
`analysis/examples.txt`. Open `analysis/examples.png` with the available image
viewer and inspect every input/output pair visually. Treat the mapped ARC-GEN
principle as a useful hypothesis, then verify it against all canonical examples,
the generator source/signature, current champion, and latest failure packet.

State one semantic rule before implementing it. Do not create a generic ARC
searcher or encode observed examples as a lookup table.

## Build the graph

Edit only task-local allowed artifacts: `solution/build.py`,
`solution/explanation.md`, `devtests/`, and noncanonical analysis/state notes.
Build one legal static ONNX graph with the required `[1,10,30,30]` float32
`input` and `output` tensors and strict-positive logits.

Use any standard-domain opset accepted by the pinned ONNX checker and ONNX
Runtime. Opset 10 is not a ceiling; `Einsum` is available from opset 12. The
shared builder defaults to opset 12, and direct `onnx.helper` code remains valid.

The ONNX lab helpers are not universally opset-aware. `GraphBuilder` accepts any
positive opset, but an individual helper may emit only some versions of an
operator schema. Inspect the helper and the selected opset's schema before using
it, then run `onnx.checker.check_model(..., full_check=True)` and ONNX Runtime.
If incompatible, construct the correct version-specific node directly with
`onnx.helper`; do not change the task's opset merely to fit the helper.

The current Kaggle leaderboard score is 8178.57, about 20.4465 points per task.
Under the official scoring curve, that implies a current approximate average
objective/cost budget of 94.97 (intermediate tensor bytes plus parameter
elements) per ONNX task graph. Prioritize semantic correctness; after all
required suites pass, deliberately optimize toward 94.97 or lower. Prefer direct
signed-logit output, fewer full-grid intermediates, bool intermediates,
attributes, and small parameters.

## Verify and repair

Run the commands from `TASK_CONTEXT.md` in order:

1. Build `solution/candidate.onnx`.
2. Run task dev tests.
3. Inspect the model and cost breakdown.
4. Run ONNX checker/runtime and conversion parity checks.
5. From the repository root, run the full gate without
   `--no-auto-promote` so an eligible exact hash is promoted automatically.

If the gate fails, preserve any prior passing candidate, read the structured
failure packet, repair the common semantic cause, and rerun the complete gate.
Never weaken or edit protected acceptance files. Put suspected test or judge
defects under `test_change_requests/` using the repository review workflow.

## Finish

Update the explanation and focused development tests. Write a schema-valid
`state/codex_result.json` with commands, tests, candidate hash, and gate path.
Confirm that the gate reported full correctness and record whether auto-promotion
reported `promoted`, `already_current`, or `not_eligible`. Never invoke the
manual champion promotion command or edit champion state directly.
