# NeuroGolf repository guidance

## Mission and scope

This repository supplies deterministic infrastructure around later one-task
Codex workers for the 2026 NeuroGolf Championship. A task worker infers one ARC
rule and expresses it as a compact legal ONNX graph. This repository must not
grow into an exhaustive ARC solver.

Read this file and any closer `AGENTS.md` before editing. Task-local work is
limited to one `tasks/taskNNN/` directory unless a reusable infrastructure
change is explicitly requested.

## Sources of truth and protected files

The following are protected during ordinary task solving:

- `ARC-GEN/`, including every generator and ARC-AGI JSON file
- `data/` and `configs/task_map.json`
- `utils/neurogolf_utils.py`
- `configs/integrity.json`, `configs/solver.yaml`, `configs/scorer.yaml`,
  `sealed/`, judge report schemas, and `neurogolf/judge/`
- `champions/registry.sqlite`, champion artifacts, and promotion decisions
- repository-level tests that define the external gate

Never weaken, bypass, monkeypatch, replace, or silently refresh the acceptance
gate or integrity hashes, including through environment overrides. Proposed
corrections belong in a new JSON or Markdown request under
`test_change_requests/`; use `prompts/review_test_change.md` for an independent
review.

## Editable task artifacts

Within the assigned `tasks/taskNNN/`, a task worker may edit `solution/build.py`,
`solution/explanation.md`, `devtests/`, and noncanonical analysis/state files.
The shared ONNX lab and analysis helpers may be improved only when the change is
general, tested, and does not alter protected acceptance semantics.

## Official scorer contract

The immutable utility is final truth. Models use one float32 input and output
named `input` and `output`, both statically shaped `[1, 10, 30, 30]`; official
execution disables ONNX Runtime graph optimization and thresholds logits with
strict `> 0.0`. The objective is intermediate tensor bytes plus parameter
elements; input/output tensors and node/MAC counts are not scored. Initializers
and Constant values are parameters. Dynamic/nonpositive shapes, multiple I/O,
custom domains/functions/subgraphs, duplicate protected names, oversized files,
sequences, and excluded operations are rejected.

The default-domain opset is not restricted to 10. Use any opset supported by
the repository's pinned ONNX checker and ONNX Runtime; honor each operator's
minimum version (`Einsum`, for example, requires opset 12).

The current Kaggle leaderboard score is 8178.57, or about 20.4465 points per
task. Under the official scoring curve, that corresponds to a current
approximate average objective/cost budget of 94.97 (intermediate tensor bytes
plus parameter elements) per ONNX task graph. After full correctness, build and
golf with a goal of 94.97 or lower; never compromise semantic generalization or
gate correctness to meet the budget.

The ONNX lab helpers are optional and are not universally opset-aware.
`GraphBuilder` can record any positive opset, but an individual helper may emit
an operator schema valid only for a particular opset range. Before using a
helper at the selected opset, inspect its implementation/schema and run the
full ONNX checker plus ONNX Runtime. If its schema is incompatible, emit the
correct version-specific node directly with `onnx.helper`; never force opset 12
merely to accommodate a helper.

Prefer direct signed-logit output, fewer full-grid intermediates, bool over
float32 intermediates where useful, attributes over parameter tensors, and
small kernels. Correctness is always lexicographically more important than
golf score.

## Required task procedure

1. Read the task dossier, all canonical examples, mapped ARC-GEN principle,
   generator source/signature, current champion, and latest failure packet.
   Open and visually inspect `analysis/examples.png` in addition to the text
   rendering; do not infer the rule from text alone.
2. State a semantic rule, implement `solution/build.py`, build `candidate.onnx`,
   and update the explanation and editable development tests.
3. Run task dev tests, ONNX checker/runtime inspection, and the protected gate.
4. Repair structured counterexamples. Preserve every passing candidate before
   attempting score optimization.
5. Golf only after every required public, regression, fuzz, boundary, and sealed
   suite passes. Re-run the full gate after every model change.
6. Write a schema-valid `state/codex_result.json`. The gate CLI or parent
   orchestrator automatically promotes only the exact eligible fully passing,
   score-nonregressing candidate hash. Task workers must never invoke promotion
   directly or edit champion state.

Root verification uses `~/.venv/bin/python`, `~/.venv/bin/pytest`, or
`source ~/.venv/bin/activate && <command>`. Do not launch work for another task
from inside a task epoch.
