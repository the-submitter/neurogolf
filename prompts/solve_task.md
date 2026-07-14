# Solve one NeuroGolf task

Work only in the prepared task workspace named in `TASK_CONTEXT.md`. Read every
applicable `AGENTS.md`, then read the dossier, report, text/image examples,
canonical generator, mapped data, current champion metadata, and any failure
packet. Infer the semantic transformation; do not build a generic ARC searcher.
Open `analysis/examples.png` with an image viewer and compare every visual pair;
use the mapped ARC-GEN principle as a hypothesis to verify, not as proof.

Implement `solution/build.py`, create `solution/candidate.onnx`, explain the
rule and graph in `solution/explanation.md`, and add focused editable dev tests.
Choose any standard-domain opset accepted by the pinned checker/runtime. ONNX
lab helpers are optional and not universally opset-aware: inspect a helper's
version-specific schema before using it, and use a correct direct `onnx.helper`
node if it is incompatible. Do not select opset 12 merely because it is the lab
default.

The current Kaggle leaderboard score is 8178.57, about 20.4465 points per task,
which implies a current approximate average objective/cost budget of 94.97
(intermediate tensor bytes plus parameter elements) per ONNX task graph. First
achieve complete correctness; then deliberately build and golf toward 94.97 or
lower without sacrificing generalization.

Run the builder, ONNX checker, ONNX Runtime, conversion parity, task dev tests,
and the external gate command from `TASK_CONTEXT.md`. Use structured
counterexamples to repair until every required suite passes. Preserve a passing
candidate before golfing. Do not edit protected repository files or weaken the
gate. Submit possible test/judge defects under `test_change_requests/`.

Finish by writing `state/codex_result.json` conforming to the supplied schema.
Report the exact candidate hash and passing gate path. The isolated worker uses
the no-promotion gate command from `TASK_CONTEXT.md`; the parent orchestrator
auto-promotes the exact eligible hash after validating this result.
