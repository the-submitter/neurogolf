# Task-local NeuroGolf guidance

This workspace is exclusively for task 001 / `007bbfb7`. The root
`AGENTS.md` applies. Read `TASK_CONTEXT.md`, `analysis/`, canonical sources, and
`state/failure_packet.json` before changing the builder.

Open and inspect `analysis/examples.png` as an image in addition to reading the
text examples and dossier. Use the mapped ARC-GEN principle as a hypothesis,
not as a substitute for checking every input/output pair.

The ONNX lab helpers are optional and not universally opset-aware. The builder
accepts any positive opset, but each helper may implement only some operator
schema versions. Inspect helper/schema compatibility for the chosen opset and
validate with the full ONNX checker and ONNX Runtime. Use a version-correct
direct `onnx.helper` node when a lab helper is incompatible; do not force opset
12 just to fit a helper.

The current Kaggle leaderboard score is 8178.57, about 20.4465 points per task,
which implies a current approximate average objective/cost budget of 94.97
(intermediate tensor bytes plus parameter elements) per ONNX task graph. First
achieve complete correctness; then deliberately build and golf toward 94.97 or
lower without sacrificing generalization.

Editable: `solution/`, `devtests/`, and task-local analysis/state notes.
Protected: canonical data and generators, `utils/neurogolf_utils.py`, all
repository `neurogolf/judge/` files, integrity/sealed files, and champion state.
Never work on or launch another task from this directory. Run the external gate
instead of redefining acceptance. The parent orchestrator promotes only an exact
eligible passing hash; never invoke promotion or edit champion state directly.
