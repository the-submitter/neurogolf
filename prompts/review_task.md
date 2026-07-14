# Review one NeuroGolf solution

Read all `AGENTS.md`, the task dossier and canonical sources, builder,
visual examples, explanation, dev tests, gate result, failure packet, and model
inspection. Review semantic correctness, generalization, ONNX legality,
official-score assumptions, helper compatibility with the model's selected
opset, and test quality. Lab helpers are not universally opset-aware, so verify
emitted operator schemas rather than treating helper use as proof of legality,
and recommend a version-correct direct `onnx.helper` node when appropriate.

The current Kaggle leaderboard score is 8178.57, about 20.4465 points per task,
which implies a current approximate average objective/cost budget of 94.97
(intermediate tensor bytes plus parameter elements) per ONNX task graph. Review
whether a fully correct solution works deliberately toward 94.97 or lower while
keeping correctness and generalization lexicographically first.

Independently run the task tests and protected gate when possible. Do not edit
protected judge code or promote a model directly. Record actionable findings
and a schema-valid `state/codex_result.json`.
