# Repair one NeuroGolf candidate

Read all applicable `AGENTS.md`, `TASK_CONTEXT.md`, the dossier, current builder,
visual `analysis/examples.png`, explanation, gate result, and failure packet.
Work only on this task. Infer the common semantic cause behind the diverse
counterexamples, update the ONNX
builder and focused dev tests, rebuild, and run the complete external gate.
Lab helpers are not universally opset-aware. Treat operator/helper schema
incompatibility at the candidate's selected opset as a possible implementation
failure, and use a version-correct direct `onnx.helper` node when needed.

The current Kaggle leaderboard score is 8178.57, about 20.4465 points per task,
which implies a current approximate average objective/cost budget of 94.97
(intermediate tensor bytes plus parameter elements) per ONNX task graph. Restore
complete correctness first, then optimize toward 94.97 or lower without
sacrificing generalization.

Preserve the previous passing candidate if one exists. Do not trade correctness
for score, alter protected infrastructure, or encode the exposed cases as a
lookup table. Propose suspected gate defects through `test_change_requests/`.
Write a schema-valid `state/codex_result.json` with commands, hashes, and result.
Eligible passing hashes are auto-promoted outside the isolated worker.
