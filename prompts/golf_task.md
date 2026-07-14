# Golf one fully correct NeuroGolf candidate

Read every applicable `AGENTS.md`, the dossier, builder, explanation, official
cost report, passing gate result, and current champion. First copy/preserve the
fully passing ONNX artifact. Optimize only its representation: reduce full-grid
intermediates and parameters, prefer bool tensors and signed direct output, and
use safe graph rewrites. Do not change the inferred rule or weaken tests. Lab
helpers are not universally opset-aware; inspect operator schemas and use
direct `onnx.helper` nodes when needed.

The current Kaggle leaderboard score is 8178.57, about 20.4465 points per task,
which implies a current approximate average objective/cost budget of 94.97
(intermediate tensor bytes plus parameter elements) per ONNX task graph. Treat
94.97 or lower as the deliberate post-correctness golf target.

After every change run dev tests and the complete external gate. Keep a change
only when correctness remains complete and the official objective does not
regress. Do not promote directly; the parent orchestrator handles eligible
hashes. Write a schema-valid `state/codex_result.json`.
