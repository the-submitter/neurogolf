# NeuroGolf repository bootstrap plan

## Inspection baseline

- The canonical mapping contains exactly 400 unique `taskNNN` keys and 400
  unique ARC task IDs. The repository has 400 Kaggle JSON files, 400 mapped
  ARC-GEN task modules, and complete 400-file ARC-AGI training and evaluation
  splits.
- ARC-GEN is a Git submodule pinned at revision
  `2394883d865927b79deff2242f98f99a4fdf0f32` (`v1.0.0-3-g2394883`).
- Every mapped generator defines `generate()` and zero-argument `validate()`.
  There are 329 distinct `generate()` signatures, but every required parameter
  has a default. All task modules import `common`; two additionally import a
  standard-library module. The adapter will therefore import by absolute path,
  temporarily expose the canonical ARC-GEN root for `common`, serialize access
  to the process-global RNG/module namespace, seed `common.random`, call the
  zero-argument generator, and restore random/import state.
- The official utility is the 2026-05-14 revision. Its fixed contract is a
  float32 `[1, 10, 30, 30]` input/output, IR version 10, a 1.44 MiB file limit,
  ONNX Runtime graph optimizations disabled, output
  threshold `> 0.0`, strict static-shape accounting, initializer/Constant
  parameter accounting, and the published excluded-op list.
- Opset 10 is only the official starter model's default. Candidates may use any
  standard-domain opset accepted by the pinned ONNX checker/runtime; the ONNX
  lab defaults to opset 12 so operators such as `Einsum` are available.
- The builder's opset field is not a promise that every lab helper supports
  every historical/current operator schema. Workers must inspect helper/schema
  compatibility and may bypass helpers with version-correct `onnx.helper` code.
- The existing `requirements.txt` contains the official pinned NumPy/ONNX
  versions, but trailing notebook-redirection tokens make three lines invalid
  as a requirements export. It will be retained and normalized after the
  modern project metadata is established.
- Existing repository files and the ARC-GEN submodule are treated as user and
  competition data. Bootstrap work will not modify canonical data, ARC-GEN, or
  `utils/neurogolf_utils.py`.

## Implementation phases and verification gates

1. **Project foundation** — add `pyproject.toml`, typed YAML/environment/CLI
   configuration, package exceptions/logging, task/grid models, robust task
   mapping and JSON loaders, integrity manifest support, and Typer CLI groups.
   Gate: configuration, model, mapping, loading, and integrity unit tests pass.
2. **ARC-GEN oracle** — add absolute-path importer, signature metadata,
   deterministic seed isolation, retrying generation, validation invocation,
   tracing, and boundary-biased seed scheduling. Gate: representative modules
   from simple and complex signatures import, validate, and reproduce samples.
3. **Official scorer boundary** — load the official module without edits,
   provide exact conversion/session/sanitization/scoring adapters, candidate
   comparisons, and legality diagnostics. Gate: wrappers match official
   functions for conversions, thresholds, parameters, legal models, and
   deliberately invalid graph families.
4. **ONNX laboratory** — implement static-shape graph construction, constants,
   convolution/color maps, shifts/transforms/slices/gathers, masks, reductions,
   rendering, inspection/cost estimates, checker/runtime helpers, and
   conservative rewrites. Gate: every public helper has focused tests and
   official final scoring remains authoritative.
5. **Analysis and visualization** — implement composable deterministic color,
   spatial, component, D4/translation, local-rule, and generator AST analyses;
   build schema-valid dossiers plus text, diff, Markdown, ANSI, and PNG output.
   Gate: a real mapped task produces all required analysis artifacts.
6. **Acceptance gate** — verify protected hashes; check file/legality; run
   public, provided ARC-GEN, regression, fresh fuzz, boundary-biased, and sealed
   cases; score; compare to the champion; emit schema-valid gate and diverse
   counterexample reports. Gate: known passing/failing fixtures exercise both
   result paths. CLI/orchestrator policy then auto-promotes only eligible exact
   passing hashes through the independently protected promotion transaction.
7. **Champion lifecycle** — add transactional SQLite migrations, immutable
   history artifacts, eligibility checks tied to a passing gate result,
   lexicographic correctness/score comparison, promotion, and rollback. Gate:
   passing promotion is atomic, incorrect promotion is rejected, and rollback
   preserves history.
8. **Task workspaces and orchestration** — add resumable preparation, generated
   dev tests and a neutral candidate builder scaffold, protected root/task
   instructions, solver/repair/golf/review prompts, isolated Codex command
   construction, bounded scheduling, event/status persistence, and cancellation.
   No solver process is launched during bootstrap. Gate: command construction
   is tested with mocked subprocesses and `task prepare` creates every artifact.
9. **Documentation and release verification** — complete README and maintenance
   scripts; normalize the legacy requirements export; run doctor, all-400 map
   validation, task preparation, seeded sampling, the full test suite, Ruff,
   Pyright, scorer parity, schema checks, and task-map validation. Any failure is
   repaired before completion is reported.

## Trust boundary

Canonical data, ARC-GEN, the official utility, gate implementation, integrity
and sealed manifests, registry database, and promotion code are protected.
Task-local builders, candidate models, explanations, dev tests, analysis output,
and state reports are editable. The protected gate only establishes eligibility;
CLI/orchestrator policy may automatically call the separate transaction that
revalidates the candidate hash, judge result, integrity state, and champion
comparison.

## Deliberate non-goals

This bootstrap will not infer or encode solutions for any mapped task, launch a
task-specific Codex epoch, or implement an exhaustive ARC solver. Generated
workspace builders are neutral templates that require a later semantic solver.
