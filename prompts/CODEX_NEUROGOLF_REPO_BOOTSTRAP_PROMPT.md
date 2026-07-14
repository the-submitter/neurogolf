# Codex Repository Bootstrap Prompt: NeuroGolf Codex-First Solver Infrastructure

You are working inside the root of a NeuroGolf Championship repository in VS Code.

Your task is to **design, implement, test, and document the complete repository infrastructure required for a Codex-first solver system** before any task-specific Codex solver epochs are launched.

Do not solve any of the 400 ARC tasks in this bootstrap phase.

The finished repository must allow later Codex 5.6 Sol `Extra High` or `Max` task workers to independently analyze one mapped ARC task, implement a compact ONNX solution, run editable development tests, invoke an immutable external acceptance gate, receive structured counterexamples, and promote only verified champion models.

Work autonomously in the repository. Inspect the existing files before changing anything. Preserve existing user data and Git submodules.

---

## 1. Existing repository layout and source-of-truth files

The working root directory already contains the following important paths.

### ARC-GEN submodule

```text
ARC-GEN/
```

This is a root-level Git submodule.

Each mapped task generator is located at:

```text
ARC-GEN/tasks/task_<task-id>.py
```

Each task file contains:

```python
generate(...)
validate(...)
```

There are 400 relevant ARC-AGI-1 tasks.

Treat the ARC-GEN task files as canonical source material.

Do not modify the ARC-GEN submodule during this bootstrap unless a compatibility adapter is required outside the submodule.

### Kaggle NeuroGolf task data

```text
data/<task-num>.json
```

Each file contains:

```json
{
  "train": [...],
  "test": [...],
  "arc-gen": [...]
}
```

`<task-num>` is the Kaggle NeuroGolf task number, expected to range from 1 to 400.

### ARC-AGI-1 source data

```text
ARC-GEN/external/ARC-AGI/data/training/<task-id>.json
ARC-GEN/external/ARC-AGI/data/evaluation/<task-id>.json
```

These files contain:

```json
{
  "train": [...],
  "test": [...]
}
```

A task may be found in either the `training` or `evaluation` directory.

### Task mapping

```text
configs/task_map.json
```

This maps:

```text
<task-num> ↔ <task-id>
```

Use it as the canonical mapping between:

- Kaggle task number
- ARC-AGI task ID
- ARC-GEN generator file
- ARC-AGI source JSON

Validate this mapping thoroughly.

### Official competition utilities

```text
utils/neurogolf_utils.py
```

This is the official NeuroGolf utility module. It contains:

- Official ONNX legality checks
- Official scoring logic
- ONNX Runtime execution behavior
- Input/output conversion
- `single_layer_conv2d_network()`
- Visualization helpers
- Example verification utilities

Treat this file as immutable competition code.

Reuse its logic through adapters where practical, but do not modify the canonical file.

Create and verify an integrity hash for it.

### Official starter example

```text
utils/the-2026-neurogolf-championship.py
```

This is the py:percent conversion of the official notebook and demonstrates a handcrafted single-layer Conv2D ONNX solution.

Use it as an implementation reference where helpful.

### Existing Python dependencies

```text
requirements.txt
```

Convert the project to a modern `pyproject.toml`, while preserving all required dependencies.

Do not delete `requirements.txt` until compatibility and reproducibility are verified. It may be retained as an exported or legacy dependency file.

---

## 2. System objective

Build a **Codex-first NeuroGolf solver platform** with the following trust boundary.

### Codex-controlled components

Later task-solving Codex agents may modify:

- Task-specific solution builders
- Task-specific ONNX models
- Task-specific explanations
- Task-specific development tests
- Analysis tooling
- Visualization tooling
- Shared ONNX helper code
- Diagnostic tooling
- Proposed reusable primitives

### Deterministic and protected components

Task-solving Codex agents must not directly control:

- Canonical ARC-GEN generators
- Canonical ARC-AGI data
- Canonical Kaggle task data
- Official scorer source
- Acceptance-gate implementation
- Integrity manifests
- Champion promotion logic
- Champion registry
- Sealed validation seeds
- Canonical model promotion decisions

The task worker may invoke the acceptance gate and read its structured results, but may not redefine what constitutes a pass.

---

## 3. Core design principle

Do not implement a large deterministic ARC solver or exhaustive program synthesizer.

The semantic task solver will later be Codex 5.6 Sol Extra High.

The deterministic repository infrastructure should only provide:

1. Canonical task loading
2. Task mapping and validation
3. ARC-GEN invocation
4. Ground-truth generation
5. Generator validation
6. Structured task analysis
7. Text and image visualization
8. ONNX graph construction helpers
9. Candidate execution
10. Official scorer integration
11. Development testing
12. External acceptance gating
13. Counterexample generation
14. Champion comparison and promotion
15. Reproducible orchestration
16. Machine-readable reports

---

## 4. Required target repository structure

Adapt this structure to the existing repository rather than destructively replacing files.

```text
.
├── AGENTS.md
├── pyproject.toml
├── requirements.txt
├── README.md
├── configs/
│   ├── task_map.json
│   ├── solver.yaml
│   ├── scorer.yaml
│   └── integrity.json
│
├── ARC-GEN/
├── data/
├── utils/
│   ├── neurogolf_utils.py
│   └── the-2026-neurogolf-championship.py
│
├── neurogolf/
│   ├── __init__.py
│   ├── cli.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── loader.py
│   │
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── mapping.py
│   │   ├── loader.py
│   │   ├── models.py
│   │   └── integrity.py
│   │
│   ├── arcgen/
│   │   ├── __init__.py
│   │   ├── importer.py
│   │   ├── generator.py
│   │   ├── validator.py
│   │   ├── sampling.py
│   │   └── tracing.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── dossier.py
│   │   ├── color_analysis.py
│   │   ├── spatial_analysis.py
│   │   ├── component_analysis.py
│   │   ├── transform_analysis.py
│   │   ├── local_rule_analysis.py
│   │   ├── generator_analysis.py
│   │   └── report.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── text.py
│   │   ├── image.py
│   │   └── diff.py
│   │
│   ├── onnx_lab/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── graph.py
│   │   ├── constants.py
│   │   ├── conv.py
│   │   ├── masks.py
│   │   ├── spatial.py
│   │   ├── reductions.py
│   │   ├── rendering.py
│   │   ├── inspect.py
│   │   ├── optimize.py
│   │   ├── score_estimator.py
│   │   └── templates/
│   │
│   ├── judge/
│   │   ├── __init__.py
│   │   ├── official_adapter.py
│   │   ├── candidate_runner.py
│   │   ├── legality.py
│   │   ├── gate.py
│   │   ├── counterexamples.py
│   │   ├── sealed.py
│   │   ├── promotion.py
│   │   ├── registry.py
│   │   └── reports.py
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── prepare.py
│   │   ├── codex.py
│   │   ├── solve.py
│   │   ├── schedule.py
│   │   └── status.py
│   │
│   └── schemas/
│       ├── dossier.schema.json
│       ├── gate_result.schema.json
│       ├── failure_packet.schema.json
│       └── codex_result.schema.json
│
├── prompts/
│   ├── solve_task.md
│   ├── repair_task.md
│   ├── golf_task.md
│   ├── review_task.md
│   └── review_test_change.md
│
├── tasks/
│   └── .gitkeep
│
├── champions/
│   ├── models/
│   ├── history/
│   └── registry.sqlite
│
├── sealed/
│   ├── seeds.json
│   └── README.md
│
├── test_change_requests/
│   └── .gitkeep
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── parity/
│   └── fixtures/
│
└── scripts/
    ├── verify_environment.py
    ├── validate_task_map.py
    ├── prepare_all_tasks.py
    ├── run_gate.py
    ├── inspect_model.py
    └── export_requirements.py
```

Use sensible simplifications where appropriate, but preserve the architectural separation.

---

## 5. Implementation requirements

### 5.1 Packaging and dependency management

Create `pyproject.toml` using a modern build system.

Use a `src` layout only if it clearly improves the existing repository. Otherwise use the root package structure shown above.

Include:

- Project metadata
- Supported Python version
- Runtime dependencies
- Development dependencies
- Test dependencies
- Ruff configuration
- Pyright or mypy configuration
- Pytest configuration
- CLI entry points

Recommended dependencies include, as applicable:

```text
numpy
onnx
onnxruntime
onnx-tool
pydantic
pydantic-settings
PyYAML
jsonschema
networkx
sympy
matplotlib
pillow
rich
typer
pytest
pytest-xdist
hypothesis
```

Avoid unnecessary heavy ML frameworks.

Do not make PyTorch a required dependency unless a specific compatibility reason exists.

Preserve compatibility with the exact ONNX versions expected by the official competition utility.

Add:

```bash
python -m neurogolf.cli doctor
```

to verify the environment.

---

### 5.2 Configuration

Implement typed configuration with `pydantic-settings`.

Support:

```text
configs/solver.yaml
configs/scorer.yaml
environment variables
optional .env
CLI overrides
```

Precedence should be documented and deterministic.

Suggested configuration areas:

```yaml
paths:
  arcgen_root: ARC-GEN
  kaggle_data_root: data
  task_map: configs/task_map.json
  official_utils: utils/neurogolf_utils.py
  task_workspace_root: tasks
  champion_root: champions

generation:
  development_seed_count: 64
  regression_seed_count: 256
  gate_fuzz_count: 512
  sealed_seed_count: 2048
  max_attempts_per_seed: 50

gate:
  timeout_seconds: 600
  max_failure_examples: 8
  require_public_pass: true
  require_regression_pass: true
  require_fuzz_pass: true

codex:
  model: gpt-5.6-sol
  reasoning_effort: xhigh
  max_parallel_tasks: 4
```

---

### 5.3 Task mapping and loading

Implement a robust mapping layer over `configs/task_map.json`.

For each task, expose a typed record similar to:

```python
class TaskRecord(BaseModel):
    task_num: int
    task_id: str
    kaggle_json_path: Path
    arcgen_generator_path: Path
    arc_agi_json_path: Path
    arc_agi_split: Literal["training", "evaluation"]
```

Validate:

- Exactly 400 mapped tasks
- Unique Kaggle task numbers
- Unique ARC task IDs unless explicitly justified
- Kaggle JSON exists
- ARC-GEN generator exists
- ARC-AGI JSON exists in training or evaluation
- Generator has callable `generate`
- Generator has callable `validate`
- Kaggle JSON fields are well-formed
- ARC-AGI JSON fields are well-formed

Add:

```bash
python -m neurogolf.cli tasks validate-map
python -m neurogolf.cli tasks show 137
python -m neurogolf.cli tasks list
```

---

### 5.4 Safe ARC-GEN importing

Implement task-module loading by absolute file path.

Do not rely on fragile current-working-directory imports.

The importer must:

- Resolve the ARC-GEN root
- Preserve submodule import behavior
- Safely load `task_<task-id>.py`
- Avoid polluting module names across tasks
- Expose `generate()` and `validate()`
- Capture generator metadata
- Support reproducible seeding where possible
- Detect unsupported or unusual generator signatures
- Produce actionable diagnostics

Inspect actual generator signatures across all 400 tasks before finalizing the adapter.

Do not assume every generator has an identical signature without verification.

---

### 5.5 Generator execution and validation

Create a canonical generator-oracle API.

It should:

1. Generate an input/output pair
2. Invoke the task's `validate()` where applicable
3. Normalize the pair into the standard ARC structure
4. Reject malformed examples
5. Record generation seed and parameters when recoverable
6. Retry failed generations within configured limits
7. Return deterministic metadata

Expose a typed object similar to:

```python
class GeneratedExample(BaseModel):
    task_num: int
    task_id: str
    seed: int
    input: list[list[int]]
    output: list[list[int]]
    metadata: dict[str, Any]
```

Add CLI commands:

```bash
python -m neurogolf.cli arcgen sample 137 --count 10 --seed 123
python -m neurogolf.cli arcgen validate 137 --count 100
```

---

### 5.6 Official scorer adapter

Do not reimplement the official scorer from memory where the official implementation can be reused.

Create an adapter around:

```text
utils/neurogolf_utils.py
```

Requirements:

- Load the canonical module without modifying it
- Verify its SHA-256 hash
- Reuse its legality and scoring functions where practical
- Match its input conversion exactly
- Match its output thresholding exactly
- Match its ONNX Runtime settings exactly
- Match its static-shape restrictions
- Match excluded-op restrictions
- Match parameter counting
- Match intermediate-memory calculation
- Match score formula

Where direct reuse is awkward, write a thin wrapper or parity-tested equivalent.

Create comprehensive parity tests comparing the adapter with the official functions on:

- Identity models
- Single Conv models
- Models with constants
- Models with initializers
- Bool intermediates
- Float16 intermediates
- Invalid dynamic shapes
- Invalid multiple inputs/outputs
- Excluded operators
- Duplicate names
- Oversized files
- Output threshold edge cases

The official file must remain the source of truth.

Add:

```bash
python -m neurogolf.cli scorer inspect path/to/model.onnx
python -m neurogolf.cli scorer verify --task 137 path/to/model.onnx
```

---

### 5.7 Candidate execution

Implement a clean candidate runner that:

- Loads and sanitizes an ONNX model consistently with the official utility
- Uses ONNX Runtime with graph optimizations disabled
- Accepts a normalized ARC grid
- Converts it to official `[1, 10, 30, 30]` input
- Executes the model
- Applies official `> 0` thresholding
- Converts output back to an ARC grid
- Reports malformed one-hot cells
- Captures runtime exceptions
- Supports batch evaluation over examples
- Is deterministic

Provide a structured comparison object containing:

```text
expected
actual
pixel differences
shape differences
special output colors
runtime error
```

---

### 5.8 External gate judge

Implement an immutable acceptance gate under `neurogolf/judge/`.

The task-solving Codex worker may call it, but should not be instructed to edit it.

The gate must perform:

1. Integrity verification
2. Candidate-file existence and size check
3. Official ONNX legality checks
4. Public Kaggle train/test validation
5. Provided Kaggle `arc-gen` validation
6. Fixed regression-seed validation
7. Fresh fuzz generation
8. Boundary-biased or mutation-biased generation where feasible
9. Official memory and parameter scoring
10. Candidate/champion comparison
11. Structured result generation
12. Structured failure-packet generation

Suggested command:

```bash
python -m neurogolf.cli gate run \
  --task 137 \
  --candidate tasks/task137/solution/candidate.onnx \
  --output tasks/task137/state/gate_result.json
```

Keep promotion out of the protected gate implementation. After the gate emits
an eligible passing result, CLI/orchestrator policy must automatically invoke
the separate hash-bound promotion transaction; allow an explicit opt-out for
isolated task workers whose parent orchestrator performs the same step.

---

### 5.9 Sealed validation

Create a separate sealed validation layer.

It may still use ARC-GEN generators, but its case selection must be independent from task development tests.

Requirements:

- Stable repository-level seed manifest
- Separate seed ranges from development tests
- Configurable case count
- Integrity hash
- Version identifier
- No task-specific modification during normal solving
- Re-run after judge-version changes

The sealed seed list does not need to be cryptographically hidden from the local user, but task workers should not be allowed to silently redefine it.

---

### 5.10 Counterexample generation

When a candidate fails, write a concise machine-readable failure packet.

Include:

- Candidate hash
- Judge version
- Task number and ID
- Failure counts
- Failure-family summary
- Representative diverse failures
- Generator seeds
- Inputs
- Expected outputs
- Actual outputs
- Pixel difference coordinates
- Shape differences
- Special one-hot errors
- Runtime errors
- Relevant scorer diagnostics

Select diverse failures rather than merely the first N.

A simple clustering by failure signature is sufficient.

Create:

```text
neurogolf/schemas/failure_packet.schema.json
```

Validate every emitted packet against the schema.

---

### 5.11 Development tests

Create task-local editable development tests under:

```text
tasks/taskNNN/devtests/
```

These may later be changed by task-solving Codex agents.

The bootstrap infrastructure must generate initial tests covering:

- Kaggle train examples
- Kaggle test examples
- Kaggle arc-gen examples
- A small reproducible ARC-GEN sample
- Candidate build smoke test
- ONNX checker
- ONNX Runtime execution
- Conversion parity

The external acceptance gate remains separate.

---

### 5.12 Analysis suite

Implement a lightweight but useful deterministic task-analysis suite.

Do not attempt exhaustive ARC synthesis.

Generate a dossier containing:

- Task number
- Task ID
- All source paths
- Input/output examples
- Input/output shape relationships
- Color sets
- Added/removed/preserved colors
- Color transition matrix
- Pixel difference statistics
- Row and column histograms
- Bounding boxes
- Connected-component summaries
- D4 transform comparisons
- Best fixed translations
- Periodicity estimates
- Symmetry indicators
- Local neighborhood change tables
- Bounded-receptive-field plausibility
- ARC-GEN generator source
- ARC-GEN `task_list()` side-comment principle mapped by task number and ID
- Generator signature
- Generator AST summary
- Any discovered input/output write relationships
- Official scorer constraints
- Current champion metadata
- Existing failure summaries

Output:

```text
tasks/taskNNN/analysis/dossier.json
tasks/taskNNN/analysis/report.md
tasks/taskNNN/analysis/examples.txt
tasks/taskNNN/analysis/examples.png
```

Create JSON Schema validation for the dossier.

Codex may later improve this analysis suite.

---

### 5.13 Text visualizer

Implement a text-only visualizer because it is highly useful for Codex and terminal workflows.

It should support:

- Numeric grid rendering
- Compact color-symbol rendering
- ANSI-color rendering when enabled
- Side-by-side input/output display
- Expected/actual/diff display
- Row and column indices
- Bounding box annotations
- Changed-cell highlighting
- Plain-text mode suitable for prompt context
- Markdown-safe output

Example symbolic palette:

```text
0=.
1=B
2=R
3=G
4=Y
5=H
6=M
7=O
8=C
9=W
```

Do not rely solely on matplotlib.

Retain an image visualizer using the official utility or an improved compatible
wrapper. Task-solving prompts must explicitly require opening and visually
inspecting the rendered image as well as reading text examples.

---

### 5.14 ONNX laboratory

Implement a small scorer-aware ONNX construction toolkit.

This is not a deterministic solver DSL. It is a build-time helper library for Codex-authored solutions.

Required capabilities:

- Graph builder
- Unique safe names
- Static shape metadata
- Initializer creation
- Standard input/output creation
- Opset selection
- Any standard-domain opset supported by the pinned ONNX/ONNX Runtime versions;
  do not impose an opset-10 ceiling
- An opset-12+ `Einsum` helper
- Direct-to-output graph construction
- Conv and grouped Conv
- Depthwise Conv
- 1×1 color mapping
- Spatial shifts
- Transpose and flips
- Slice and Gather helpers
- Bool mask creation
- Row/column reductions
- Color counts
- ArgMax/ArgMin helpers
- Broadcast/Expand
- Signed-logit rendering
- ONNX checker
- ONNX Runtime parity execution
- Model inspection
- Score estimation
- Final official score call
- Basic graph rewrites
- Dead-node removal
- Constant folding where safe
- Redundant Cast removal
- Redundant Transpose removal
- Common-subexpression handling where it improves official cost
- Final-node fusion opportunities

Every helper must be tested.

Allow task-specific builders to bypass helpers and use `onnx.helper` directly.
Until a helper is explicitly version-matrix tested, instruct workers that it is
not universally opset-aware. They must inspect the helper and selected operator
schema, validate with the full checker/runtime, and use a correct direct node
rather than forcing the lab's default opset.

---

### 5.15 Scorer-aware optimization model

Implement a local cost estimator matching:

```text
objective = intermediate_tensor_bytes + parameter_elements
points = max(1, 25 - log(max(1, objective)))
```

Use the official scorer for final truth.

Record the current 8178.57-point Kaggle leaderboard baseline: about 20.4465
points per task and an approximate objective of 94.97. Use it as a
post-correctness graph
budget.

The optimizer must understand:

- Input and output tensors are excluded from intermediate-memory cost
- Node count is not directly scored
- MACs are not scored
- Full-grid intermediates are expensive
- Bool intermediates are cheaper than float32
- Initializer elements are parameters
- Constant node values are parameters
- Operator attributes may be free
- ONNX Runtime graph optimizations are disabled
- Direct-to-output computation is highly desirable
- Signed logits avoid final threshold nodes

Provide a per-node/tensor cost report.

Suggested command:

```bash
python -m neurogolf.cli onnx inspect model.onnx --cost-breakdown
```

---

### 5.16 Champion registry

Use SQLite for the canonical champion registry.

Track:

- Task number
- Task ID
- Candidate SHA-256
- ONNX path
- Builder path
- Semantic explanation path
- Judge version
- Generator/submodule revision
- Official utility hash
- Public pass counts
- Regression pass counts
- Fuzz pass counts
- Sealed pass counts
- Memory
- Parameters
- Objective
- Points
- Creation timestamp
- Parent candidate
- Promotion history
- Status
- Notes

Promotion must be atomic.

Never overwrite the prior champion artifact.

Store historical models under:

```text
champions/history/taskNNN/
```

Provide:

```bash
python -m neurogolf.cli champions list
python -m neurogolf.cli champions show 137
python -m neurogolf.cli champions promote --task 137 --candidate ...
python -m neurogolf.cli champions rollback --task 137 --sha ...
```

Only a fully passing candidate may be promoted.

Correctness is lexicographically more important than score.

---

### 5.17 Task workspace preparation

Implement:

```bash
python -m neurogolf.cli task prepare 137
```

This must create:

```text
tasks/task137/
├── AGENTS.md
├── analysis/
├── devtests/
├── solution/
│   ├── build.py
│   └── explanation.md
└── state/
```

It must populate:

- Dossier
- Text examples
- Image examples
- Initial development tests
- Task-local prompt context
- Current champion metadata if present
- Task-local `AGENTS.md`

Do not solve the task.

Also support:

```bash
python -m neurogolf.cli task prepare-all
```

with safe resumability.

---

### 5.18 Codex prompt templates

Create these prompt templates:

```text
prompts/solve_task.md
prompts/repair_task.md
prompts/golf_task.md
prompts/review_task.md
prompts/review_test_change.md
```

They must instruct later Codex task workers to:

- Read all applicable `AGENTS.md`
- Work on one task only
- Read the dossier, generator, data, current champion, and failures
- Infer the semantic rule
- Write `solution/build.py`
- Build `candidate.onnx`
- Add development tests
- Invoke the external gate
- Open and visually inspect `analysis/examples.png` in addition to text examples
- Repair failures
- Golf only after correctness
- Preserve passing candidates
- Write a schema-valid `codex_result.json`
- Avoid modifying protected infrastructure
- Propose judge/test corrections through `test_change_requests/`

Do not launch these prompts during bootstrap.

---

### 5.19 Root AGENTS.md

Create a repository-root `AGENTS.md`.

It must define:

- Repository mission
- Protected files
- Editable files
- Sources of truth
- Official scorer rules
- Task-solving procedure
- ONNX optimization priorities
- Testing obligations
- Champion-promotion rules
- Required output artifacts
- Prohibition against weakening the external gate
- Mechanism for submitting test-change requests
- Scope rules for task-local work

Also create a task-local `AGENTS.md` template.

The root `AGENTS.md` belongs at:

```text
<project-root>/AGENTS.md
```

Do not place it under `.codex/`.

---

### 5.20 Codex orchestrator

Implement an outer orchestrator, but do not invoke Codex task solving during bootstrap.

The orchestrator should later support:

```bash
python -m neurogolf.cli solve task 137
python -m neurogolf.cli solve many 1 2 3 4
python -m neurogolf.cli solve pending
python -m neurogolf.cli solve failed
python -m neurogolf.cli solve golfed
```

Use one independent Codex CLI process per task.

Target configuration:

```text
model: gpt-5.6-sol
reasoning effort: xhigh
sandbox: workspace-write
approval policy: never only in isolated task workspace
```

Use the `neurogolf-xhigh` profile (`~/.codex/neurogolf-xhigh.config.toml`) by
default while keeping the model and reasoning effort configurable. Also create
a repo-scoped `.agents/skills/solve-neurogolf-task/` skill for one-task Codex IDE
work; it must respect the model/reasoning selection active in the IDE.

Support:

- Per-task working directory
- Per-task event logs
- JSON output
- Output schema
- Final-message capture
- Timeout
- Retry epochs
- Fresh epoch
- Resume epoch
- Repair prompt
- Golf prompt
- Maximum parallel workers
- Graceful cancellation
- Status persistence
- Failure isolation

Do not use one shared mutable task directory across concurrent workers.

Do not start Codex task workers in this bootstrap assignment.

---

## 6. Testing requirements

Create comprehensive tests.

### Unit tests

Cover:

- Configuration
- Task mapping
- JSON loading
- ARC-GEN importing
- Generator invocation
- Validator invocation
- Conversion functions
- Text visualization
- Dossier generation
- ONNX builder helpers
- Score estimation
- Registry operations
- Schema validation
- Integrity hashes

### Integration tests

Cover:

- Prepare one real mapped task
- Generate examples from one real ARC-GEN task
- Validate generated examples
- Build an identity model
- Build a simple Conv model
- Run official scorer adapter
- Run gate on a known passing trivial fixture
- Run gate on a known failing fixture
- Produce a failure packet
- Promote a passing champion
- Reject an incorrect promotion
- Roll back a champion

### Official parity tests

Compare local wrappers against `utils/neurogolf_utils.py`.

Do not mock away the official behavior.

### Property-based tests

Use Hypothesis where useful for:

- Grid conversion
- Text rendering
- Difference calculations
- Shape validation
- Small ONNX helper invariants

### Smoke tests

Provide:

```bash
python -m neurogolf.cli doctor
python -m neurogolf.cli tasks validate-map
python -m neurogolf.cli task prepare 1
python -m neurogolf.cli arcgen sample 1 --count 3
pytest -q
```

All must pass before declaring completion.

---

## 7. Security and integrity requirements

Implement SHA-256 integrity verification for:

- `utils/neurogolf_utils.py`
- `configs/task_map.json`
- Sealed seed manifest
- ARC-GEN submodule revision
- Optional key judge files

Do not make normal development impossible, but make unauthorized or accidental changes visible.

The external gate should refuse canonical promotion when protected integrity checks fail.

Provide an explicit maintenance command to refresh hashes after intentional reviewed changes.

---

## 8. Error handling and logging

Use structured logging.

Every major command must:

- Return a nonzero exit code on failure
- Print a concise human-readable summary
- Optionally write a JSON report
- Preserve traceback information in debug mode
- Avoid swallowing exceptions
- Avoid silently skipping malformed examples
- Distinguish infrastructure error from candidate failure

Use clear exception classes.

---

## 9. Documentation

Create or update `README.md` with:

- Architecture overview
- Trust boundaries
- Setup
- Dependency installation
- Task mapping
- Preparing a workspace
- Running analysis
- Building an ONNX candidate
- Running development tests
- Running the gate
- Inspecting score
- Promoting champions
- Codex orchestration
- Parallel task solving
- Troubleshooting
- Protected-file policy
- Repository conventions

Add focused module docstrings and type hints.

Avoid excessive comments that merely repeat code.

---

## 10. Implementation sequence

Follow this order.

### Phase 1: inspect and plan

- Inspect the repository
- Inspect `requirements.txt`
- Inspect `configs/task_map.json`
- Inspect several ARC-GEN generators
- Inspect the official scorer
- Inspect the starter script
- Identify generator signature variations
- Write a concise implementation plan to `docs/bootstrap-plan.md`

### Phase 2: packaging and core task model

- Create `pyproject.toml`
- Create package skeleton
- Implement typed configuration
- Implement task mapping
- Implement task/data loading
- Add validation commands
- Add tests

### Phase 3: ARC-GEN adapter

- Implement safe import
- Implement generator invocation
- Implement validator invocation
- Implement reproducible sampling
- Add tests across a representative set of generators

### Phase 4: official scorer integration

- Implement immutable adapter
- Implement parity tests
- Implement candidate execution
- Implement score inspection
- Add integrity checking

### Phase 5: analysis and visualization

- Implement text visualizer
- Implement image wrapper
- Implement dossier generation
- Implement generator source analysis
- Add schemas and tests

### Phase 6: ONNX laboratory

- Implement builder helpers
- Implement scorer-aware inspection
- Implement optimization utilities
- Add exact parity tests

### Phase 7: gate and counterexamples

- Implement regression/fuzz/sealed testing
- Implement gate reports
- Implement failure packets
- Add schemas
- Add integration tests

### Phase 8: champion registry

- Implement SQLite registry
- Implement promotion
- Implement rollback
- Implement history
- Add tests

### Phase 9: task preparation and prompts

- Implement task workspace generation
- Create root and task-local `AGENTS.md`
- Create Codex prompt templates
- Add tests

### Phase 10: orchestrator

- Implement Codex CLI command construction
- Implement scheduling
- Implement status persistence
- Do not execute task-solving Codex runs
- Add mocked orchestration tests

### Phase 11: final verification

Run:

```bash
python -m neurogolf.cli doctor
python -m neurogolf.cli tasks validate-map
python -m neurogolf.cli task prepare 1
python -m neurogolf.cli arcgen sample 1 --count 3 --seed 123
pytest -q
```

Also run linting and type checking.

---

## 11. Completion criteria

Do not stop after creating placeholders.

The bootstrap is complete only when:

- `pyproject.toml` works
- All 400 mappings validate
- Representative ARC-GEN generators can be imported and sampled
- Generator validation is exercised
- Official scorer adapter passes parity tests
- Candidate execution matches official conversion behavior
- Text visualization works
- Dossier generation works
- ONNX helper toolkit has tests
- Gate judge produces valid structured reports
- Failure packets are useful and schema-valid
- Sealed validation is separate from development tests
- Champion promotion is protected and atomic
- Task workspace preparation works
- Root and task-local `AGENTS.md` exist
- All Codex prompt templates exist
- Codex orchestration commands are implemented but not executed
- README documentation is complete
- Tests, linting, and type checks pass
- No task-specific ONNX solution has been attempted
- Existing ARC-GEN submodule and canonical data remain intact

---

## 12. Required final response

When the implementation is complete, provide a concise final report containing:

1. Major files created or changed
2. Architecture decisions
3. Commands run
4. Test results
5. Mapping validation result
6. Official scorer parity result
7. Known limitations
8. Exact command the user should run next to prepare the first task
9. Exact command the user should run later to launch one task-solving Codex epoch

Do not claim completion for unimplemented or untested features.

Begin by inspecting the repository and writing `docs/bootstrap-plan.md`. Then implement the system phase by phase without launching any task-specific solver.
