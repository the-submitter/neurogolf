# Task004 (`025d127b`, principle: `tilt`)

The ARC-GEN generator draws one or more colored parallelogram outlines.  The
output tilts each outline by moving every non-bottom row one cell right while
leaving the bottom edge fixed.  There is one extra corner case: the right pixel
of the penultimate row is already at the bottom-right extent and must remain in
place rather than move one cell beyond it.

## Model

`build.py` creates an opset-12 graph with three nodes:

1. An `Einsum` reduces nonblack pixels to one colored-cell count per row.
2. A two-tap 1-D `Conv` reads the following row count `N` and emits the two row
   coefficients `[N, 1-N]`.  They select shifted rows when `N >= 2` and unchanged
   rows when `N = 0`, while overlapping pixels retain coefficient one.
3. A final `Einsum` writes the `[1,10,30,30]` output directly.  It reuses one
   dense identity/successor relation for both horizontal movement and
   directly-below tests.  The polynomial `N*(N-2)` distinguishes the
   penultimate right endpoint from the geometrically similar top-left endpoint,
   adds that pixel at its unchanged location, and removes its shifted copy.
   A third input factor is the logical-grid validity mask, preventing values
   from leaking into 30x30 padding.  The color basis reconstructs black as the
   valid mask minus transformed colored occupancy.

The high-arity final `Einsum` is deliberate: because it writes graph output
directly, the scorer charges no full-grid intermediate activation.  All
initializers are dense.  ONNX 1.21 full shape inference rejects sparse tensor
initializers as `Einsum` inputs even though ONNX Runtime can execute them, so a
sparse version does not pass the official scorer's checker.

## Official metric

Measured locally with the pinned competition packages and
`utils/neurogolf_utils.py`:

- Intermediate memory: **360 bytes**
- Parameters: **2,048**
- Objective cost: **2,408**
- Task score: **17.213448** (`25 - ln(2408)`)
- Serialized model size: **8,876 bytes**

The two charged tensors are only `[1,1,30]` and `[1,2,30]` float32 arrays.  The
large fixed relation tables affect parameter count but do not create charged
activations.

## Validation and build

The final model passed all **265/265** packaged train, test, and ARC-GEN cases
through ONNX Runtime with graph optimizations disabled.  The equivalent final
Einsum algebra also passed **20,000/20,000** additional seeded calls to the
ARC-GEN `generate()` function.  ONNX full checking and the official sanitizer,
profiler, memory counter, and parameter counter all pass.

Build, verify, score, and copy both artifacts with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache ~/.venv/bin/python tasks/task004/build.py
```

The command writes `tasks/task004/task004.onnx` and
`submission/task004.onnx`.
