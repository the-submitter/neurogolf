# Task 011 (`09629e4f`, principle: `four`)

The final model is `task011.onnx`; the identical submission copy is
`../../submission/task011.onnx`.

## Result

| Measurement | Result |
|---|---:|
| Official intermediate memory | 64 bytes |
| Official parameter count | 250 |
| Official cost (`memory + parameters`) | **314** |
| Estimated official task score | **19.250607014092** |
| Serialized model size | 1,451 bytes |
| Packaged examples | 267 / 267 |
| Fresh seeded generator cases | 2,000 / 2,000 |
| SHA-256 | `3a7ab87dc2e1d503de3eca5fdf304a7db422537d10ec681d0c8f64de4cf4cdfd` |

The score is the competition formula from `utils/neurogolf_utils.py`:
`max(1, 25 - ln(memory + parameters))`.

## Transformation

The 11x11 grid is a 3x3 array of 3x3 mini-grids separated by gray lines.
Exactly one mini-grid contains four colored cells; every other mini-grid
contains five. The output takes the four-cell mini-grid and transposes its
two coordinate levels: a source cell at local coordinate `(p, q)` becomes a
solid 3x3 output block at outer coordinate `(p, q)`. Its color is preserved,
black local cells become black blocks, and the gray separators remain gray.

## ONNX graph

`build.py` emits an opset-12 model with two `Einsum` nodes and three dense
initializers:

1. A `[30,4]` outer-coordinate table, a `[30,4]` inner-coordinate table, and
   a ten-element channel vector reduce the input to a `[1,4,4]` selector.
   For ordinary coordinates, the channel weights score a four-color block as
   `5*5 - 4*4 = 9` and a five-color block as `4*5 - 5*4 = 0`. The fourth
   coordinate represents the two gray separator positions and deliberately
   receives a positive score.
2. The terminal `Einsum` reuses the two coordinate tables six times. It routes
   the selected source block by matching source outer coordinates to selector
   coordinates, and source inner coordinates to output outer coordinates.
   The fourth coordinate propagates the gray rows and columns. Zero table rows
   after coordinate ten keep the 30x30 padding empty.

The first `Einsum` is the only charged intermediate: 16 float32 values, or 64
bytes. The graph output is excluded from the memory count. The two coordinate
tables contain 120 elements each and the channel vector contains ten, giving
250 parameters. Integer weights provide exact zeros; across all validation,
true logits are at least 9 and false logits are at most 0.

A one-node fused prototype cost 378 parameters and was numerically fragile
when fractional block weights rounded an intended zero slightly positive. A
sparse-initializer prototype would reduce stored coefficients, but ONNX full
shape inference rejects sparse tensors as `Einsum` operands, so it is not legal
under the official scorer.

## Rebuild and verification

From the repository root:

```bash
MPLCONFIGDIR=/home/rohit-raje/.cache/matplotlib \
  ./.venv/bin/python tasks/task011/build.py
```

The builder runs ONNX full checking, all packaged train/test/ARC-GEN cases,
2,000 deterministic fresh calls to the generator (covering all nine possible
selected blocks), and the official sanitizer, profiler, memory counter, and
parameter counter. It then writes both the workspace and submission models.
