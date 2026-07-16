# Task 009 (`06df4c85`)

Task 009 uses the `gridlines` principle. Each logical cell is rendered as a
2×2 pixel block, with a one-pixel gridline after it. Equal-colored nonblack
cells are connected when they share a logical row or column; cells strictly
between the two endpoints receive the endpoint color. Everything else,
including gridlines, isolated cells, and padding, is copied unchanged.

## Model

`build.py` emits one opset-12 `Einsum` node. The single node writes graph
`output` directly, so the official metric charges no intermediate activation
memory.

The contraction has three algebraic branches:

- copy: a dummy lattice coordinate accepts all pixels and passes through the
  local one-hot channel;
- horizontal: row coordinates are equal and column coordinates are strictly
  ordered around the output cell;
- vertical: the same relations are swapped.

An 11×30 shared coordinate table maps the 30 pixel positions to ten possible
logical cells plus the dummy copy coordinate. One shared 2×11×11 table encodes
equality and strict ordering. A 2×2×2 mode core chooses copy, horizontal, or
vertical without duplicating these tables.

Channel routing uses the monomial basis `[1, x, x², x³]`. Copy mode evaluates
`0.5 - (u - z)²`, which is positive only when output channel `u` matches the
local one-hot channel `z`. A connection evaluates
`c * (0.5 - (u - c)²)`: it is zero for black endpoint pairs, positive only on
the endpoint color, and negative on black and all incorrect channels. The
connection polynomial is scaled by `1e9`, safely dominating the positive
carrier used by the copy branch while remaining well within float32 range.

The contraction operands are deliberately interleaved with their coordinate
factors. This lets ONNX Runtime eliminate 30-element axes early instead of
materializing the Cartesian product of all three input occurrences.

## Cost

The dense initializer element count is:

| Initializer | Shape | Parameters |
|---|---:|---:|
| `cell` | 11×30 | 330 |
| `cell_kind` | 2×11 | 22 |
| `relation` | 2×11×11 | 242 |
| `channel_feature` | 4×10 | 40 |
| `channel_behavior` | 3×4×4 | 48 |
| `local_behavior` | 2×3 | 6 |
| `endpoint_behavior` | 2×3 | 6 |
| `mode` | 2×2×2 | 8 |
| **Total** | | **702** |

Official local scorer result:

- intermediate memory: **0 bytes**
- parameters: **702**
- total objective cost: **702**
- score: **18.446067** (`25 - ln(702)`)
- serialized model size: **3,564 bytes**

This improves the previous task-local model from 4,500 parameters and an
estimated 16.588 points.

## Build and verification

From the repository root:

```bash
MPLCONFIGDIR=/tmp/mplconfig ~/.venv/bin/python tasks/task009/build.py --random-cases 1000
cp tasks/task009/task009.onnx submission/task009.onnx
```

The final artifact passes all **265** packaged train/test/ARC-GEN examples and
**1,000** additional deterministic calls to the ARC-GEN `generate()` function.
`build.py` also runs the attached official memory/parameter scorer with graph
optimization disabled.
