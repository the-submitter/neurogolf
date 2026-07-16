# Task 010 — bar-height ranking

## Result

The final model is [`task010.onnx`](task010.onnx). The identical submission
artifact is at [`../../submission/task010.onnx`](../../submission/task010.onnx).

| Measurement | Result |
|---|---:|
| Official intermediate memory | 128 bytes |
| Official parameter count | 207 |
| Official cost (`memory + parameters`) | **335** |
| Estimated official task score | **19.185869** |
| ONNX file size | 1,671 bytes |
| Packaged examples | 265 / 265 |
| Exhaustive generator cases | 3,024 / 3,024 |
| SHA-256 | `c6d34f361fb32569c38319ce0b4da59124a4050d18c0c3e245a0b9678c2c56f2` |

The score is computed exactly as in `utils/neurogolf_utils.py`:
`max(1, 25 - log(memory + parameters))`.

## Transformation

Task 010 maps to ARC-GEN task `08ed6ac7`, principle `barchart`. Every 9x9
input contains four gray vertical bars at columns 1, 3, 5, and 7. Their heights
are distinct values from 1 through 9. The output recolors the tallest bar 1,
the next tallest 2, then 3, and the shortest 4. Black cells remain black.

## Graph design

The graph uses opset 12 and only four nodes:

1. **Einsum** reads the four heights. A 4x30 spatial table is an identity at
   the four possible bar columns and a partition of unity elsewhere.
2. **TopK** sorts the four heights in descending order.
3. **Gather** reorders the four rows of a 4x4 spatial identity using the TopK
   permutation.
4. **Einsum** applies the dynamic routing and emits the full
   `[1, 10, 30, 30]` output directly.

The final channel decision uses the sign code

```text
score(o) = 0.5 - (o - g*q)^2
```

where `o` is the output channel, `q` is the bar's rank color (1–4), and `g` is
1 for gray input and 0 for black input. This is positive at exactly the desired
channel and non-positive everywhere else. It expands into three separable
terms, so the last Einsum needs only small input/rank/output factors rather
than a dense dynamic routing tensor. The four spatial selectors sum to one,
which preserves black without a fifth background route.

The official 128-byte memory total is the sum of the four profiled
intermediates:

| Tensor | Shape/type | Bytes |
|---|---|---:|
| `rank_input` | `[4]` float32 | 16 |
| `sorted_heights` | `[4]` float32 | 16 |
| `rank_indices` | `[4]` int64 | 32 |
| `ranked_spatial_left` | `[4,4]` float32 | 64 |

Inputs, outputs, and initializers are excluded from intermediate memory by the
official metric. The eight initializers contain 207 elements in total.

## Correctness argument

At a bar column, the spatial table has one value of 1 and three zeros. The
height-reading Einsum therefore returns the exact four gray pixel counts.
Because generator heights are distinct, TopK produces a unique permutation.
Gather applies that permutation to the same selectors, so a gray pixel sees
exactly its rank `q`. At every valid cell, the selectors sum to one; a black
pixel therefore sees `g=0` and gets a positive score only on channel 0. Cells
outside the 9x9 grid have no active input channel and remain empty.

In addition to all provided examples, the builder checks every generator case:
`C(9,4) * 4! = 3,024` distinct height-set/order combinations.

## Rebuild and verify

From the repository root:

```bash
MPLCONFIGDIR=/home/rohit-raje/.cache/matplotlib \
  ~/.venv/bin/python tasks/task010/build_task010.py
```

This rebuilds the model, runs packaged and exhaustive checks, profiles it with
the official scorer, and copies it into `submission/`. The verified environment
used ONNX 1.21.0, ONNX Runtime 1.24.4, and NumPy 2.4.4.
