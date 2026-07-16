# Task 003 (`017c7c7b`, “beanstalk”)

## Result

The final model is a single opset-12 `Einsum` node. It passes all 265 examples
in `kaggle_tasks_data/task003.json` and an exhaustive enumeration of all 292
parameter combinations admitted by the default ARC-GEN generator.

| Item | Value |
|---|---:|
| Counted intermediate memory | 0 bytes |
| Counted parameters | 140 |
| Official cost | 140 |
| Official task score | 20.058358 |
| ONNX file size | 842 bytes |
| SHA-256 | `e79d4c27de29b8f6bb95f8d5021720260deebee6500e108d5a9800a721ae434b` |

The score is `max(1, 25 - ln(max(1, memory + parameters)))`, as implemented in
`utils/neurogolf_utils.py`.

## Transformation

The ARC-GEN generator creates a 6×3 blue/black pattern with period two or
three, extends that pattern to 9×3, and recolors blue cells to red. For a
two-row period it can also alternate horizontal reflection on each block.

There are only 292 legal default-generator configurations:

- period 2: `2 * C(6, 3) = 40` (unflipped or alternating reflection)
- period 3: `C(9, 4) + C(9, 5) = 252`

The build script enumerates this complete domain independently and verifies the
model against it in addition to the competition examples.

## Model

Let `X[b,i,r,c]` be the one-hot input, `p[i]` the input-color vector, `q[o]`
the output-color vector, and `A[k,r]` a shared rank-four row factor. The node is:

```text
Einsum("birc,i,o,kr,kh->bohc", X, p, q, A, A)
```

This computes:

```text
Y[b,o,h,c] = q[o] * sum(i,r,k, X[b,i,r,c] * p[i] * A[k,r] * A[k,h])
```

`p` maps black/blue to the signed values -1/+1. `q` maps a negative result to
positive black-channel output and a positive result to positive red-channel
output; all other output channels stay at zero. The rank-four Gram transform
`A.T @ A` reproduces the signs of all nine target rows. Columns 9–29 of `A`
are zero, so the padded region has no active output channel.

The row factor was found with the batched search in
`optimize_rank_torch.py`, rounded to small integers, and rechecked exhaustively.
Its minimum signed margin over the complete generator domain is 7, avoiding a
fragile near-zero threshold. Passing the same initializer as both row operands
is important: the scorer counts its 120 values only once.

The parameter accounting is therefore:

```text
input color vector     10
output color vector    10
row factor (4 × 30)   120
                      ---
total                 140
```

The node writes directly to graph output, so the official scorer counts no
intermediate tensor memory.

## Rebuild and verify

From the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib-task003 ~/.venv/bin/python tasks/task003/build.py
```

The script performs full ONNX checking, runs ONNX Runtime with the same
sanitization/profiling setup as the official utility, verifies the supplied and
exhaustive cases, computes the official cost, writes
`tasks/task003/task003.onnx`, and copies the identical file to
`submission/task003.onnx`.

## Explored alternatives

- The original exact grouped `Conv` used an 8×1 kernel and cost 400.
- Exhaustive receptive-field search found no exact dense single-`Conv` kernel
  with area below eight, with or without bias.
- The targeted `ConvTranspose` search found no exact one- or two-tap topology;
  three taps would already cost at least 150 and cannot beat the final model.
- A rank-six shared-factor `Einsum` cost 200; the rank-four factor reduces this
  to 140.
- Sparse convolution weights looked attractive under raw parameter counting,
  but ONNX full shape inference rejects sparse `Conv` weights, so that model is
  not competition-valid.
