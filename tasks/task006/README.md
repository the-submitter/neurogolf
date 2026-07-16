# Task 006

Task 006 maps to ARC-GEN task `0520fde7` and the `intersect` principle.  Its
input contains two complete 3×3 black/blue panels separated by one gray
column.  An output cell is red only when both aligned input cells are blue;
all other cells in the 3×3 output are black.

## Model

`task006.onnx` is a single grouped `Conv` in opset 10.  Its two horizontal
kernel taps have dilation 4, so output column `c` reads input columns `c` and
`c+4`.  `group=2` is the largest valid group count that still lets output
channels 0 (black) and 2 (red) read input channels 0 and 1.

For left/right black and blue indicators `Lb`, `Lu`, `Rb`, and `Ru`, the two
active output channels are:

```text
black = -Lu + 2*Rb + Ru
red   = -Lb + Ru
```

The official runner thresholds at `> 0`.  These expressions give:

| aligned colors | black value | red value | emitted color |
|---|---:|---:|---|
| black / black | 2 | -1 | black |
| black / blue | 1 | 0 | black |
| blue / black | 1 | 0 | black |
| blue / blue | 0 | 1 | red |

They also suppress the unpaired right panel at output columns 4–6 without a
bias: an isolated black gives `(0, -1)` and an isolated blue gives `(-1, 0)`.
Only five kernel entries are nonzero, although ONNX `Conv` requires the dense
logical weight shape `[10, 5, 1, 2]`.  A sparse initializer was considered,
but ONNX 1.21 full shape inference—the validation path called by the official
scorer—rejects sparse tensors as direct `Conv` weights.  The valid dense tensor
therefore costs 100 parameters.  A second node that reduced the logical kernel
would introduce at least one full-grid intermediate and cost much more in
scored memory.

The grouping is also intentional.  `group=5` or `group=10` would use fewer
weights, but output channel 2 would then be disconnected from input channels 0
and 1, so it could not detect a blue/blue pair.  `group=2` is the largest group
count that keeps both required output channels connected to both binary input
channels.  With ten output channels, five visible input channels per group,
and two spatial taps, its dense parameter count is necessarily
`10 * 5 * 1 * 2 = 100`.

An archived Boolean `Slice`/`Cast`/`And`/`Not` design claimed cost 18 based on
counting only two Boolean tensors.  Rechecking that design with the attached
May 14 scorer counts all statically shaped intermediates and `Constant`
payloads: it costs 336 bytes + 33 parameters = 369.  It is exact but not an
optimization over this convolution.  A five-nonzero sparse convolution runs
in ONNX Runtime, but `onnx.checker.check_model(..., full_check=True)` rejects
the sparse weight as an unsupported `sparse_tensor(float)` Conv input, so it
is not submission-valid.

## Build and verification

Run with the pinned project environment:

```bash
~/.venv/bin/python tasks/task006/build_task006.py
```

This rebuilds the model, checks every supplied example plus all four local
black/blue truth-table cases at every aligned cell, checks 10,000 fresh
default-parameter calls to the authoritative ARC-GEN generator, measures it through
`sanitize_model()` and `score_network()` from the official utility, and copies
the result to `submission/task006.onnx`.

Final local results:

- Exact validation: 266/266 examples (3 train, 1 test, 262 ARC-GEN)
- Synthetic truth table: 4/4 cases, with each case repeated over all 9 cells
- Fresh seeded ARC-GEN validation: 10,000/10,000
- Intermediate tensor memory: 0 bytes
- Parameters: 100
- Total official cost: 100
- Estimated official task score: `25 - ln(100) = 20.394830`
- ONNX file size: 648 bytes
- Graph: one `Conv` node, IR version 10, opset 10

This improves the previous bias-based convolution from cost 110 and score
20.299520 while preserving the same exact transformation.
