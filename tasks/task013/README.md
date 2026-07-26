# Task013 (`0a938d79`, `columns`)

## Rule

The input contains two nonzero seed cells.  In the generator's canonical
orientation the grid is 20–30 cells wide and 6–12 cells high.  The seeds are
at long-axis positions `p` and `q = p + d`, where `d = sep + 1`; each seed may
independently be on either short-axis edge.  The output fills complete lines
at

```text
p, p + d, p + 2d, p + 3d, ...
```

and alternates the two seed colors.  All other cells are background.  With
`xpose=1`, the same rule operates on rows instead of columns.

The implementation supports any positive in-grid spacing representable by
the fixed 30-cell NeuroGolf canvas, rather than hard-coding the generator's
default `d = 2..6` range.

## Final ONNX graph

`build.py` emits an opset-14 graph with 31 nodes.  Its compact data path is:

1. Two small `Einsum` reductions produce the horizontal and vertical
   foreground-coordinate vectors.
2. First/last `ArgMax` pairs recover the two seed positions on both possible
   axes.  The positions are cast to `INT8`; `Sub`, `Mod`, and boolean tests
   form the forward periodic masks for the first color, second color, and
   background.
3. Those masks are routed into row and column factors.  Only these two
   `2 x 3 x 30` factors are converted to `FLOAT`.
4. A terminal `Einsum` reads the seed colors directly from the original
   input, expands the selected component into complete rows or columns, and
   uses the input again as the exact padding/occupancy mask.

Coordinate 13 is guaranteed to be outside the generator's 6–12 cell short
side and inside its 20–30 cell long side.  Two one-hot probe rows therefore
gate horizontal versus vertical routing inside the terminal contraction,
without a scored orientation tensor.  The terminal equation's operand order
contracts that gate before the spatial expansion; this matters for practical
ONNX Runtime execution time when graph optimizations are disabled.

The graph emits positive logits only for the intended one-hot channel.
Cross-group source/channel terms are negative and all other logits are zero,
which exactly matches the official `(result > 0.0)` decoding.

## Official metric

Measured with the repository's unmodified `utils/neurogolf_utils.py`, after
the official sanitizer and with ONNX Runtime graph optimizations disabled:

| Component | Cost |
| --- | ---: |
| Intermediate tensor memory | **3,014 bytes** |
| Initializer parameters | **128** |
| Objective (`memory + parameters`) | **3,142** |
| Points (`25 - ln(3142)`) | **16.947385181184** |
| Serialized model size | **2,690 bytes** |

The 128 parameters are: foreground/channel selectors (20), coordinate and
modulo constants (33), row/column routing selectors (4), component signs (3),
the 60-value probe-coordinate table, and two 2×2 probe routes (8).

Final model SHA-256:

```text
972e3115245c48b9f836630e57584a0a6d9e1b50412313a6f9ec3158acabace1
```

## Validation and build

The final artifact passed:

- all **267/267** packaged train, test, and ARC-GEN examples;
- **2,000/2,000** fresh default-generator samples;
- a **320/320** structured boundary suite covering both orientations, both
  size extremes, start extremes, every default spacing, and all four seed-edge
  combinations;
- all 267 packaged examples again with ONNX Runtime optimizations enabled;
- `onnx.checker.check_model(..., full_check=True)`, strict ONNX shape
  inference, the official sanitizer, and the profiler-based scorer.

Rebuild and re-verify both artifacts from the repository root with:

```bash
MPLCONFIGDIR=/tmp/mpl-task013 ./.venv/bin/python tasks/task013/build.py
```

This writes byte-identical copies to `tasks/task013/task013.onnx` and
`submission/task013.onnx`.

The task mapping and rule were taken from the repository's ARC-GEN generator
and validator.  Competition context: [overview](https://www.kaggle.com/competitions/neurogolf-2026/overview)
and [data](https://www.kaggle.com/competitions/neurogolf-2026/data).
