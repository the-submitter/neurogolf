# Task001 (`007bbfb7`, fractal)

## Rule

The input is a 3×3 stencil in one nonzero color. If `M` is its binary
foreground mask, the 9×9 output is its Kronecker square:

```text
Y[3 * a + p, 3 * b + q] = M[a, b] * M[p, q].
```

The foreground color is preserved, zeros are background, and the unused part
of the fixed 30×30 NeuroGolf tensor contains no active channel.

## Final ONNX graph

`build.py` emits an opset-12 model containing one terminal `Einsum`. There are
no scored intermediate tensors.

The graph uses a homogeneous coordinate row
`[1, floor(r/3), r mod 3]` for each logical output coordinate. Two small
quadratic factors turn either base-3 digit into exact Lagrange indicators for
coordinates 0, 1, and 2. Sharing the role index between the row and column
relations selects the two outer digits together or the two inner digits
together. This gives the signed spatial test

```text
signed_stencil[outer_row, outer_col]
+ signed_stencil[inner_row, inner_col].
```

Background has sign `+2` and the foreground colors have sign `-1`. Therefore
the spatial test is negative exactly when both selected stencil cells are
foreground. Reusing the same sign vector on the output channel makes the
foreground channel positive in that case and makes background positive in all
other cases. Inactive foreground channels are multiplied by a zero channel
count. The zero coordinate rows 9–29 suppress the padded output tail.

## Official metric

Measured with the unmodified `utils/neurogolf_utils.py`, after its sanitizer,
with ONNX Runtime graph optimizations disabled:

| Component | Cost |
| --- | ---: |
| Intermediate tensors | **0 bytes** |
| Initializer parameters | **129** |
| Objective (`memory + parameters`) | **129** |
| Points (`25 - ln(129)`) | **20.140187595638327** |
| Serialized model size | **1,097 bytes** |

The 129 parameters are a 10-value channel-sign vector, a 30×3 coordinate
table, a 2×3×3 digit basis, a 3×3 Lagrange decoder, and a two-value inner-role
selector. Compared with the previous model, the objective falls from 305 to
129 and the score rises from 19.27968822339259 by about 0.86050 points.

## Validation and build

The final model passed all 268 examples in
`kaggle_tasks_data/task001.json` through ONNX Runtime with graph optimizations
disabled. It also passes `onnx.checker.check_model(..., full_check=True)`, the
official sanitizer, and the official profiler-based scorer.

Rebuild both byte-identical artifacts with:

```bash
~/.venv/bin/python tasks/task001/build.py
```

This writes `tasks/task001/task001.onnx` and
`submission/task001.onnx`.
