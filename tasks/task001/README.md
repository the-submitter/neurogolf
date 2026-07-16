# Task001 (`007bbfb7`, fractal)

## Rule

The input is a 3×3 stencil drawn in one nonzero color. If `M` is its binary
foreground mask, the 9×9 output foreground is the Kronecker square

```text
Y[3 * a + i, 3 * b + j] = M[a, b] * M[i, j].
```

The foreground color is preserved, and every other cell in the logical 9×9
output is background. The remainder of the fixed 30×30 NeuroGolf tensor is
zero in every channel.

## ONNX implementation

`build.py` creates an opset-12, IR-v10 graph with seven nodes:

1. `Slice` extracts the input's 3×3 background plane.
2. `Less(0.5)` converts its zero foreground cells into a nine-element boolean
   mask. A `Cast` converts that mask to float16.
3. A stride-3 `ConvTranspose` uses the stencil as both data and dynamic 3×3
   kernel. Its `-0.5` bias produces positive logits exactly on `M ⊗ M` and
   negative logits on the rest of the logical 9×9 grid.
4. `Einsum` sums each input color plane while applying signs `[-1,+1,…,+1]`.
   Its float16 cast is a dynamic 1×1 color kernel: background is negative and
   the active foreground color is positive.
5. The final `ConvTranspose` multiplies the spatial logits by that color
   kernel. Negative trailing pads create the required 30×30 output directly;
   the extended area is exactly zero under the scorer's strict `> 0` test.

The boolean mask saves nine scored bytes over the earlier float16 affine
encoding while retaining exact behavior for empty, full, and single-cell
stencils as well as the generator's normal 2–8 foreground-cell range.

## Official metric

Using the unmodified scorer in `utils/neurogolf_utils.py` with ONNX Runtime
graph optimizations disabled:

| Component | Cost |
| --- | ---: |
| Intermediate tensors | 285 bytes |
| Initializer parameters | 20 elements |
| Objective (`memory + parameters`) | **305** |
| Points (`25 - ln(305)`) | **19.27968822339259** |
| Serialized model size | 728 bytes |

The intermediate-memory breakdown is 36 bytes for the cropped float32 plane,
9 for the boolean mask, 18 for its float16 cast, 162 for the float16 9×9
spatial logits, 40 for the float32 signed color vector, and 20 for its float16
cast. Parameters are eight slice bounds, two scalar thresholds/biases, and ten
color signs.

## Validation and build

The final model passes all 268 examples in `kaggle_tasks_data/task001.json`.
It was also exhaustively checked on every one of the 512 possible binary 3×3
stencils for each foreground color 1–9 (4,608 cases), with zero failures.

Rebuild both identical artifacts with:

```bash
~/.venv/bin/python tasks/task001/build.py
```

This writes `tasks/task001/task001.onnx` and
`submission/task001.onnx`.
