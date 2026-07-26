# Task 012 (`0962bcdd`, principle: `supernova`)

The final model is `task012.onnx`; the identical submission copy is
`../../submission/task012.onnx`.

## Result

| Measurement | Result |
|---|---:|
| Official intermediate memory | **0 bytes** |
| Official parameter count | **730** |
| Official cost (`memory + parameters`) | **730** |
| Estimated official task score | **18.406955465858** |
| Serialized model size | 3,151 bytes |
| Packaged examples | 265 / 265 |
| Exhaustive structural generator cases | 392 / 392 |
| True-logit minimum | 0.029999733 |
| False-logit maximum | -0.0299999714 |
| SHA-256 | `4e2d8df579940ee5520aae1f0f8073e118d9aefdbcf454786555d7aac7d942a5` |

The score uses the competition formula from `utils/neurogolf_utils.py`:
`max(1, 25 - ln(memory + parameters))`.

## Transformation

Every input contains two five-cell crosses. The center has the first color;
its four orthogonal neighbors have the second color. Each cross becomes a
5x5 eight-ray “supernova”:

- the first color occupies the center and diagonal distances one and two;
- the second color occupies the four axial distances one and two;
- all other cells remain black.

The generator initially puts the two centers six rows apart, permits every
column from 3 through 9, and then applies one of four gravity transforms.

## ONNX graph

The model is one opset-12 depthwise `Conv` (`group=10`) with an 8x9 kernel,
pads `[3, 4, 4, 4]`, and a ten-element bias. Its dense weight tensor has shape
`[10, 1, 8, 9]`:

- channel 0 has a background kernel that removes black logits exactly where
  the expanded rays appear;
- channels 1 through 9 reuse the same foreground kernel numerically, making
  the rule color-equivariant;
- every channel uses bias `-1`.

An 8x9 same-channel neighborhood is sufficient because it sees enough of the
two fixed-distance sprites to distinguish the isolated center-color pair from
the two four-arm color patterns. The builder forms every unique neighborhood
in the supplied corpus and solves two linear feasibility problems. With logit
`patch * weight - 1`, true and false constraints are separated from zero by a
requested margin of 0.03.

The terminal convolution writes directly to `output`, which the official
memory counter excludes. There are no intermediate tensors, so memory cost is
zero. Parameters are `10 * 1 * 8 * 9 = 720` weights plus 10 biases, for a total
cost of 730.

Contiguous receptive fields with lower parameter area were tested and were
not linearly separable for the foreground rule, including 7x9, 9x7, every
centered 8x8 alignment, and centered 7x10/10x7 variants. A full 5x5
cross-channel convolution works but costs 2,500 parameters. Algorithmic
multi-node candidates used fewer parameters but paid more than 2 KB of
intermediate memory. The one-node 8x9 depthwise graph therefore had the best
measured official cost among the tested legal candidates.

## Validation coverage

`build.py` checks:

1. ONNX full model validation and shape inference through the official scorer;
2. all 2 train, 1 test, and 262 ARC-GEN examples in `task012.json`;
3. all `7 * 7 * 4 = 196` column-pair/gravity structures twice, swapping two
   representative foreground colors to cover both color roles;
4. official sanitization, ONNX Runtime profiling with optimizations disabled,
   memory counting, and parameter counting;
5. byte-identical installation into `submission/task012.onnx`.

Because the same foreground kernel is repeated for all nine nonblack
channels, the two representative colors cover the complete structural
generator space; changing their numeric color IDs cannot change a logit.

## Rebuild

From the repository root:

```bash
MPLCONFIGDIR=/home/rohit-raje/.cache/matplotlib \
  ./.venv/bin/python tasks/task012/build.py
```

The builder requires NumPy, SciPy, ONNX, and ONNX Runtime from the
repository-local virtual environment. It deterministically refits the two
kernels, validates the result, writes `tasks/task012/task012.onnx`, and copies
the same bytes to `submission/task012.onnx`.
