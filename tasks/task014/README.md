# Task 014 (`0b148d64`, principle: `minstatic`)

The final model is `task014.onnx`; the byte-identical submission copy is
`../../submission/task014.onnx`.

## Result

| Measurement | Result |
|---|---:|
| Official intermediate memory | **2,456 bytes** |
| Official parameter count | **40** |
| Official cost (`memory + parameters`) | **2,496** |
| Estimated official task score | **17.177555270511** |
| Serialized model size | 2,743 bytes |
| Packaged examples | 266 / 266 |
| Fresh ARC-GEN samples | 2,000 / 2,000 |
| SHA-256 | `c80a5feed4b91a61be2e4b39c0904125a67f2cd85ce59ab51b2fb49a76c91e7f` |

The score uses the competition formula from `utils/neurogolf_utils.py`:
`max(1, 25 - ln(memory + parameters))`.

## Transformation

The input has two nonzero colors distributed among four rectangular regions
separated by black rows and columns. One color fills one region while the
other fills the remaining three. The output is the least frequent nonzero
color, cropped to its exact occupied bounding box; black holes inside that
box are retained.

The implementation follows the generator's general rule rather than assuming
that the selected region touches all four of its nominal quadrant edges.

## ONNX graph

The opset-12 graph performs these operations:

1. Spatially reduce the ten input channels to color counts. Channel zero and
   absent colors are made ineligible, then `ArgMin` selects the rare nonzero
   color.
2. Convert the rare index to a ten-element selector. Two `Einsum` nodes use
   that selector to calculate only the 30 row counts and 30 column counts for
   the rare color. Boolean occupancy is cast to `uint8`; first/last `ArgMax`
   operations recover the exact bounding box.
3. A dynamic `Slice` extracts the selected channel and its bounding box.
   `QuantizeLinear` maps black and rare pixels from floating-point `0, 1` to
   unsigned-byte `0, 2`. `Pad` fills the unused part of the fixed 30x30 tensor
   with the middle value `1`.
4. A terminal 1x1 `QLinearConv` uses input zero-point 1, so the three spatial
   states become `-1, 0, +1`. Its dynamic weight is `-1` for output channel
   zero, `+1` for the rare output channel, and zero elsewhere. Unsigned output
   saturation therefore emits a positive value only for crop-background in
   channel zero and rare pixels in their original color; the padded tail and
   inactive channels remain zero.

The terminal output is `uint8`, which is valid because the official runner
classifies every output cell with `result > 0.0`.

## Official memory breakdown

The scorer profiles every intermediate with graph optimizations disabled and
uses its maximum observed runtime shape. The largest packaged crop has area
168.

| Intermediate group | Bytes |
|---|---:|
| Rare-color counting and selector | 138 |
| Exact row/column bounds | 392 |
| Slice bounds | 72 |
| Float crop plus quantized crop | 840 |
| Crop dimensions and pad vector | 96 |
| Padded 30x30 `uint8` crop | 900 |
| Dynamic quantized convolution weight | 18 |
| **Total** | **2,456** |

The 40 parameters are small scalar/vector control constants, the ten-element
background weight, the rare-channel update, and quantization scales and zero
points. The graph has no dense spatial initializer.

## Validation and rebuild

`build.py` performs full ONNX validation, runs every train/test/ARC-GEN example
from `kaggle_tasks_data/task014.json`, checks 2,000 deterministic fresh calls
to `ARC-GEN/tasks/task_0b148d64.py:generate()`, applies the official sanitizer,
profiles with ONNX Runtime optimizations disabled, calculates the official
memory and parameter costs, and installs a byte-identical submission copy.

From the repository root:

```bash
MPLCONFIGDIR=/home/rohit-raje/.cache/matplotlib \
  ./.venv/bin/python tasks/task014/build.py
```

This writes `tasks/task014/task014.onnx` and
`submission/task014.onnx`.
