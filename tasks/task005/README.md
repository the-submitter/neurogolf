# Task 005 (`045e512c`)

Task 005 uses the ARC-GEN `stamp` principle. The input contains one complete
3×3 monochrome sprite near the center and two or three partial copies four
cells away. Each partial copy identifies a direction and color. The output
completes that copy and repeats the complete sprite at four-cell intervals
along its ray, clipped to the 21×21 task grid.

## Model

`build_model.py` builds the final opset-16 ONNX graph. Its compact pipeline is:

1. Slice the only possible 9×9 anchor region from input channel zero. A 3×3
   negative convolution finds the complete sprite as the unique minimum of
   background population, and `MaxPool` supplies its flattened position.
2. Use `GridSample` to read the nine central sprite cells and a 16-point
   direction-marker hitting set. The generator guarantees one sampled cell
   for every cardinal marker and at least one of two sampled cells for every
   diagonal marker. Including the central sprite, only 25 input points are
   sampled instead of all 81 cells in nine 3×3 windows.
3. Encode sampled foreground color `c` as the exact scalar `100+c`. Small
   reductions recover the central and direction color codes; int8 outer
   products combine each code with the nine-cell sprite mask.
4. Render the center and three potential steps for all eight directions with
   `ScatterElements`. Spatial coordinates are flattened, halving index memory
   relative to two-component `ScatterND` indices. The int8 canvas uses 100 for
   valid background and 120 for padding, so `reduction="max"` safely absorbs
   absent and wrapped off-grid updates.
5. A broadcast `Equal` against codes 100 through 109 directly emits the final
   boolean one-hot tensor. The official runner thresholds output at `> 0`, so
   boolean output is equivalent to float one-hot output and costs no extra
   conversion tensor.

Build, verify, stress-test, and copy the submission artifact with:

```bash
MPLCONFIGDIR=/tmp/matplotlib ~/.venv/bin/python \
  tasks/task005/build_model.py --random-cases 10000
```

Use `--no-copy` to leave `submission/task005.onnx` unchanged. The default
output is `tasks/task005/task005.onnx`.

## Verification and official cost

The final model passes:

- 4/4 ARC-AGI train/test examples;
- 262/262 supplied ARC-GEN examples;
- 10,000/10,000 additional fresh calls to the checked-in `generate()`.

Using the checked-in official scorer with ONNX Runtime graph optimization
disabled:

- intermediate tensor memory: **6,246 bytes**;
- parameters: **1,250 elements**;
- objective cost (`memory + parameters`): **7,496**;
- task score (`25 - ln(cost)`): **16.077875**.

The ONNX model is deliberately small in activation width: its largest dynamic
tensor is the 1,800-byte int64 flattened scatter-index tensor. The 900-cell
int8 canvas is an initializer, so it contributes parameter count rather than
intermediate memory.
