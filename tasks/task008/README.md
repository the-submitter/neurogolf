# Task 008 — ARC 05f2a901 (`magnets`)

## Result

The final model is [`task008.onnx`](task008.onnx), built by
[`build_task008.py`](build_task008.py). The identical submission artifact is
`submission/task008.onnx`.

| Metric | Value |
|---|---:|
| Bundled examples | 266 / 266 passed |
| Additional random ARC-GEN cases | 5,000 / 5,000 passed |
| Official intermediate memory | 4,348 bytes |
| Official parameters | 214 |
| Official total cost | 4,562 |
| Estimated task score | 16.574484 |
| Serialized model size | 4,492 bytes |
| Nodes / initializers | 44 / 12 |
| ONNX opset | 16 |
| SHA-256 | `192968cadff1e4f5fc99b2cf2bd299264dfeb46a2f1994bf21a0364976c91b26` |

The estimated score is calculated with the official formula
`25 - ln(memory + parameters)`.

## Transformation

The input contains an irregular red rectangle and a solid 2x2 cyan magnet.
The red object moves along the one axis separating the objects until its
bounding box touches the cyan block. Its internal holes and orientation are
preserved; the cyan block does not move.

ARC-GEN creates the red object from a 2x4, 2x5, 3x4, or 3x5 rectangle and may
transpose and/or vertically flip the scene. Nibbles can remove boundary cells,
but validation explicitly rejects a completely removed row or column. The
model uses this guarantee to recover the bounding box exactly.

## Graph design

The graph operates on NeuroGolf's fixed `[1, 10, 30, 30]` one-hot tensor.

1. Two `Einsum` nodes project red occupancy onto rows and columns. `Sign`
   converts each projection to a contiguous interval, and `ArgMax` with first
   and last tie selection gives the four bounding-box coordinates.
2. Two more `Einsum` contractions locate the cyan block. A 30-element affine
   coordinate vector makes the contraction of a 2x2 block equal its top/left
   coordinate directly, so no spatial cyan mask is materialized.
3. Scalar comparisons and arithmetic compute the signed row or column shift
   that makes the red bounding box adjacent to cyan.
4. A fixed 3x5 candidate lattice covers every possible generator rectangle.
   The row/column offsets swap for a horizontal move. Four scalar origins are
   broadcast over this lattice to make source-black, source-red,
   destination-black, and destination-red index groups without materializing
   separate source and destination coordinate grids.
5. `GatherND` reads the source red mask. A small `Einsum` produces additive
   deltas `[+red, -red, -red, +red]`, and one `ScatterND(reduction="add")`
   applies all black/red source and destination edits directly to the input.
   Candidates that correspond to holes or an unused rectangle row/column have
   a zero source-red value and therefore make no change.

The dominant irreducible tensor is the int64 `ScatterND` index array: 60 edits
times four coordinates = 1,920 bytes. Broadcasting grouped scalar origins
instead of building four full coordinate grids reduced the pre-existing
candidate from a cost of 5,519 to 4,562 (17.3%).

## Rebuild and verify

From the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib ~/.venv/bin/python tasks/task008/build_task008.py
```

The script rebuilds the model, checks every train/test/ARC-GEN case in
`kaggle_tasks_data/task008.json`, measures it with
`utils/neurogolf_utils.py`, and copies the verified artifact into
`submission/task008.onnx`. Use `--no-copy` to leave the submission file
untouched.

