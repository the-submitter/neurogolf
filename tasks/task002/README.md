# Task 002: honeypots (`00d62c1b`)

The final model implements the `honeypots` rule from
`ARC-GEN/tasks/task_00d62c1b.py`: retain the green pot borders and replace the
black interiors of rectangles from 3x3 through 8x8 with yellow.  The four
rectangle corners are deliberately unconstrained, matching the generator.

## Final result

| Item | Value |
|---|---:|
| Supplied examples | **268 / 268 exact** |
| Memory | **15,464 bytes** |
| Parameters | **3,658** |
| Official cost (`memory + parameters`) | **19,122** |
| Task score (`25 - ln(cost)`) | **15.141405** |
| Serialized size | **4,810 bytes** |
| Opset / IR | **19 / 10** |
| SHA-256 | `ac06f18bf492cbb64a36036af792a739fee81f7c56c1d1fcfd5f88c8b0459c54` |

The 268 checked examples comprise 5 ARC training examples, 1 ARC test example,
and 262 ARC-GEN examples from `kaggle_tasks_data/task002.json`.

## Graph design

1. `Slice` extracts the meaningful 20x20 green channel from the fixed
   `[1, 10, 30, 30]` competition input, and `Cast` quantizes it to `uint8`.
2. The first `QLinearConv` has one signed 8x8 template for each of the 36
   `(height, width)` pairs in `[3, 8] x [3, 8]`.  Required green border cells
   have positive weights summing to 24, interior cells have weight -1, and a
   -23 bias makes a channel positive only when every border cell is green and
   no interior cell is green.
3. A second `QLinearConv` expands each top-left detection over that pot's
   interior.  Its kernel is the minimal 6x6 needed for the largest 8x8 pot.
4. `Cast` converts the fill plane to a Boolean condition.  `Where` broadcasts
   a yellow one-hot vector over detected interiors while retaining the original
   input everywhere else.

The intermediate-memory total is 1,600 bytes for the cropped float plane, 400
for its quantized copy, 11,664 for the 36-channel detector, and 900 bytes each
for the quantized and Boolean fill planes.  The 3,658 counted parameters are
2,304 detector weights, 36 detector biases, 1,296 fill weights, and 22 scalar
or metadata values.

The prior model used a 7x7 fill tensor whose last row and column were always
zero.  Reducing it to 6x6 preserves all offsets while removing 468 parameters;
cost fell from 19,590 to 19,122 and score rose from 15.117225 to 15.141405.

## Rebuild and verification

From the repository root:

```bash
MPLCONFIGDIR=/tmp/mpl-task002 ~/.venv/bin/python tasks/task002/build_task002.py
cp tasks/task002/task002.onnx submission/task002.onnx
```

The builder runs ONNX validation and ONNX Runtime inference over every supplied
example.  It then sanitizes and profiles the model with the local copy of the
official scorer in `utils/neurogolf_utils.py`, with runtime graph optimization
disabled as required by that metric.
