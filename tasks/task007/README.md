# Task 007: `diagstripes` (`05269061`)

## Transformation

The ARC-GEN generator assigns each nonzero color to one diagonal residue class:

```text
class(row, column) = (row + column) mod 3
```

The input reveals parts of three diagonal classes. The output fills the entire
7x7 grid with the color associated with each class. Rows and columns 7 through
29 are padding and must remain empty in the scorer's 30x30 representation.

## ONNX implementation

The final model contains one opset-12 `Einsum` node:

```text
nkij,k,ia,jb,ru,cv,abq,uvq->nkrc
```

`F` assigns the three coordinate residues the exact two-dimensional vectors
`(1,0)`, `(0,1)`, and `(-1,-1)`. Equal-class dot products are positive, while
unequal-class dot products are zero or negative. This directly matches the
official scorer's final `output > 0` threshold and avoids a comparison node.
Rows 7 through 29 of `F` are zero, which also masks output padding.

`M` implements bilinear multiplication in `R[x]/(x^2+x+1)`. Given the vectors
for coordinates `a` and `b`, it produces the vector for `(a+b) mod 3`. The same
2x2x2 tensor is reused for the source and destination coordinate pairs. `D`
is a ten-element channel mask that suppresses input background channel zero.

The initializers contain:

- `D`: 10 parameters
- `F`: 30 x 2 = 60 parameters
- `M`: 2 x 2 x 2 = 8 parameters
- Total: 78 parameters

Because the graph has only one node and that node writes directly to `output`,
the official memory calculation finds no scored intermediate activations.

## Verified result

Using the scorer in `utils/neurogolf_utils.py` with graph optimization disabled:

```text
Examples passed: 266 / 266
Generated fuzz:  1000 / 1000
Memory:          0 bytes
Parameters:      78
Official cost:   78
Official score:  25 - ln(78) = 20.6432911733
Model size:      675 bytes
SHA-256:         07e5c939d4980959e9f29fdc0b1c33a93ccd2a9c6c46e44dc5e05e208341e81f
```

This cost is below the stated leaderboard-average budget of about 80.93.

A sparse-initializer version would store only 23 nonzero values, but it is not
submission-valid: the official `onnx.checker.check_model(..., full_check=True)`
path treats sparse `Einsum` initializers as rankless and rejects the equation
during strict shape inference. The final artifact therefore uses dense,
officially accepted initializers.

## Build and test

From the repository root:

```bash
~/.venv/bin/python tasks/task007/build_model.py
```

The script builds the model, checks all train, test, and ARC-GEN examples in
`kaggle_tasks_data/task007.json`, and copies the verified artifact to
`submission/task007.onnx`.
