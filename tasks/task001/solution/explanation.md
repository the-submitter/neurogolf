# Task 001 / `007bbfb7`

## Semantic rule

The input is a 3×3 binary stencil drawn in one nonzero color. The output is a
3×3 arrangement of 3×3 blocks. An occupied input cell selects a block, and
each selected block contains an exact copy of the input stencil. Empty input
cells select blank blocks, and the foreground color is unchanged.

For foreground mask `M`,

```text
output_mask[3 * block_row + inner_row, 3 * block_col + inner_col]
    = M[block_row, block_col] * M[inner_row, inner_col].
```

Thus the 9×9 foreground is the Kronecker square `M ⊗ M`.

## Optimized ONNX graph

The graph has the required float32 `[1,10,30,30]` input/output, IR version 10,
and standard-domain opset 9. Opset 9 is deliberate: `Slice` coordinates are
attributes in that schema and therefore add no parameter tensors. Both
`ConvTranspose` uses follow the schema available since opset 1 and pass the
full checker and ONNX Runtime.

1. `Slice` reads the input background channel over the 3×3 grid. `Add(-2)`
   encodes foreground as −2 and background as −1.
2. A stride-3 `ConvTranspose` uses that same 3×3 tensor as both data and a
   dynamic kernel. Its products are 4 for foreground×foreground, 2 for a
   mixed pair, and 1 for background×background. Bias −2.5 therefore gives a
   positive logit exactly for the Kronecker foreground and a negative logit
   everywhere else in the 9×9 extent.
3. `ReduceMax` records which input color channels occur. `Mul` negates channel
   0 only, producing a signed dynamic color vector: background is negative and
   the active foreground color is positive.
4. A final 1×1 `ConvTranspose` multiplies the signed spatial logits by that
   color vector directly into `output`. Negative trailing pads extend 9×9 to
   30×30 with zeros, absent under strict `> 0.0` thresholding.

The construction also handles empty, single-cell, and fully occupied stencils
and every foreground color without lookup cases. It has six nodes and just
five scored intermediates: 476 intermediate tensor bytes plus 12 parameter
elements, for objective 488.
