# Task 015 (`0ca9ddb6`, principle: `twinkle`)

The model artifact is `task015.onnx`; the byte-identical submission copy is
`../../submission/task015.onnx`.

## Transformation

The input is a 9x9 field containing one to six isolated stars in colors 1, 2,
6, and 8. The output preserves every star and makes two colors “twinkle”:

- each color-1 star gains color-7 cells at its four orthogonal neighbors;
- each color-2 star gains color-4 cells at its four diagonal neighbors;
- colors 6 and 8 remain unchanged.

The generator moves twinkling stars away from the boundary and removes stars
whose sprites could overlap, so every added arm is in bounds and replaces a
black cell.

## ONNX graph

The opset-12 graph consists of one factored `Einsum` node. Five shared
initializers describe the complete transform:

1. A 30x9 coordinate table extracts the generator's 9x9 grid from the fixed
   30x30 NeuroGolf canvas and embeds the result back into that canvas.
2. Two 9x9 spatial operators represent identity and a symmetric one-cell
   shift. Their products produce the center, rook-neighbor, and
   bishop-neighbor patterns.
3. Seven rank-one channel terms preserve the five input channels used by the
   generator (`0`, `1`, `2`, `6`, and `8`), route color 1 to color 7 along
   orthogonal shifts, and route color 2 to color 4 along diagonal shifts.
4. The two twinkle terms also subtract the corresponding logits from channel
   0. This suppresses background exactly where a new arm is painted.

The contraction writes directly to `output`. Positive logits therefore form
the intended one-hot output under the official `result > 0.0` decoder.

## Artifact and expected metric

| Measurement | Result |
|---|---:|
| Expected official intermediate memory | **0 bytes** |
| Official parameter count | **600** |
| Expected cost (`memory + parameters`) | **600** |
| Expected task score | **18.603070344784** |
| Serialized model size | 2,815 bytes |
| Packaged corpus size | 265 examples |
| Structured single-star suite in `build.py` | 260 cases |
| SHA-256 | `b22cefea4dfffa98c365b2ce8fa86897e1dbff9a1e275c5d55c147270c21be1f` |

The parameter count follows directly from the initializer shapes:

| Initializer role | Shape | Parameters |
|---|---:|---:|
| Shared coordinate map | 30x9 | 270 |
| Output-channel factors | 10x7 | 70 |
| Input-channel factors | 10x7 | 70 |
| Spatial-term routing | 7x2x2 | 28 |
| Two 9x9 spatial operators | 2x9x9 | 162 |
| **Total** |  | **600** |

Because the only node writes to the terminal graph output, the repository's
official scorer has no graph intermediate to charge. The score uses
`max(1, 25 - ln(memory + parameters))` from
`utils/neurogolf_utils.py`.

## Validation and rebuild status

`build.py` deterministically constructs the model, applies full ONNX checking
and strict shape inference, installs the submission copy, and contains checks
for:

- all 3 train, 1 test, and 261 ARC-GEN examples in `task015.json`;
- all 260 valid single-star placements across colors 1, 2, 6, and 8;
- official sanitization, profiling, memory counting, and parameter counting;
- byte identity between the task and submission artifacts.

The two checked-in ONNX files are currently byte-identical, and their graph
structure, initializer count, size, and digest were inspected statically.
However, the full builder was **not safely rerun** while preparing this
README: executing the current high-arity `Einsum` through ONNX Runtime caused
severe resource exhaustion in the development host. Treat the score and
zero-memory figure above as structurally derived expectations until runtime
verification is repeated in a resource-isolated process. Do not run
`tasks/task015/build.py` in the editor host until that runtime behavior has
been isolated or the graph has been replaced with a safer equivalent.
