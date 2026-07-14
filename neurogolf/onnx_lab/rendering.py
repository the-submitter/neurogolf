"""Signed-logit output rendering that avoids a final threshold node."""

from __future__ import annotations

import numpy as np
import onnx

from neurogolf.onnx_lab.builder import GraphBuilder
from neurogolf.onnx_lab.conv import conv2d
from neurogolf.onnx_lab.masks import cast_mask


def signed_logits(builder: GraphBuilder, value: str, *, output: str = "output") -> str:
    """Declare already-signed ten-channel logits as the graph output."""

    if value == output:
        builder.add_output(output, onnx.TensorProto.FLOAT, (1, 10, 30, 30))
        return output
    return builder.direct_output(value, dtype=onnx.TensorProto.FLOAT, shape=(1, 10, 30, 30))


def render_color_mask(
    builder: GraphBuilder,
    mask: str,
    foreground: int,
    *,
    background: int = 0,
    output: str | None = None,
) -> str:
    if not 0 <= foreground <= 9 or not 0 <= background <= 9:
        raise ValueError("Foreground/background colors must be 0..9")
    float_mask = cast_mask(builder, mask, shape=(1, 1, 30, 30))
    base = np.full(10, -1.0, dtype=np.float32)
    target = np.full(10, -1.0, dtype=np.float32)
    base[background] = 1.0
    target[foreground] = 1.0
    weights = (target - base).reshape(10, 1, 1, 1)
    return conv2d(builder, float_mask, weights, bias=base, output=output)
