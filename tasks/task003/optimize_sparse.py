#!/usr/bin/env python3
"""Find minimum-support weights for the task003 causal Conv topology."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def legal_examples() -> list[tuple[np.ndarray, np.ndarray]]:
    """Enumerate every parameterization admitted by the ARC-GEN generator."""

    examples = []
    for steps, count, flip in [(2, 3, 0), (2, 3, 1), (3, 4, 0), (3, 5, 0)]:
        for pixels in itertools.combinations(range(3 * steps), count):
            source = np.zeros((6, 3), dtype=np.int8)
            target = np.zeros((9, 3), dtype=np.int8)
            flipped = 0
            for offset in range(0, 9, steps):
                for pixel in pixels:
                    row, col = divmod(pixel, 3)
                    actual_col = col if not flipped else 2 - col
                    if offset + row < 6:
                        source[offset + row, actual_col] = 1
                    if offset + row < 9:
                        target[offset + row, actual_col] = 1
                if flip:
                    flipped = 1 - flipped
            examples.append((source, target))
    return examples


def constraints(examples, output_channel: int) -> tuple[np.ndarray, np.ndarray]:
    """Return unique causal height-eight patches and black/red labels."""

    unique: dict[bytes, tuple[np.ndarray, int]] = {}
    for source, red in examples:
        inp = np.zeros((2, 30, 30), dtype=np.int8)
        inp[1, :6, :3] = source
        inp[0, :6, :3] = 1 - source
        for row in range(30):
            for col in range(30):
                patch = np.zeros((2, 8), dtype=np.int8)
                for offset in range(8):
                    input_row = row - 7 + offset
                    if 0 <= input_row < 30:
                        patch[:, offset] = inp[:, input_row, col]
                label = int(
                    row < 9
                    and col < 3
                    and ((not red[row, col]) if output_channel == 0 else red[row, col])
                )
                vector = patch.ravel()
                key = vector.tobytes()
                if key in unique:
                    assert unique[key][1] == label
                else:
                    unique[key] = (vector, label)
    features = np.asarray([value[0] for value in unique.values()], dtype=np.float64)
    labels = np.asarray([value[1] for value in unique.values()], dtype=np.int8)
    return features, labels


def constraints_for_deltas(
    examples, output_channel: int, deltas: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique constraints for arbitrary relative vertical offsets."""

    unique: dict[bytes, tuple[np.ndarray, int]] = {}
    for source, red in examples:
        inp = np.zeros((2, 30, 30), dtype=np.int8)
        inp[1, :6, :3] = source
        inp[0, :6, :3] = 1 - source
        for row in range(30):
            for col in range(30):
                vector = np.zeros((2, len(deltas)), dtype=np.int8)
                for offset, delta in enumerate(deltas):
                    input_row = row + delta
                    if 0 <= input_row < 30:
                        vector[:, offset] = inp[:, input_row, col]
                label = int(
                    row < 9
                    and col < 3
                    and ((not red[row, col]) if output_channel == 0 else red[row, col])
                )
                vector = vector.ravel()
                key = vector.tobytes()
                if key in unique:
                    assert unique[key][1] == label
                else:
                    unique[key] = (vector, label)
    return (
        np.asarray([value[0] for value in unique.values()], dtype=np.float64),
        np.asarray([value[1] for value in unique.values()], dtype=np.int8),
    )


def minimum_support(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Minimize nonzero weights using a mixed-integer linear formulation."""

    width = features.shape[1]
    big_m = 100.0
    # Variables are [weights..., active indicators...].
    rows = len(features) + 2 * width
    matrix = lil_matrix((rows, 2 * width), dtype=np.float64)
    lower = np.full(rows, -np.inf)
    upper = np.full(rows, np.inf)
    for index, (feature, label) in enumerate(zip(features, labels)):
        matrix[index, :width] = feature
        if label:
            lower[index] = 1.0
        else:
            upper[index] = 0.0
    for index in range(width):
        # w_i <= M z_i and -w_i <= M z_i.
        matrix[len(features) + 2 * index, index] = 1.0
        matrix[len(features) + 2 * index, width + index] = -big_m
        upper[len(features) + 2 * index] = 0.0
        matrix[len(features) + 2 * index + 1, index] = -1.0
        matrix[len(features) + 2 * index + 1, width + index] = -big_m
        upper[len(features) + 2 * index + 1] = 0.0
    objective = np.r_[np.zeros(width), np.ones(width)]
    bounds = Bounds(np.r_[np.full(width, -big_m), np.zeros(width)],
                    np.r_[np.full(width, big_m), np.ones(width)])
    result = milp(
        objective,
        integrality=np.r_[np.zeros(width), np.ones(width)],
        bounds=bounds,
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 300.0},
    )
    assert result.success, result.message
    weights = result.x[:width]
    weights[np.abs(weights) < 1e-7] = 0.0
    signed = features @ weights
    assert np.all(signed[labels == 1] >= 1.0 - 1e-6)
    assert np.all(signed[labels == 0] <= 1e-6)
    return weights


def main() -> None:
    examples = legal_examples()
    print(f"legal configurations: {len(examples)}")
    total = 0
    for channel, name in [(0, "black"), (2, "red")]:
        features, labels = constraints(examples, channel)
        print(f"{name}: {len(features)} unique constraints")
        weights = minimum_support(features, labels)
        total += np.count_nonzero(weights)
        print(weights.reshape(2, 8))
        print(f"{name} support: {np.count_nonzero(weights)}")
    print(f"total support: {total}")

    deltas = tuple(range(-29, 30))
    print(f"unrestricted relative-offset lower bound over {len(deltas)} deltas")
    total = 0
    for channel, name in [(0, "black"), (2, "red")]:
        features, labels = constraints_for_deltas(examples, channel, deltas)
        weights = minimum_support(features, labels)
        matrix = weights.reshape(2, len(deltas))
        used = [
            (input_channel, delta, matrix[input_channel, index])
            for input_channel in range(2)
            for index, delta in enumerate(deltas)
            if matrix[input_channel, index]
        ]
        total += len(used)
        print(f"{name} support {len(used)}: {used}")
    print(f"unrestricted total support: {total}")


if __name__ == "__main__":
    main()
