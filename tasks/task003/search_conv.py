#!/usr/bin/env python3
"""Search for a smaller exact single-Conv task003 kernel."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[2]
with (ROOT / "kaggle_tasks_data" / "task003.json").open() as handle:
    task = json.load(handle)
examples = task["train"] + task["test"] + task["arc-gen"]

inputs = np.zeros((len(examples), 2, 30, 30), dtype=np.int8)
targets = np.zeros((len(examples), 2, 30, 30), dtype=np.int8)
for example_index, example in enumerate(examples):
    source = np.asarray(example["input"])
    result = np.asarray(example["output"])
    inputs[example_index, 1, :6, :3] = source
    inputs[example_index, 0, :6, :3] = 1 - source
    targets[example_index, 1, :9, :3] = result == 2
    targets[example_index, 0, :9, :3] = result == 0


def dataset(kernel_height, kernel_width, dilation_height, dilation_width, pad_top, pad_left):
    """Build unique patch/label constraints for a Conv configuration."""

    feature_batches = []
    label_batches = [[], []]
    for output_row, output_col in itertools.product(range(30), repeat=2):
        rows = output_row + np.arange(kernel_height) * dilation_height - pad_top
        cols = output_col + np.arange(kernel_width) * dilation_width - pad_left
        overlaps = any(0 <= row < 6 for row in rows) and any(
            0 <= col < 3 for col in cols
        )
        if not overlaps and not targets[:, :, output_row, output_col].any():
            continue
        patches = np.zeros(
            (len(examples), 2, kernel_height, kernel_width), dtype=np.int8
        )
        for kernel_row, input_row in enumerate(rows):
            for kernel_col, input_col in enumerate(cols):
                if 0 <= input_row < 30 and 0 <= input_col < 30:
                    patches[:, :, kernel_row, kernel_col] = inputs[
                        :, :, input_row, input_col
                    ]
        feature_batches.append(patches.reshape(len(examples), -1))
        for channel in range(2):
            label_batches[channel].append(targets[:, channel, output_row, output_col])
    return np.concatenate(feature_batches), [
        np.concatenate(labels) for labels in label_batches
    ]


def solve(features, labels, affine=False):
    """Find a threshold separator, or return None on a conflict."""

    if affine:
        features = np.c_[features, np.ones(len(features), dtype=features.dtype)]

    unique = {}
    for feature, label in zip(features, labels):
        key = bytes(feature)
        label = int(label)
        if key in unique and unique[key][1] != label:
            return None
        unique[key] = (feature, label)
    unique_features = np.asarray([item[0] for item in unique.values()], dtype=float)
    unique_labels = np.asarray([item[1] for item in unique.values()])
    lhs = np.where(unique_labels[:, None], -unique_features, unique_features)
    rhs = np.where(unique_labels, -1.0, 0.0)
    result = linprog(
        np.zeros(unique_features.shape[1]),
        A_ub=lhs,
        b_ub=rhs,
        bounds=[(None, None)] * unique_features.shape[1],
        method="highs",
    )
    return result.x if result.success else None


def main():
    affine = True
    tested = 0
    for area in range(1, 8):
        for kernel_height in range(1, area + 1):
            if area % kernel_height:
                continue
            kernel_width = area // kernel_height
            for dilation_height in range(1, 10):
                for dilation_width in range(1, 5):
                    total_vertical_pad = (kernel_height - 1) * dilation_height
                    total_horizontal_pad = (kernel_width - 1) * dilation_width
                    for pad_top in range(total_vertical_pad + 1):
                        for pad_left in range(total_horizontal_pad + 1):
                            tested += 1
                            features, labels = dataset(
                                kernel_height,
                                kernel_width,
                                dilation_height,
                                dilation_width,
                                pad_top,
                                pad_left,
                            )
                            black = solve(features, labels[0], affine=affine)
                            if black is None:
                                continue
                            red = solve(features, labels[1], affine=affine)
                            if red is None:
                                continue
                            config = (
                                kernel_height,
                                kernel_width,
                                dilation_height,
                                dilation_width,
                                pad_top,
                                pad_left,
                            )
                            print(
                                f"exact configuration: {config}; "
                                f"logical params={50 * area + (10 if affine else 0)}"
                            )
                            print("black:", black.reshape(2, kernel_height, kernel_width))
                            print("red:", red.reshape(2, kernel_height, kernel_width))
                            return
        print(f"no exact kernel of area {area}; tested {tested} configurations", flush=True)
    print(f"no exact kernel below area 8; tested {tested} configurations")


if __name__ == "__main__":
    main()
