#!/usr/bin/env python3
"""Search compact one-node ConvTranspose solutions for task003."""

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


def dataset(kernel_height, stride, dilation, pad_top):
    feature_batches = []
    label_batches = [[], []]
    for output_row, output_col in itertools.product(range(30), repeat=2):
        patches = np.zeros((len(examples), 2, kernel_height), dtype=np.int8)
        for kernel_row in range(kernel_height):
            numerator = output_row + pad_top - kernel_row * dilation
            if numerator % stride:
                continue
            input_row = numerator // stride
            if 0 <= input_row < 30:
                patches[:, :, kernel_row] = inputs[:, :, input_row, output_col]
        if not patches.any() and not targets[:, :, output_row, output_col].any():
            continue
        feature_batches.append(patches.reshape(len(examples), -1))
        for channel in range(2):
            label_batches[channel].append(targets[:, channel, output_row, output_col])
    return np.concatenate(feature_batches), [np.concatenate(x) for x in label_batches]


def solve(features, labels, affine):
    if affine:
        features = np.c_[features, np.ones(len(features), dtype=features.dtype)]
    unique = {}
    for feature, label in zip(features, labels):
        key = feature.tobytes()
        if key in unique and unique[key][1] != label:
            return None
        unique[key] = (feature, int(label))
    x = np.asarray([value[0] for value in unique.values()], dtype=float)
    y = np.asarray([value[1] for value in unique.values()])
    lhs = np.where(y[:, None], -x, x)
    rhs = np.where(y, -1.0, 0.0)
    result = linprog(np.zeros(x.shape[1]), A_ub=lhs, b_ub=rhs,
                     bounds=[(None, None)] * x.shape[1], method="highs")
    return result.x if result.success else None


def main():
    tested = 0
    for affine in [False, True]:
        for kernel_height in range(1, 8):
            for stride in range(1, 7):
                for dilation in range(1, 13):
                    for output_padding in range(stride):
                        total_pad = 29 * (stride - 1) + output_padding + dilation * (kernel_height - 1)
                        for pad_top in range(total_pad + 1):
                            tested += 1
                            features, labels = dataset(kernel_height, stride, dilation, pad_top)
                            black = solve(features, labels[0], affine)
                            if black is None:
                                continue
                            red = solve(features, labels[1], affine)
                            if red is None:
                                continue
                            print("exact:", (kernel_height, stride, dilation, output_padding, pad_top, total_pad-pad_top, affine))
                            print("cost:", 50 * kernel_height + (10 if affine else 0))
                            print("black:", black.reshape(2, kernel_height) if not affine else black)
                            print("red:", red.reshape(2, kernel_height) if not affine else red)
                            print("tested:", tested)
                            return
            print(f"no solution at height {kernel_height}, affine={affine}; tested={tested}", flush=True)
    print("no solution; tested", tested)


if __name__ == "__main__":
    main()
