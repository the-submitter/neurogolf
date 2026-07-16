#!/usr/bin/env python3
"""Search low-rank shared Gram row factors for the task003 Einsum."""

from __future__ import annotations

import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
from optimize_sparse import legal_examples  # noqa: E402


def dataset() -> tuple[np.ndarray, np.ndarray]:
    mapping = {}
    for source, continuation in legal_examples():
        for col in range(3):
            signed_source = tuple((2 * source[:, col] - 1).tolist())
            signed_target = tuple((2 * continuation[:, col] - 1).tolist())
            if signed_source in mapping:
                assert mapping[signed_source] == signed_target
            mapping[signed_source] = signed_target
    return np.asarray(list(mapping.keys()), dtype=np.float64), np.asarray(
        list(mapping.values()), dtype=np.float64
    )


def loss_and_gradient(vector, rank, source, target):
    split = rank * 6
    basis = vector[:split].reshape(rank, 6)
    continuation = vector[split:].reshape(rank, 3)
    factors = np.concatenate([basis, continuation], axis=1)
    projected = source @ basis.T
    logits = projected @ factors
    deficit = np.maximum(0.0, 1.0 - target * logits)
    loss = np.sum(deficit**2)
    derivative = -2.0 * target * deficit
    basis_gradient = (derivative @ factors.T).T @ source
    basis_gradient += projected.T @ derivative[:, :6]
    continuation_gradient = projected.T @ derivative[:, 6:]
    gradient = np.concatenate(
        [basis_gradient.ravel(), continuation_gradient.ravel()]
    )
    return loss, gradient


def main() -> None:
    source, target = dataset()
    print(f"unique signed sequences: {len(source)}", flush=True)
    rng = np.random.default_rng(20260716)
    for rank in range(2, 7):
        best = None
        for restart in range(24):
            initial = rng.normal(size=rank * 9)
            result = minimize(
                loss_and_gradient,
                initial,
                args=(rank, source, target),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 1500, "ftol": 1e-13, "gtol": 1e-9},
            )
            if best is None or result.fun < best.fun:
                best = result
            if result.fun < 1e-12:
                break
        assert best is not None
        print(f"rank {rank}: loss={best.fun:.12g}; restarts={restart + 1}")
        if best.fun < 1e-10:
            basis = best.x[: rank * 6].reshape(rank, 6)
            continuation = best.x[rank * 6 :].reshape(rank, 3)
            factors = np.zeros((rank, 30))
            factors[:, :6] = basis
            factors[:, 6:9] = continuation
            margin = target * (source @ basis.T @ factors[:, :9])
            print("minimum margin:", margin.min())
            print(factors[:, :9])
            break


if __name__ == "__main__":
    main()
