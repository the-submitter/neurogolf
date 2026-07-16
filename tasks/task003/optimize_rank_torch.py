#!/usr/bin/env python3
"""Batched low-rank search for the task003 shared Gram row factor."""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
from optimize_rank import dataset  # noqa: E402


def main() -> None:
    source_np, target_np = dataset()
    source = torch.tensor(source_np, dtype=torch.float32)
    target = torch.tensor(target_np, dtype=torch.float32)
    torch.manual_seed(20260716)
    restarts = 64
    for rank in range(2, 5):
      for negative_count in range(0, rank // 2 + 1):
        signature = torch.ones(rank, dtype=torch.float32)
        signature[:negative_count] = -1
        parameters = torch.randn(
            restarts, rank, 9, dtype=torch.float32, requires_grad=True
        )
        optimizer = torch.optim.Adam([parameters], lr=0.03)
        best_loss = float("inf")
        best_parameters = None
        for step in range(1500):
            optimizer.zero_grad()
            basis = parameters[:, :, :6]
            projected = torch.einsum("ns,rks->rnk", source, basis)
            logits = torch.einsum(
                "rnk,k,rkt->rnt", projected, signature, parameters
            )
            deficits = torch.relu(1.0 - target[None, :, :] * logits)
            losses = torch.sum(deficits.square(), dim=(1, 2))
            losses.sum().backward()
            optimizer.step()
            value, index = torch.min(losses.detach(), dim=0)
            if value.item() < best_loss:
                best_loss = value.item()
                best_parameters = parameters.detach()[index].clone()
            if best_loss < 1e-12:
                break
        print(
            f"rank {rank}, negatives {negative_count}: "
            f"best loss={best_loss:.12g}; steps={step + 1}"
        )
        if best_loss < 1e-10:
            assert best_parameters is not None
            basis = best_parameters[:, :6]
            logits = (source @ basis.T * signature) @ best_parameters
            margins = target * logits
            print("minimum margin:", margins.min().item())
            print(best_parameters.numpy())
            return


if __name__ == "__main__":
    main()
