"""Full-batch training for MET-MAP."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    decoder_weight: torch.Tensor,
    lasso_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean reconstruction error plus decoder weight L1."""
    mse = nn.functional.mse_loss(pred, target, reduction="mean")
    l1 = lasso_lambda * torch.norm(decoder_weight, 1)
    return mse + l1, l1


def train_model(
    model: nn.Module,
    coords: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a fresh model on preprocessed training spots only."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=cfg["lr_end_factor"],
        total_iters=cfg["lr_total_iters"],
    )
    total_losses = np.zeros(cfg["epochs"])
    l1_losses = np.zeros(cfg["epochs"])
    # Match the reference input-gradient path.
    coords = coords.detach().clone().requires_grad_(True)

    for epoch in range(cfg["epochs"]):
        optimizer.zero_grad()
        coords.grad = None
        prediction, _ = model(coords)
        total, l1 = compute_loss(prediction, targets, model.decoder.weight, cfg["lasso_lambda"])
        total_losses[epoch] = total.item()
        l1_losses[epoch] = l1.item()
        total.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}: total={total.item():.6f}, MSE={(total - l1).item():.6f}", flush=True)

    return total_losses, l1_losses
