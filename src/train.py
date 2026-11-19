"""Full-batch training, shared by every model in src/model.py."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Adam, or AdamW when weight_decay > 0.

    With AdamW, weight decay applies to weight matrices only, not to biases or norm layers.
    With weight_decay = 0 this is plain Adam, which is what the MET-MAP reference uses.
    """
    weight_decay = cfg.get("weight_decay", 0.0)
    if weight_decay > 0:
        return torch.optim.AdamW([
            {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": weight_decay},
            {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
        ], lr=cfg["learning_rate"])
    return torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])


def train_model(
    model: nn.Module,
    coords: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Train a fresh model on preprocessed training spots only.

    The loss is the mean squared error plus the model's own penalty, if it has one.
    Returns the total loss per epoch and the penalty per epoch (None if the model has none).
    """
    model.train()
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=cfg["lr_end_factor"],
        total_iters=cfg["lr_total_iters"],
    )
    total_losses = np.zeros(cfg["epochs"])
    penalty_losses = np.zeros(cfg["epochs"])
    has_penalty = False
    # Match the reference input-gradient path.
    coords = coords.detach().clone().requires_grad_(True)

    for epoch in range(cfg["epochs"]):
        optimizer.zero_grad()
        coords.grad = None
        prediction, _ = model(coords)
        mse = nn.functional.mse_loss(prediction, targets, reduction="mean")
        penalty = model.penalty(cfg["lasso_lambda"])
        if penalty is None:
            total = mse
        else:
            total = mse + penalty
            penalty_losses[epoch] = penalty.item()
            has_penalty = True
        total_losses[epoch] = total.item()
        total.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}: total={total.item():.6f}, MSE={mse.item():.6f}", flush=True)

    return total_losses, (penalty_losses if has_penalty else None)
