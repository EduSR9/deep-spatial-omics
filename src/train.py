"""Full-batch training for MET-MAP."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.attention import AttentionMetMap


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
) -> tuple[np.ndarray, np.ndarray | None]:
    """Train a fresh model on preprocessed training spots only."""
    device = torch.device(cfg.get("device", "cpu"))
    model.to(device).train()
    model.train()
    attention = isinstance(model, AttentionMetMap)
    if attention:
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if p.ndim >= 2],
             "weight_decay": cfg.get("weight_decay", 0.01)},
            {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
        ], lr=cfg["learning_rate"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=cfg["lr_end_factor"],
        total_iters=cfg["lr_total_iters"],
    )
    total_losses = np.zeros(cfg["epochs"])
    l1_losses = None if attention else np.zeros(cfg["epochs"])
    penalty_losses = np.zeros(cfg["epochs"])
    has_penalty = False
    # Match the reference input-gradient path.
    coords = coords.detach().clone().to(device).requires_grad_(True)
    targets = targets.to(device)
    
    for epoch in range(cfg["epochs"]):
        optimizer.zero_grad()
        coords.grad = None
        prediction, _ = model(coords)
        if l1_losses is not None:
            total, l1 = compute_loss(
                prediction,
                targets,
                model.decoder.weight,
                cfg["lasso_lambda"],
            )
            l1_losses[epoch] = l1.item()
            penalty_losses[epoch] = l1.item()
            has_penalty = True
        else:
            total = nn.functional.mse_loss(
                prediction,
                targets,
                reduction="mean",
            )
        total_losses[epoch] = total.item()
        total.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 500 == 0:
            mse = total_losses[epoch] - (l1_losses[epoch] if l1_losses is not None else 0.0)
            print(f"Epoch {epoch + 1}: total={total.item():.6f}, MSE={mse:.6f}", flush=True)

    # The shared evaluation code uses CPU tensors and NumPy.
    model.cpu()
    return total_losses, (penalty_losses if has_penalty else None)

            
            

