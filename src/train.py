"""Full-batch training, shared by every model in src/model.py."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.model import AttentionMetMap


def _forward_batch(model, batch, reference_coords, self_start=None):
    if not isinstance(model, AttentionMetMap):
        return model(batch)
    if reference_coords is None:
        raise ValueError("Isodepth KNN requires the coordinates of the training spots")
    self_indices = None
    if self_start is not None:
        self_indices = torch.arange(self_start, self_start + len(batch), device=batch.device)
    return model(batch, reference_coords, self_indices)


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

    Spot batches accumulate gradients, with one optimizer update per epoch.
    The loss is the mean squared error plus the model's own penalty, if it has one.
    Returns the total loss per epoch and the penalty per epoch (None if the model has none).
    """
    device = torch.device(cfg.get("device", "cpu"))
    model.to(device).train()
    batch_size = cfg.get("batch_size", len(coords))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
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
    reference_coords = coords.detach().to(device) if isinstance(model, AttentionMetMap) else None

    for epoch in range(cfg["epochs"]):
        optimizer.zero_grad()
        coords.grad = None
        mse = torch.zeros((), device=device)
        for start in range(0, len(coords), batch_size):
            x = coords[start:start + batch_size].to(device)
            y = targets[start:start + batch_size].to(device)
            prediction, _ = _forward_batch(model, x, reference_coords, self_start=start)
            batch_loss = nn.functional.mse_loss(prediction, y, reduction="mean") * len(x) / len(coords)
            batch_loss.backward()
            mse += batch_loss.detach()
        penalty = model.penalty(cfg["lasso_lambda"])
        if penalty is None:
            total = mse
        else:
            total = mse + penalty
            penalty_losses[epoch] = penalty.item()
            has_penalty = True
            penalty.backward()
        total_losses[epoch] = total.item()
        optimizer.step()
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}: total={total.item():.6f}, MSE={mse.item():.6f}", flush=True)

    return total_losses, (penalty_losses if has_penalty else None)


@torch.no_grad()
def predict(model: nn.Module, coords: torch.Tensor, batch_size: int | None = None,
            *, reference_coords: torch.Tensor | None = None, same_spots: bool = False):
    """Evaluate spot batches and return predictions and isodepths on the CPU."""
    model.eval()
    device = next(model.parameters()).device
    if isinstance(model, AttentionMetMap) and reference_coords is not None:
        reference_coords = reference_coords.to(device)
    batch_size = len(coords) if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    predictions, isodepths = [], []
    for start in range(0, len(coords), batch_size):
        batch = coords[start:start + batch_size].to(device)
        prediction, depth = _forward_batch(model, batch, reference_coords, start if same_spots else None)
        predictions.append(prediction.cpu())
        isodepths.append(depth.cpu())
    return torch.cat(predictions), torch.cat(isodepths)
