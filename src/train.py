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
    """MSE over all spots and features plus L1 on the decoder weights.

    Parameters
    ----------
    pred, target : torch.Tensor, shape (n_spots, n_features)
    decoder_weight : torch.Tensor
        model.decoder.weight (bias is not penalized).
    lasso_lambda : float

    Returns
    -------
    total : torch.Tensor
        Scalar MSE + L1.
    l1 : torch.Tensor
        Scalar L1 term alone, for logging.
    """
    # TODO: mse = nn.functional.mse_loss(pred, target, reduction="mean")
    # TODO: l1 = lasso_lambda * torch.norm(decoder_weight, 1)   (sum of |w|, not mean)
    # TODO: return mse + l1, l1
    raise NotImplementedError


def train_model(
    model: nn.Module,
    coords: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Train on all spots (no train/test split, as in the original method).

    Parameters
    ----------
    model : nn.Module
    coords : torch.Tensor, shape (n_spots, 2)
    targets : torch.Tensor, shape (n_spots, n_features)
    cfg : dict
        The "training" section of config.yaml.

    Returns
    -------
    total_losses : np.ndarray, shape (epochs,)
        Objective recorded before each parameter update.
    l1_losses : np.ndarray, shape (epochs,)
    """
    # TODO: optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    # TODO: scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1,
    #       end_factor=cfg["lr_end_factor"], total_iters=cfg["lr_total_iters"])
    # TODO: each epoch: zero_grad, forward, compute_loss, record losses, backward, optimizer.step, scheduler.step
    # TODO: print total and MSE every 500 epochs
    raise NotImplementedError
