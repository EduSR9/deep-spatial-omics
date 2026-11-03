"""Evaluation over train/test splits: a freshly trained model per split, held-out metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def evaluate_splits(
    model_cls: type,
    model_cfg: dict,
    train_cfg: dict,
    coords: torch.Tensor,
    targets: torch.Tensor,
    splits: list[tuple[np.ndarray, np.ndarray]],
    regions: np.ndarray,
    protocol: str,
) -> pd.DataFrame:
    """Train a new model for each split and score it on the held-out spots.

    Parameters
    ----------
    model_cls, model_cfg : model class and the "model" section of config.yaml
    train_cfg : the "training" section of config.yaml
    coords, targets : torch.Tensor, all spots
    splits : list of (train_idx, test_idx)
    regions : np.ndarray of str, shape (n_spots,), from splits.region_labels
    protocol : "random" or "spatial", stored in the output table

    Returns
    -------
    pd.DataFrame, one row per split with n_train, n_test, held-out counts per region,
    test_mse and median_r2.
    """
    # TODO: for each split i, set torch.manual_seed(train_cfg["seed"] + i)
    # TODO: build a new model_cls(n_features, hidden_dim, n_isodepth) from model_cfg
    # TODO: train_model on the training spots only (index coords and targets with torch.as_tensor(train_idx))
    # TODO: predict on the held-out spots with no_grad and model.eval()
    # TODO: r2 = reconstruction_r2(held-out targets, prediction); test_mse; median_r2 = nanmedian(r2)
    # TODO: count held-out spots per region (Tumor, Stroma, Interface) and append one row per split
    raise NotImplementedError


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of test_mse and median_r2 for each protocol."""
    # TODO: groupby("protocol") and agg mean and std (ddof=1) for test_mse and median_r2
    raise NotImplementedError
