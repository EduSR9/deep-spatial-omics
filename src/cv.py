"""Evaluation over train/test splits: a freshly trained model per split, held-out metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from src.train import train_model
from src.evaluate import reconstruction_r2


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
    results = []

    for i, (train_idx, test_idx) in enumerate(splits):
        torch.manual_seed(train_cfg["seed"] + i)

        model = model_cls(
            n_features=targets.shape[1],
            hidden_dim=model_cfg["hidden_dim"],
            n_isodepth=model_cfg["n_isodepth"],
            )

        train_t = torch.as_tensor(train_idx)
        test_t = torch.as_tensor(test_idx)
        train_coords = coords[train_t]
        train_targets = targets[train_t]
        test_coords = coords[test_t]
        test_targets = targets[test_t]


        # Train a new model
        train_model(model, train_coords, train_targets, train_cfg)

        # Test the model
        model.eval()
        with torch.no_grad():
            predictions, _ = model(test_coords) # we discard the isodepths

        # Get the r2, median r2 and mse
        predictions = predictions.numpy()
        test_targets = targets[test_t].numpy()

        r2 = reconstruction_r2(test_targets, predictions)
        median_r2 = float(np.nanmedian(r2))
        test_mse = float(np.mean((predictions - test_targets) ** 2))


        # Count regions in the test set
        test_regions = regions[test_idx]
        tumor_count = np.sum(test_regions == "Tumor")
        stroma_count = np.sum(test_regions == "Stroma")
        interface_count = np.sum(test_regions == "Interface")

        # Save results for this split
        results.append({
            "protocol": protocol,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "Tumor": tumor_count,
            "Stroma": stroma_count,
            "Interface": interface_count,
            "test_mse": test_mse,
            "median_r2": median_r2
        })

    return pd.DataFrame(results)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of test_mse and median_r2 for each protocol."""
    return results.groupby("protocol")[["test_mse", "median_r2"]].agg(["mean", "std"])