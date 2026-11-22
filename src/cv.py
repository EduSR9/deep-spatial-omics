"""Evaluation over train/test splits: a freshly trained model per split, held-out metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from src.train import train_model
from src.evaluate import reconstruction_r2
from src.data import preprocess_split


def evaluate_splits(
    model_cls: type,
    model_cfg: dict,
    train_cfg: dict,
    coords: np.ndarray,
    msi: np.ndarray,
    mz: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    regions: np.ndarray,
    protocol: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train a new model for each split and score it on the held-out spots.

    Parameters
    ----------
    model_cls, model_cfg : model class and the "model" section of config.yaml
    train_cfg : the "training" section of config.yaml
    coords, msi : np.ndarray, raw coordinates and MSI for all spots
    mz : np.ndarray, original m/z labels in MSI column order
    splits : list of (train_idx, test_idx)
    regions : np.ndarray of str, shape (n_spots,), from splits.region_labels
    protocol : "random" or "spatial", stored in the output table

    Returns
    -------
    Two tables with split-level metrics and held-out R^2 per feature.
    """
    results = []
    feature_results = []

    for i, (train_idx, test_idx) in enumerate(splits):
        # seed + i: every split starts from different weights, and runs stay reproducible
        torch.manual_seed(train_cfg["seed"] + i)

        arrays, keep = preprocess_split(coords, msi, train_idx, test_idx, model_cfg["n_features"])
        train_coords, test_coords, train_targets, test_targets = (
            torch.tensor(x, dtype=torch.float32) for x in arrays
        )

        model = model_cls(
            n_features=train_targets.shape[1],
            hidden_dim=model_cfg["hidden_dim"],
            n_isodepth=model_cfg["n_isodepth"],
        )

        # 1. Train a new model on the training spots only
        train_model(model, train_coords, train_targets, train_cfg)

        # 2. Predict the held-out spots
        model.eval()
        with torch.no_grad():
            predictions, _ = model(test_coords)  # The isodepths are not needed here

        # 3. Held-out R2 per metabolite, its median and the MSE
        predictions = predictions.numpy()
        test_targets = test_targets.numpy()

        r2 = reconstruction_r2(test_targets, predictions)
        median_r2 = float(np.nanmedian(r2))
        test_mse = float(np.mean((predictions - test_targets) ** 2))

        # 4. Count the regions in the test spots
        test_regions = regions[test_idx]
        tumor_count = np.sum(test_regions == "Tumor")
        stroma_count = np.sum(test_regions == "Stroma")
        interface_count = np.sum(test_regions == "Interface")

        # 5. Save the results of this split
        results.append({
            "protocol": protocol,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "Tumor": tumor_count,
            "Stroma": stroma_count,
            "Interface": interface_count,
            "test_mse": test_mse,
            "median_r2": median_r2,
            "split": i,
            "n_valid_r2": int(np.count_nonzero(~np.isnan(r2))),  # Metabolites with a computable R2
        })
        feature_results.append(pd.DataFrame({
            "protocol": protocol,
            "split": i,
            "feature_index": keep,
            "mz": mz[keep],
            "r2": r2,
        }))

    features = pd.concat(feature_results, ignore_index=True) if feature_results else pd.DataFrame()
    return pd.DataFrame(results), features


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of test_mse and median_r2 for each protocol."""
    return results.groupby("protocol")[["test_mse", "median_r2"]].agg(["mean", "std"])
