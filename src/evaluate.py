"""Post-training metrics: per-metabolite fit and gradient loadings."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reconstruction_r2(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """In-sample R^2 = 1 - SSE/SST for each metabolite (negative values kept).

    Parameters
    ----------
    target, prediction : np.ndarray, shape (n_spots, n_features)

    Returns
    -------
    np.ndarray, shape (n_features,)
        NaN where a feature has zero variance.
    """
    SSE = np.sum((target-prediction) ** 2,axis=0)
    ss_total = np.sum((target-target.mean(axis = 0)) ** 2, axis = 0)
    SST = np.where(ss_total == 0, np.nan, ss_total)
    return 1 - (SSE/SST)


def scaled_loadings(weights: np.ndarray, isodepths: np.ndarray) -> np.ndarray:
    """|decoder weight| x std(isodepth), so loadings are comparable across gradients.

    Parameters
    ----------
    weights : np.ndarray, shape (n_features, n_isodepth)
    isodepths : np.ndarray, shape (n_spots, n_isodepth)

    Returns
    -------
    np.ndarray, shape (n_features, n_isodepth)
    """
    # Multiplying the absolute values of the weight by each isodepth std
    return np.abs(weights) * isodepths.std(axis=0, ddof=0) 


def rank_metabolites(mz: np.ndarray, r2: np.ndarray, loadings: np.ndarray) -> pd.DataFrame:
    """Table of metabolites sorted by R^2, descending.

    Columns: feature_index, mz, reconstruction_r2, strongest_gradient (1-based),
    scaled_loading_1 ... scaled_loading_K.
    """
    data = {
        "feature_index": np.arange(len(mz)),
        "mz": mz,
        "reconstruction_r2": r2,
        "strongest_gradient": np.argmax(loadings, axis=1) + 1
    }
    for k in range(loadings.shape[1]):
        data[f"scaled_loading_{k + 1}"] = loadings[:, k]

    rank_df = pd.DataFrame(data)
    rank_df = rank_df.sort_values("reconstruction_r2", ascending=False, kind="stable")
    return rank_df.reset_index(drop=True)
