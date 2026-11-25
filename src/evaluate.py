"""Post-training metrics: per-metabolite fit and gradient loadings."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reconstruction_r2(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """R^2 = 1 - SSE/SST for each metabolite, on whatever spots are passed (all spots for
    reconstruction, test spots in cv.py). Negative values are kept.

    Parameters
    ----------
    target, prediction : np.ndarray, shape (n_spots, n_features)

    Returns
    -------
    np.ndarray, shape (n_features,)
        NaN when a metabolite barely varies in those spots (SST < 1e-8).
    """
    sse = np.sum((target - prediction) ** 2, axis=0)
    sst = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    # Metabolites that barely vary in these spots give a near-zero denominator and an
    # R^2 that explodes (e.g. -1e15). The data is standardized (variance about 1 per spot),
    # so 1e-8 only catches metabolites that are practically constant. Treat them as not computable.
    sst = np.where(sst < 1e-8, np.nan, sst)
    return 1 - sse / sst


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
    # The scale of each isodepth is arbitrary: a decoder weight only means something
    # together with how much its isodepth varies, so both are multiplied
    return np.abs(weights) * isodepths.std(axis=0, ddof=0)


def rank_metabolites(mz: np.ndarray, r2: np.ndarray, loadings: np.ndarray | None) -> pd.DataFrame:
    """Table of metabolites sorted by R^2, descending.

    Columns: feature_index, mz, reconstruction_r2, strongest_gradient (1-based),
    scaled_loading_1 ... scaled_loading_K. Loading columns require a linear decoder.
    """
    data = {
        "feature_index": np.arange(len(mz)),
        "mz": mz,
        "reconstruction_r2": r2,
    }
    if loadings is not None:
        data["strongest_gradient"] = np.argmax(loadings, axis=1) + 1
        for k in range(loadings.shape[1]):
            data[f"scaled_loading_{k + 1}"] = loadings[:, k]

    rank_df = pd.DataFrame(data)
    rank_df = rank_df.sort_values("reconstruction_r2", ascending=False, kind="stable")
    return rank_df.reset_index(drop=True)
