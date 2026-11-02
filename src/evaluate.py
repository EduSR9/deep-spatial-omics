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
    # TODO: ss_total per column, ss_residual per column, r2 = 1 - ss_residual / ss_total
    raise NotImplementedError


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
    # TODO: np.abs(weights) * isodepths.std(axis=0, ddof=0)[None, :]
    raise NotImplementedError


def rank_metabolites(mz: np.ndarray, r2: np.ndarray, loadings: np.ndarray) -> pd.DataFrame:
    """Table of metabolites sorted by R^2, descending.

    Columns: feature_index, mz, reconstruction_r2, strongest_gradient (1-based),
    scaled_loading_1 ... scaled_loading_K.
    """
    # TODO: build the DataFrame, sort by reconstruction_r2 (kind="stable"), reset index
    raise NotImplementedError
