"""Data loading and preprocessing for BC-515 spatial metabolomics."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np


def load_bc515(filepath: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load BC-515 metabolomics, coordinates and annotations.

    The metabolite matrix lives in adata.uns["msi"]; adata.X is RNA and is not used.

    Parameters
    ----------
    filepath : str or Path
        Path to BC_515_Section_1.h5ad.

    Returns
    -------
    coords : np.ndarray, shape (3508, 2)
    msi : np.ndarray, shape (3508, 2086)
        Raw metabolite intensities.
    annotation : np.ndarray of str, shape (3508,)
        "Tumor", "Stroma" or "Unlabelled". Used for plots only, never for training.
    mz : np.ndarray of str, shape (2086,)
        m/z label of each metabolite.
    """
    # TODO: read the file with anndata.read_h5ad
    # TODO: coords = adata.obsm["spatial"], msi = adata.uns["msi"] (both float64)
    # TODO: annotation = adata.obs["annotation"] as str, mz = adata.uns["mz_features"] as str
    # TODO: assert msi and coords are finite and msi >= 0
    raise NotImplementedError


def tic_normalize(msi: np.ndarray) -> np.ndarray:
    """Total-ion-count normalization followed by log1p.

    Instrument TIC is not available, so the row sum of measured features is used.
    Formula: log1p(X_ij / TIC_i * mean(TIC)).

    Parameters
    ----------
    msi : np.ndarray, shape (n_spots, n_metabolites)

    Returns
    -------
    np.ndarray, same shape as msi
    """
    # TODO: tic = msi.sum(axis=1); assert all tic > 0
    # TODO: return np.log1p(msi / tic[:, None] * tic.mean())
    raise NotImplementedError


def select_top_metabolites(msi_log: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n metabolites with the highest mean intensity.

    Parameters
    ----------
    msi_log : np.ndarray, shape (n_spots, n_metabolites)
    n : int

    Returns
    -------
    np.ndarray, shape (n,)
    """
    # TODO: rank columns by mean, descending, and return the first n indices
    raise NotImplementedError


def standardize(x: np.ndarray) -> np.ndarray:
    """Z-score each column (population std, ddof=0, same as sklearn StandardScaler).

    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    np.ndarray, same shape as x
    """
    # TODO: subtract column mean, divide by column std (ddof=0)
    raise NotImplementedError
