"""Data loading and preprocessing for BC-515 spatial metabolomics."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np


def load_bc515(
    filepath: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    adata = ad.read_h5ad(filepath)

    coords = np.asarray(
        adata.obsm["spatial"],
        dtype=np.float64,
    )

    msi = np.asarray(
        adata.uns["msi"],
        dtype=np.float64,
    )

    annotation = np.asarray(
        adata.obs["annotation"].astype(str)
    )

    mz = np.asarray(
        adata.uns["mz_features"],
        dtype=str,
    )

    assert np.isfinite(coords).all(), \
        "Coordinates contain NaN or Inf"

    assert np.isfinite(msi).all(), \
        "MSI contains NaN or Inf"

    assert (msi >= 0).all(), \
        "MSI contains negative values"

    return coords, msi, annotation, mz


def tic_normalize(msi: np.ndarray) -> np.ndarray:
    """Total-ion-count normalization followed by log1p.

    Formula:
        log1p(X_ij / TIC_i * mean(TIC))

    Parameters
    ----------
    msi : np.ndarray
        Shape (n_spots, n_metabolites)

    Returns
    -------
    np.ndarray
        Same shape as msi.
    """

    tic = msi.sum(axis=1)

    assert np.all(tic > 0), \
        "At least one spot has TIC = 0"

    msi_norm = msi / tic[:, None] * tic.mean()

    return np.log1p(msi_norm)


def select_top_metabolites(
    msi_log: np.ndarray,
    n: int,
) -> np.ndarray:
    """Indices of the n metabolites with highest mean intensity."""

    mean_intensity = msi_log.mean(axis=0)

    return np.argsort(mean_intensity)[::-1][:n]


def standardize(x: np.ndarray) -> np.ndarray:
    """Z-score each column (ddof=0)."""

    mean = x.mean(axis=0)

    std = x.std(axis=0, ddof=0)

    std = np.where(
        std == 0,
        1.0,
        std,
    )

    return (x - mean) / std