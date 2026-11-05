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
        "Tumor", "Stroma" or "Unlabelled".

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


def tic_normalize(msi: np.ndarray, tic_mean: float | None = None) -> np.ndarray:
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

    if tic_mean is None:
        tic_mean = tic.mean()
    msi_norm = msi / tic[:, None] * tic_mean

    return np.log1p(msi_norm)


def select_top_metabolites(
    msi_log: np.ndarray,
    n: int,
) -> np.ndarray:
    """Indices of the n metabolites with highest mean intensity."""

    mean_intensity = msi_log.mean(axis=0)

    return np.argsort(mean_intensity)[::-1][:n]


def standardize(x: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Z-score using reference statistics when supplied (ddof=0)."""

    reference = x if reference is None else reference
    mean = reference.mean(axis=0)

    std = reference.std(axis=0, ddof=0)

    std = np.where(
        std == 0,
        1.0,
        std,
    )

    return (x - mean) / std


def preprocess_split(coords, msi, train_idx, test_idx, n_features=None):
    """Fit preprocessing on training spots and apply it to both subsets."""
    train_coords, test_coords = coords[train_idx], coords[test_idx]
    tic_mean = msi[train_idx].sum(axis=1).mean()
    train_log = tic_normalize(msi[train_idx], tic_mean)
    test_log = tic_normalize(msi[test_idx], tic_mean)
    if n_features is not None:
        keep = select_top_metabolites(train_log, n_features)
        train_log, test_log = train_log[:, keep], test_log[:, keep]
    return (
        standardize(train_coords), standardize(test_coords, train_coords),
        standardize(train_log), standardize(test_log, train_log),
    )
