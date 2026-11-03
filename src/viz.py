"""Figures for MET-MAP results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_annotations(coords: np.ndarray, annotation: np.ndarray, save_path: str | Path) -> None:
    """Tissue map colored by Tumor / Stroma / Unlabelled, with counts in the legend."""
    # TODO: one scatter per label (Tumor "#c96862", Stroma "#62a6a5", Unlabelled "#b8b8b8")
    # TODO: equal aspect, legend "Label (n=...)", save and close
    raise NotImplementedError


def plot_spatial_folds(
    coords: np.ndarray,
    fold_ids: np.ndarray,
    regions: np.ndarray,
    save_path: str | Path,
) -> None:
    """Left: spots colored by spatial fold. Right: spots colored by region (Tumor, Stroma, Interface)."""
    # TODO: two side-by-side panels with equal aspect
    # TODO: left: scatter coords colored by fold_ids (categorical colormap), title "Spatial folds"
    # TODO: right: scatter coords colored by regions (Tumor, Stroma, Interface, Unlabelled), legend with counts
    # TODO: save and close
    raise NotImplementedError


def plot_training_loss(total_losses: np.ndarray, l1_losses: np.ndarray, save_path: str | Path) -> None:
    """Left: total objective and MSE (total - l1) per epoch. Right: L1 penalty per epoch."""
    # TODO: two side-by-side panels, save and close
    raise NotImplementedError


def plot_isodepths(coords: np.ndarray, isodepths: np.ndarray, save_path: str | Path) -> None:
    """One tissue map per isodepth, each scaled to [0, 1] for display.

    Sign and scale of an isodepth are arbitrary, so only the relative pattern is shown.
    """
    # TODO: scaled = (d - d.min(0)) / (np.ptp(d, 0) + 1e-12)
    # TODO: one panel per isodepth, cmap "viridis", vmin 0, vmax 1, colorbar, equal aspect
    raise NotImplementedError


def plot_top_metabolites(
    coords: np.ndarray,
    msi_log: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    r2: np.ndarray,
    top_idx: np.ndarray,
    mz: np.ndarray,
    save_path: str | Path,
) -> None:
    """For the top-ranked metabolites: tissue map (top row) and observed vs reconstructed (bottom row)."""
    # TODO: top row: scatter coords colored by msi_log[:, j], title with m/z
    # TODO: bottom row: target[:, j] vs prediction[:, j] with dashed y = x, title with R^2
    raise NotImplementedError


def plot_gradient_profile(
    isodepth: np.ndarray,
    intensity: np.ndarray,
    save_path: str | Path,
    n_bins: int = 20,
) -> None:
    """Mean intensity +- SEM in equal-frequency bins along one isodepth."""
    # TODO: bin isodepth with quantiles (pd.qcut, duplicates="drop")
    # TODO: per bin: mean isodepth, mean intensity, SEM = std / sqrt(n); errorbar plot
    raise NotImplementedError
