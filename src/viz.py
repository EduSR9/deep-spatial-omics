"""Figures for MET-MAP results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_annotations(
    coords: np.ndarray,
    annotation: np.ndarray,
    save_path: str | Path,
) -> None:
    """
    Tissue map coloured by Tumor / Stroma / Unlabelled, with counts in the legend.
    """
    print("Running my new func!")
    colors = {
        "Tumor": "#c96862",
        "Stroma": "#62a6a5",
        "Unlabelled": "#b8b8b8",
    }

    plt.figure(figsize=(6, 6))

    for label, color in colors.items():

        mask = annotation == label

        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            s=5,
            label=f"{label} (n={mask.sum()})",
        )

    plt.gca().set_aspect("equal")

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    plt.title("BC-515 Tissue Annotation")

    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.close()  


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


def plot_training_loss(
    total_losses: np.ndarray,
    l1_losses: np.ndarray,
    save_path: str | Path,
) -> None:

    mse_losses = total_losses - l1_losses

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
    )

    axes[0].plot(
        total_losses,
        label="Total loss",
    )

    axes[0].plot(
        mse_losses,
        label="MSE",
    )

    axes[0].set_title("Training Objective")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        l1_losses,
        color="orange",
    )

    axes[1].set_title("L1 Penalty")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Penalty")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.close()


def plot_isodepths(
    coords: np.ndarray,
    isodepths: np.ndarray,
    save_path: str | Path,
) -> None:
    
    """One tissue map per isodepth, each scaled to [0, 1] for display.

    Sign and scale of an isodepth are arbitrary, so only the relative pattern is shown.
    """

    n_iso = isodepths.shape[1]

    fig, axes = plt.subplots(
        1,
        n_iso,
        figsize=(5 * n_iso, 5),
    )

    if n_iso == 1:
        axes = [axes]

    for i, ax in enumerate(axes):

        d = isodepths[:, i]

        scaled = (
            d - d.min()
        ) / (
            np.ptp(d) + 1e-12
        )

        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=scaled,
            cmap="viridis",
            s=5,
            vmin=0,
            vmax=1,
        )

        ax.set_title(
            f"Isodepth {i + 1}"
        )

        ax.set_aspect("equal")

        fig.colorbar(
            scatter,
            ax=ax,
        )

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.close()


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
    
    """
    Mean intensity +- SEM in equal-frequency bins along one isodepth.
    """

    bins = pd.qcut(
        isodepth,
        q=n_bins,
        duplicates="drop",
    )

    means_x = []
    means_y = []
    sems = []

    for category in bins.categories:

        mask = bins == category

        x = isodepth[mask]
        y = intensity[mask]

        means_x.append(
            x.mean()
        )

        means_y.append(
            y.mean()
        )

        sems.append(
            y.std(ddof=1)
            / np.sqrt(len(y))
        )

    plt.figure(figsize=(6, 4))

    plt.errorbar(
        means_x,
        means_y,
        yerr=sems,
        marker="o",
        capsize=3,
    )

    plt.xlabel("Isodepth")
    plt.ylabel("Metabolite intensity")

    plt.title(
        "Metabolite Gradient Profile"
    )

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.close()
