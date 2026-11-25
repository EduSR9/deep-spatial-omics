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

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    scatter = axes[0].scatter(
        coords[:, 0],
        coords[:, 1],
        c=fold_ids,
        cmap="tab10",
        s=5,
    )

    axes[0].set_title("Spatial folds")
    axes[0].set_aspect("equal")

    fig.colorbar(
        scatter,
        ax=axes[0],
    )

    colors = {
        "Tumor": "#c96862",
        "Stroma": "#62a6a5",
        "Interface": "#e0c24d",
        "Unlabelled": "#b8b8b8",
    }

    for label, color in colors.items():

        mask = regions == label

        if np.any(mask):

            axes[1].scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                s=5,
                label=f"{label} (n={mask.sum()})",
            )

    axes[1].set_title("Regions")
    axes[1].set_aspect("equal")
    axes[1].legend()

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.close()


def plot_training_loss(
    total_losses: np.ndarray,
    l1_losses: np.ndarray | None,
    save_path: str | Path,
) -> None:

    if l1_losses is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(np.arange(1, len(total_losses) + 1), total_losses)
        ax.set(xlabel="Epoch", ylabel="Training MSE")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return

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

    n = len(top_idx)

    fig, axes = plt.subplots(
        2,
        n,
        figsize=(4 * n, 8),
        squeeze=False,
    )

    for col, j in enumerate(top_idx):

        scatter = axes[0, col].scatter(
            coords[:, 0],
            coords[:, 1],
            c=msi_log[:, j],
            cmap="viridis",
            s=5,
        )

        axes[0, col].set_title(
            str(mz[j])
        )

        axes[0, col].set_aspect("equal")

        fig.colorbar(
            scatter,
            ax=axes[0, col],
        )

        axes[1, col].scatter(
            target[:, j],
            prediction[:, j],
            s=5,
            alpha=0.5,
        )

        xmin = min(
            target[:, j].min(),
            prediction[:, j].min(),
        )

        xmax = max(
            target[:, j].max(),
            prediction[:, j].max(),
        )

        axes[1, col].plot(
            [xmin, xmax],
            [xmin, xmax],
            "k--",
        )

        axes[1, col].set_title(
            f"R² = {r2[j]:.3f}"
        )

        axes[1, col].set_xlabel("Observed")
        axes[1, col].set_ylabel("Predicted")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.close()


def plot_model_boxplots(out_dir: str | Path, models: list[dict]) -> pd.DataFrame:
    """Boxplot of R² per metabolite, one box per model for each test: reconstruction
    (in-sample), random and spatial (held-out).

    Each box holds one value per metabolite: that metabolite's R² averaged over the splits of
    the test. Reads <out_dir>/<model name>/cv_feature_r2.csv for every model in the list.
    Saves boxplot_r2.png and boxplot_summary.csv in out_dir and returns the summary table.
    """
    from matplotlib.patches import Patch

    out_dir = Path(out_dir)
    protocols_all = ["reconstruction", "random", "spatial"]
    protocol_labels = {"reconstruction": "reconstruction (in-sample)", "random": "random", "spatial": "spatial"}
    rows = []
    for m in models:
        path = out_dir / m["name"] / "cv_feature_r2.csv"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {m['name']}")
            continue
        df = pd.read_csv(path)
        recon_path = out_dir / m["name"] / "reconstruction_r2.csv"
        if recon_path.exists():
            df = pd.concat([df, pd.read_csv(recon_path)], ignore_index=True)
        for protocol in protocols_all:
            sub = df[df.protocol == protocol]
            if sub.empty:
                continue
            per_metabolite = sub.groupby("feature_index")["r2"].mean().dropna()
            # Drop numerically broken values (metabolites with almost no variance in the test
            # spots): their tiny SST makes R2 explode, which says nothing about the model.
            broken = per_metabolite < -10
            if broken.any():
                print(f"NOTE: {m['name']} / {protocol}: {int(broken.sum())} metabolites excluded "
                      f"(R² < -10, numerically unstable)")
            per_metabolite = per_metabolite[~broken]
            rows.append(pd.DataFrame({
                "model": m["name"], "protocol": protocol, "r2": per_metabolite.to_numpy(),
            }))
    if not rows:
        print("No per-metabolite results found, no boxplot written.")
        return pd.DataFrame()

    values = pd.concat(rows, ignore_index=True)
    present = [m for m in models if m["name"] in set(values["model"])]
    protocols = [p for p in protocols_all if p in set(values["protocol"])]

    groups = {
        "Coordinate input": [m for m in present if not m.get("measured_input")],
        "Coordinates and measured metabolites": [m for m in present if m.get("measured_input")],
    }
    panels = [(label, group) for label, group in groups.items() if group]
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), sharey=True, squeeze=False)
    y_min = -2.0
    for ax, (label, group) in zip(axes[0], panels):
        panel_values = values[values["model"].isin(m["name"] for m in group)]
        width = 0.8 / len(group)
        for j, protocol in enumerate(protocols):
            for i, m in enumerate(group):
                vals = panel_values.loc[
                    (panel_values["model"] == m["name"]) & (panel_values["protocol"] == protocol), "r2"
                ]
                pos = j + (i - (len(group) - 1) / 2) * width
                color = m.get("color", "#888888")
                box = ax.boxplot(vals, positions=[pos], widths=width * 0.85, patch_artist=True,
                                 medianprops=dict(color="black"), showfliers=False)
                for patch in box["boxes"]:
                    patch.set(facecolor=color, alpha=0.65, edgecolor=color)

        below = int((panel_values["r2"] < y_min).sum())
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(y_min, 1.05)
        ax.set_title(f"{label}\n({below} values below {y_min:g} outside the axes)")
        ax.set_xticks(range(len(protocols)), [protocol_labels[p] for p in protocols])
        ax.legend(handles=[Patch(facecolor=m.get("color", "#888888"), alpha=0.65, label=m["name"])
                           for m in group])
    axes[0, 0].set_ylabel("R² per metabolite (averaged over splits)")
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_r2.png", dpi=300)
    plt.close(fig)

    summary = values.groupby(["protocol", "model"])["r2"].agg(
        n_metabolites="size",
        median_r2="median",
        mean_r2="mean",
        share_above_0_3=lambda x: (x > 0.3).mean(),
        share_above_0_5=lambda x: (x > 0.5).mean(),
    )
    summary.to_csv(out_dir / "boxplot_summary.csv")
    return summary


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
