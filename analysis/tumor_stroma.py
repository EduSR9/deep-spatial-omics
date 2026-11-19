"""Tumor vs Stroma metabolites, with the model's R² next to them.

Steps:
1. Load the measured intensities and the Tumor/Stroma labels.
2. For each metabolite, compare the median in Tumor spots and in Stroma spots.
3. Rank the metabolites by how big that difference is.
4. Show the model's R² next to each one, to see how well the model explains it.
5. List the big changes that the model does NOT explain well (R² <= 0.3).
6. Save two figures: the top differences, and a plot of change vs R² that highlights
   those poorly explained big changes.

The labels are used only to look at differences, never for training.
m/z values are labels only, not compound names.

Usage (from the project root):
    python analysis/tumor_stroma.py <name>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_bc515, tic_normalize

DATA = ROOT / "data" / "BC_515_Section_1.h5ad"
TOP_N = 10


def main() -> None:
    parser = argparse.ArgumentParser(description="Tumor vs Stroma look at one MET-MAP run")
    parser.add_argument("run", help="folder name inside results/")
    args = parser.parse_args()

    RUN = ROOT / "results" / args.run
    OUT = ROOT / "results" / "biology" / args.run
    if not (RUN / "outputs.npz").exists():
        raise SystemExit(f"No outputs.npz found in {RUN}. Check the folder name.")
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load the data
    coords, msi, annotation, mz = load_bc515(DATA)
    log = tic_normalize(msi)                    # same scale as the model
    r2 = np.load(RUN / "outputs.npz")["r2"]     # the model's R² for each metabolite
    tumor = annotation == "Tumor"
    stroma = annotation == "Stroma"
    print(f"Tumor spots: {tumor.sum()} | Stroma spots: {stroma.sum()}")

    # 2. Median in each region, and the difference between them
    table = pd.DataFrame({
        "feature_index": np.arange(len(mz)),
        "mz": mz.astype(float),
        "median_tumor": np.median(log[tumor], axis=0),
        "median_stroma": np.median(log[stroma], axis=0),
        "model_r2": r2,
    })
    # (+) x times higher in tumor, (-) x times higher in stroma, 0 the same in both tissues
    table["difference_tumor_minus_stroma"] = table["median_tumor"] - table["median_stroma"]

    # 3. Rank by the size of the difference, biggest first
    table["size"] = table["difference_tumor_minus_stroma"].abs()
    table = table.sort_values("size", ascending=False).reset_index(drop=True)
    table.to_csv(OUT / "tumor_stroma_table.csv", index=False)

    well_explained = 0.3

    top = table.head(TOP_N)
    print(f"\nTop {TOP_N} metabolites by difference between Tumor and Stroma:")
    print(top[["mz", "difference_tumor_minus_stroma", "model_r2"]].round(3).to_string(index=False))
    print(f"\nOf these {TOP_N}, {(top['model_r2'] > well_explained).sum()} are well explained by the model (R² > {well_explained}).")

    # Big changes that the model does NOT explain well
    poorly = table[table["model_r2"] <= well_explained].head(TOP_N)
    poorly.to_csv(OUT / "big_change_poorly_explained.csv", index=False)
    print(f"\nTop {TOP_N} with big differences but R² <= {well_explained} (not well explained):")
    print(poorly[["mz", "difference_tumor_minus_stroma", "model_r2"]].round(3).to_string(index=False))

    # Figure 1: the top metabolites, red = higher in Tumor, blue = higher in Stroma
    colors = np.where(top["difference_tumor_minus_stroma"] > 0, "#c0392b", "#2c6fbb")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([f"m/z {m:.3f}" for m in top["mz"]][::-1],
            top["difference_tumor_minus_stroma"][::-1], color=colors[::-1])
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Median log intensity: Tumor minus Stroma")
    ax.set_title("Metabolites that differ most between Tumor and Stroma")
    fig.tight_layout()
    fig.savefig(OUT / "top_differences.png", dpi=200)
    plt.close(fig)

    # Figure 2: change vs model R², one dot per metabolite, colored by category
    big_change = 1.0   # |difference| of 1 in log scale is about 2.7 times more (e^1)
    big = table["difference_tumor_minus_stroma"].abs() >= big_change
    good = table["model_r2"] > well_explained
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(table.loc[~big, "difference_tumor_minus_stroma"], table.loc[~big, "model_r2"],
               s=8, color="#bbbbbb", alpha=0.5, label="small change")
    ax.scatter(table.loc[big & good, "difference_tumor_minus_stroma"], table.loc[big & good, "model_r2"],
               s=14, color="#2e8b57", label=f"big change, well explained ({(big & good).sum()})")
    ax.scatter(table.loc[big & ~good, "difference_tumor_minus_stroma"], table.loc[big & ~good, "model_r2"],
               s=14, color="#c0392b", label=f"big change, poorly explained ({(big & ~good).sum()})")
    ax.axhline(well_explained, color="gray", linestyle="--", linewidth=1)
    ax.axvline(big_change, color="gray", linestyle=":", linewidth=1)
    ax.axvline(-big_change, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Median log intensity: Tumor minus Stroma")
    ax.set_ylabel("Model R²")
    ax.set_title("Which big changes does the model explain?")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "change_vs_r2.png", dpi=200)
    plt.close(fig)

    print(f"\nFiles written to {OUT}")


if __name__ == "__main__":
    main()
