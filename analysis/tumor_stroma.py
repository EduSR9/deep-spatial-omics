"""Tumor vs Stroma: what does MET-MAP explain well, what is it, and where is it in the tissue?

Steps:
    1. Load the measured intensities, the Tumor/Stroma labels and one trained MET-MAP run.
    2. For each metabolite, compare its median intensity in Tumor and in Stroma spots,
       and put the model R2 next to it.
    3. Give each big change a likely name (paper or HMDB) and a chemical class.
    4. Figures:
         change_vs_r2.png              all metabolites, change vs R2, well explained zone marked
         well_explained_identities.png zoom on the well explained big changes, with names
         isodepth_profiles.png         where those metabolites are along the tumor isodepth
    5. Summary of the biology: chemical classes higher in Tumor and higher in Stroma.

The Tumor/Stroma labels are only used here to look at the results. The model never saw them.

Run from the project root (after training a run with src/main.py):
    python analysis/tumor_stroma.py <run name>

Input:  results/<run name>/outputs.npz
Output: results/biology/<run name>/ (CSV tables and PNG figures)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch

from src.data import load_bc515, tic_normalize
from hmdb_candidates import hmdb_path, paper_annotation, paper_path, search_hmdb

# Settings
default_data_path = root / "data" / "BC_515_Section_1.h5ad"  # Used if the run has no config.yaml

big_change = 1.0          # |Tumor - Stroma| in log scale that counts as big (1.0 is about 2.7x)
well_explained = 0.6      # A metabolite with model R2 above this is "well explained"
n_bins = 20               # Number of columns in the isodepth heatmap

tumor_color = "#c0392b"
stroma_color = "#2c6fbb"
class_colors = {
    "lipid": "#8e44ad",
    "fatty acid": "#d35400",
    "sugar": "#16a085",
    "other": "#2c3e50",
    "unknown": "#bdc3c7",
}
lipid_names = {"PI", "PE", "PG", "PS", "PC", "PA", "CL", "LPE", "LPI", "LPC", "LPA", "LPG", "LPS", "PI-Cer",
               "Cer", "SHexCer", "SM", "DG", "TG"}
max_label = 35            # Long HMDB names are cut to this length in the plots


def build_difference_table(log: np.ndarray, tumor: np.ndarray, stroma: np.ndarray,
                           mz: np.ndarray, r2: np.ndarray) -> pd.DataFrame:
    """One row per metabolite: median in Tumor and Stroma, their difference and the model R2.
    Positive difference = higher in Tumor. Sorted by the size of the difference."""
    table = pd.DataFrame({
        "feature_index": np.arange(len(mz)),
        "mz": mz.astype(float),
        "median_tumor": np.median(log[tumor], axis=0),
        "median_stroma": np.median(log[stroma], axis=0),
        "model_r2": r2,
    })
    table["difference_tumor_minus_stroma"] = table["median_tumor"] - table["median_stroma"]
    table["size"] = table["difference_tumor_minus_stroma"].abs()
    return table.sort_values("size", ascending=False).reset_index(drop=True)


def chemical_class(name: str, hmdb_class: str = "") -> str:
    """Simple chemical class from the lipid name (e.g. PI 34:1, PS(20:0/...)) or the HMDB class."""
    head = name.split("(")[0].split(" ")[0]  # "PI 34:1" -> "PI", "PS(20:0/...)" -> "PS"
    if head == "FA" or hmdb_class == "Fatty Acyls":
        return "fatty acid"
    if head in lipid_names or hmdb_class in ("Glycerophospholipids", "Sphingolipids", "Glycerolipids"):
        return "lipid"
    if name == "Hexose" or "Carbohydrate" in hmdb_class or hmdb_class == "Organooxygen compounds":
        return "sugar"
    return "other"


def identify(big_rows: pd.DataFrame, paper: pd.DataFrame | None, hmdb: pd.DataFrame | None) -> pd.DataFrame:
    """Give each big change a name: the paper's annotation if there is one, otherwise the best
    HMDB candidate, otherwise just the m/z. HMDB names are only mass matches:
    "?" = within 10 ppm, "??" = only between 10 and 50 ppm (less reliable)."""
    rows = []
    for _, r in big_rows.iterrows():
        mz = r["mz"]
        name, source, chem = f"m/z {mz:.3f}", "unknown", "unknown"

        ann = paper_annotation(mz, paper)
        if ann is not None:
            name, source, chem = ann["name"], "paper", chemical_class(ann["name"])
        elif hmdb is not None:
            # search_hmdb gives the candidates within 10 ppm, or if there are none,
            # the closest ones up to 50 ppm (wide_match). Already sorted, so the first is the best
            cand = search_hmdb(mz, hmdb)
            if len(cand):
                best = cand.iloc[0]
                mark = "??" if best["wide_match"] else "?"
                hmdb_class = str(best["class"]) if pd.notna(best["class"]) else ""
                name = f"{best['name']}{mark}"
                source = "HMDB 10-50 ppm" if best["wide_match"] else "HMDB"
                chem = chemical_class(best["name"], hmdb_class)

        rows.append({
            "feature_index": int(r["feature_index"]),
            "mz": mz,
            "difference_tumor_minus_stroma": r["difference_tumor_minus_stroma"],
            "model_r2": r["model_r2"],
            "well_explained": r["model_r2"] > well_explained,
            "identity": name,
            "source": source,
            "chemical_class": chem,
        })
    return pd.DataFrame(rows)


def short(name: str) -> str:
    """Cut long names so the plot labels stay readable."""
    return name if len(name) <= max_label else name[:max_label - 4] + "...?"


def plot_change_vs_r2(table: pd.DataFrame, path: Path) -> None:
    """All metabolites: Tumor - Stroma difference vs model R2.
    The shaded zones are the big changes that the model explains well."""
    table = table.dropna(subset=["model_r2"])  # Metabolites the model did not use have no R2
    diff = table["difference_tumor_minus_stroma"]
    r2 = table["model_r2"]
    big = diff.abs() >= big_change
    good = r2 > well_explained

    fig, ax = plt.subplots(figsize=(7, 5))
    x_max = diff.abs().max() * 1.05
    for x0, x1 in [(big_change, x_max), (-x_max, -big_change)]:
        ax.fill_between([x0, x1], well_explained, 1.02, color="#2e8b57", alpha=0.1, linewidth=0)
    ax.scatter(diff[~big], r2[~big], s=8, color="#bbbbbb", alpha=0.5, label="small change")
    ax.scatter(diff[big & good], r2[big & good], s=14, color="#2e8b57",
               label=f"big change, well explained ({(big & good).sum()})")
    ax.scatter(diff[big & ~good], r2[big & ~good], s=14, color=tumor_color,
               label=f"big change, poorly explained ({(big & ~good).sum()})")
    ax.axhline(well_explained, color="gray", linestyle="--", linewidth=1)
    ax.axvline(big_change, color="gray", linestyle=":", linewidth=1)
    ax.axvline(-big_change, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Tumor - Stroma (median log intensity)")
    ax.set_ylabel("Model R2")
    ax.legend(loc="lower center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_identities(ids: pd.DataFrame, path: Path) -> None:
    """Zoom on the well explained big changes: one row each, sorted by the difference,
    labeled with its identity and colored by chemical class."""
    ids = ids.sort_values("difference_tumor_minus_stroma")
    colors = [class_colors[c] for c in ids["chemical_class"]]

    fig, ax = plt.subplots(figsize=(7, 0.18 * len(ids) + 1.5))
    y = np.arange(len(ids))
    ax.hlines(y, 0, ids["difference_tumor_minus_stroma"], color=colors, linewidth=1)
    ax.scatter(ids["difference_tumor_minus_stroma"], y, c=colors, s=25, zorder=3)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([short(n) for n in ids["identity"]], fontsize=6)
    ax.set_xlabel("Tumor - Stroma (median log intensity)")
    present = [c for c in class_colors if c in set(ids["chemical_class"])]
    ax.legend(handles=[Patch(color=class_colors[c], label=c) for c in present],
              loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_isodepth_profiles(ids: pd.DataFrame, iso: np.ndarray, log: np.ndarray,
                           tumor: np.ndarray, stroma: np.ndarray, path: Path) -> None:
    """Heatmap of the given metabolites along the isodepth that best follows Tumor/Stroma.
    Columns go from the stroma side to the tumor side, yellow = where a metabolite peaks."""
    # 1. Isodepth most correlated with Tumor, flipped so tumor is high (its sign means nothing)
    labelled = tumor | stroma
    corr = [np.corrcoef(iso[labelled, k], tumor[labelled])[0, 1] for k in range(iso.shape[1])]
    k = int(np.argmax(np.abs(corr)))
    depth = iso[:, k] * np.sign(corr[k])

    # 2. Split the spots into groups of equal size along it
    group = pd.qcut(depth, q=n_bins, labels=False, duplicates="drop")
    n_groups = group.max() + 1

    # 3. Mean intensity per metabolite and group, then scale each row to 0-1
    idx = ids["feature_index"].to_numpy().astype(int)
    profile = np.array([log[group == g][:, idx].mean(axis=0) for g in range(n_groups)]).T
    low = profile.min(axis=1, keepdims=True)
    profile = (profile - low) / (np.ptp(profile, axis=1, keepdims=True) + 1e-9)

    # 4. Sort the rows by their peak column (the row mean only breaks ties)
    order = np.argsort(profile.argmax(axis=1) + profile.mean(axis=1) * 1e-3)
    tumor_share = [tumor[group == g].sum() / max(labelled[group == g].sum(), 1) for g in range(n_groups)]

    # 5. Draw: Tumor share on top, heatmap below, labels colored by chemical class
    height = 0.16 * len(idx)
    fig, (ax_top, ax) = plt.subplots(2, 1, figsize=(8, height + 2.5), sharex=True,
                                     gridspec_kw={"height_ratios": [1, height]})
    ax_top.plot(np.arange(n_groups), tumor_share, color=tumor_color)
    ax_top.set_ylabel("Tumor\nshare", fontsize=8)
    ax_top.set_ylim(0, 1)

    ax.imshow(profile[order], aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([short(n) for n in ids["identity"].to_numpy()[order]], fontsize=5)
    for tick, chem in zip(ax.get_yticklabels(), ids["chemical_class"].to_numpy()[order]):
        tick.set_color(class_colors[chem])
    # Axis values: median isodepth of each column, scaled 0-1 like the map in isodepths.png
    scaled = (iso[:, k] - iso[:, k].min()) / (np.ptp(iso[:, k]) + 1e-12)
    ticks = np.linspace(0, n_groups - 1, 5).round().astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{np.median(scaled[group == g]):.2f}" for g in ticks])
    ax.set_xlabel(f"Isodepth {k + 1}")
    present = [c for c in class_colors if c in set(ids["chemical_class"])]
    ax.legend(handles=[Patch(color=class_colors[c], label=c) for c in present],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def biology_summary(ids: pd.DataFrame) -> pd.DataFrame:
    """Count the chemical classes of the well explained big changes, Tumor-high vs Stroma-high."""
    ids = ids.assign(direction=np.where(ids["difference_tumor_minus_stroma"] > 0,
                                        "higher in tumor", "higher in stroma"))
    return (ids.groupby(["direction", "chemical_class"])
               .agg(n=("identity", "size"), examples=("identity", lambda s: ", ".join(s.head(5))))
               .reset_index()
               .sort_values(["direction", "n"], ascending=[True, False]))


def run_data_path(run: Path) -> Path:
    """Data file used to train this run (from its config.yaml), or BC-515 if it is not there."""
    config_path = run / "config.yaml"
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text())
        filepath = cfg.get("data", {}).get("filepath")
        if filepath:
            return root / filepath
    return default_data_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tumor vs Stroma look at one MET-MAP run")
    parser.add_argument("run", help="folder name inside results/")
    args = parser.parse_args()

    run = root / "results" / args.run
    out = root / "results" / "biology" / args.run
    if not (run / "outputs.npz").exists():
        raise SystemExit(f"No outputs.npz found in {run}. Check the folder name.")
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load the same data the model was trained on, and the outputs of the trained model
    coords, msi, annotation, mz = load_bc515(run_data_path(run))
    log = tic_normalize(msi)  # Same scale used to train the model
    outputs = np.load(run / "outputs.npz")
    iso = outputs["isodepths"]

    # Put each R2 next to its metabolite. Models trained on fewer metabolites leave the rest
    # empty (NaN). Old runs without feature_index used all of them.
    used = outputs["feature_index"] if "feature_index" in outputs else np.arange(len(mz))
    r2 = np.full(len(mz), np.nan)
    r2[used] = outputs["r2"]
    print(f"Model R2 available for {len(used)} of {len(mz)} metabolites")
    tumor = annotation == "Tumor"
    stroma = annotation == "Stroma"
    print(f"Tumor spots: {tumor.sum()} | Stroma spots: {stroma.sum()}")

    # 2. Tumor vs Stroma table with the model R2
    table = build_difference_table(log, tumor, stroma, mz, r2)
    table.to_csv(out / "tumor_stroma_table.csv", index=False)

    # 3. Identities of the big changes (paper and HMDB are optional)
    paper = pd.read_csv(paper_path) if paper_path.exists() else None
    hmdb = pd.read_csv(hmdb_path, low_memory=False) if hmdb_path.exists() else None
    if paper is None:
        print(f"No paper reference at {paper_path}, paper names skipped.")
    if hmdb is None:
        print(f"No HMDB table at {hmdb_path}, HMDB names skipped.")
    big_rows = table[table["size"] >= big_change]
    ids = identify(big_rows, paper, hmdb)
    ids.to_csv(out / "big_change_identities.csv", index=False)
    good_ids = ids[ids["well_explained"]]
    print(f"\nBig changes: {len(ids)} | well explained (R2 > {well_explained}): {len(good_ids)}")
    print(ids.groupby("source").size().to_string())

    # 4. Figures
    plot_change_vs_r2(table, out / "change_vs_r2.png")
    if len(good_ids) == 0:
        print("No big change is well explained by this model, identity and isodepth figures skipped.")
        return
    plot_identities(good_ids, out / "well_explained_identities.png")
    plot_isodepth_profiles(good_ids, iso, log, tumor, stroma, out / "isodepth_profiles.png")

    # 5. Biology summary
    summary = biology_summary(good_ids)
    summary.to_csv(out / "biology_summary.csv", index=False)
    print("\nWell explained big changes by chemical class:")
    print(summary.to_string(index=False))

    print(f"\nFiles written to {out}")


if __name__ == "__main__":
    main()
