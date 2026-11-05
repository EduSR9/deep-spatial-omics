"""Run MET-MAP end to end: load -> preprocess -> train -> evaluate -> plot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Lets `python src/main.py` find the `src` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import yaml

from src.cv import evaluate_splits, summarize
from src.data import load_bc515, select_top_metabolites, standardize, tic_normalize
from src.dummy_data import create_dummy_bc515
from src.evaluate import rank_metabolites, reconstruction_r2, scaled_loadings
from src.model import MetMapMLP
from src.splits import (
    coverage_table,
    coverage_warnings,
    random_splits,
    region_labels,
    spatial_folds,
)
from src.train import train_model
from src.viz import (
    plot_annotations,
    plot_gradient_profile,
    plot_isodepths,
    plot_spatial_folds,
    plot_top_metabolites,
    plot_training_loss,
)

MODEL_REGISTRY = {
    "metmap": MetMapMLP,
}

CLI_OVERRIDES = {
    "n_features": ("model", "n_features"),
    "n_isodepth": ("model", "n_isodepth"),
    "hidden_dim": ("model", "hidden_dim"),
    "epochs": ("training", "epochs"),
    "learning_rate": ("training", "learning_rate"),
    "lasso_lambda": ("training", "lasso_lambda"),
    "seed": ("training", "seed"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MET-MAP on BC-515")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--exp-name", default="default", help="Results go to results/<exp-name>/")
    parser.add_argument("--dummy", action="store_true", help="Use synthetic data instead of the .h5ad file")
    parser.add_argument("--n-features", type=int)
    parser.add_argument("--n-isodepth", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lasso-lambda", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cv", action="store_true",
                        help="Run the evaluation protocol (random splits + spatial CV) instead of the full-data fit")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for arg_name, (section, key) in CLI_OVERRIDES.items():
        value = getattr(args, arg_name)
        if value is not None:
            cfg[section][key] = value
    return cfg


def set_deterministic(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def run_cv(
    cfg: dict,
    out_dir: Path,
    coords: np.ndarray,
    annotation: np.ndarray,
    msi: np.ndarray,
) -> None:
    """Evaluation protocol: random 80/20 repeats and spatial k-fold, a fresh model per split."""
    eval_cfg = cfg["evaluation"]
    seed = cfg["training"]["seed"]
    model_cls = MODEL_REGISTRY[cfg["model"]["type"]]
    n_spots = coords.shape[0]

    # Check coverage before training, so a bad split is caught early.
    regions = region_labels(annotation, coords, eval_cfg["interface_neighbors"])
    fold_ids = spatial_folds(coords, eval_cfg["n_spatial_folds"], seed=seed)
    table = coverage_table(regions, fold_ids)
    table.to_csv(out_dir / "coverage.csv")
    for warning in coverage_warnings(table, eval_cfg["min_fold_fraction"]):
        print(f"WARNING: {warning}")
    plot_spatial_folds(coords, fold_ids, regions, out_dir / "spatial_folds.png")

    random = random_splits(n_spots, eval_cfg["n_random_repeats"], eval_cfg["test_fraction"], seed)
    spatial = [
        (np.flatnonzero(fold_ids != k), np.flatnonzero(fold_ids == k))
        for k in range(eval_cfg["n_spatial_folds"])
    ]

    results = pd.concat([
        evaluate_splits(model_cls, cfg["model"], cfg["training"], coords, msi,
                        random, regions, "random"),
        evaluate_splits(model_cls, cfg["model"], cfg["training"], coords, msi,
                        spatial, regions, "spatial"),
    ], ignore_index=True)
    results.to_csv(out_dir / "cv_results.csv", index=False)

    summary = summarize(results)
    summary.to_csv(out_dir / "cv_summary.csv")
    print(summary.to_string())
    print(f"CV results in {out_dir}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    out_dir = Path("results") / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    set_deterministic(cfg["training"]["seed"])

    # Data
    if args.dummy:
        coords, msi, annotation, mz = create_dummy_bc515()
    else:
        coords, msi, annotation, mz = load_bc515(cfg["data"]["filepath"])
    if args.cv:
        run_cv(cfg, out_dir, coords, annotation, msi)
        return

    msi_log = tic_normalize(msi)
    n_features = cfg["model"]["n_features"]
    if n_features is not None:
        keep = select_top_metabolites(msi_log, n_features)
        msi_log, mz = msi_log[:, keep], mz[keep]
    coords_t = torch.tensor(standardize(coords), dtype=torch.float32)
    targets_t = torch.tensor(standardize(msi_log), dtype=torch.float32)
    print(f"Spots: {coords_t.shape[0]} | metabolites: {targets_t.shape[1]}")

    # Model
    model_cls = MODEL_REGISTRY[cfg["model"]["type"]]
    model = model_cls(
        n_features=targets_t.shape[1],
        hidden_dim=cfg["model"]["hidden_dim"],
        n_isodepth=cfg["model"]["n_isodepth"],
    )

    # Train
    total_losses, l1_losses = train_model(model, coords_t, targets_t, cfg["training"])

    # Evaluate
    model.eval()
    with torch.no_grad():
        prediction, isodepths = model(coords_t)
    prediction, isodepths = prediction.numpy(), isodepths.numpy()
    target = targets_t.numpy()
    weights = model.decoder.weight.detach().numpy()

    r2 = reconstruction_r2(target, prediction)
    loadings = scaled_loadings(weights, isodepths)
    ranking = rank_metabolites(mz, r2, loadings)
    final_mse = float(np.mean((prediction - target) ** 2))

    # Save
    ranking.to_csv(out_dir / "metabolite_ranking.csv", index=False)
    np.savez_compressed(
        out_dir / "outputs.npz",
        coords=coords, mz=mz, isodepths=isodepths, weights=weights,
        reconstruction=prediction, r2=r2, total_loss=total_losses, l1_loss=l1_losses,
    )
    torch.save(model.state_dict(), out_dir / "model_state_dict.pt")
    summary = {"n_spots": len(coords), "n_features": len(mz), "final_mse": final_mse,
               "median_r2": float(np.nanmedian(r2))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Plots
    top_idx = ranking["feature_index"].to_numpy()[:3]
    best = int(top_idx[0])
    plot_annotations(coords, annotation, out_dir / "annotations.png")
    plot_training_loss(total_losses, l1_losses, out_dir / "training_loss.png")
    plot_isodepths(coords, isodepths, out_dir / "isodepths.png")
    plot_top_metabolites(coords, msi_log, target, prediction, r2, top_idx, mz,
                         out_dir / "top_metabolites.png")
    plot_gradient_profile(isodepths[:, int(loadings[best].argmax())], msi_log[:, best],
                          out_dir / "gradient_profile.png")

    print(f"Final MSE: {final_mse:.4f} | results in {out_dir}")


if __name__ == "__main__":
    main()
