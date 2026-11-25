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
from src.factory import (
    MEASURED_INPUT_MODELS,
    MODEL_REGISTRY,
    create_model,
    train_selected_model,
    predict_model,
    fit_pca,
    transform_pca,
)
from src.splits import (
    coverage_table,
    coverage_warnings,
    random_splits,
    region_labels,
    spatial_folds,
)

from src.viz import (
    plot_annotations,
    plot_gradient_profile,
    plot_isodepths,
    plot_model_boxplots,
    plot_spatial_folds,
    plot_top_metabolites,
    plot_training_loss,
)

# =====================================================
# COMMAND-LINE CONFIGURATION
# =====================================================

# Maps command-line arguments to locations in config.yaml.
# Values provided through the CLI override the default configuration.
CLI_OVERRIDES = {
    "model": ("model", "type"),
    "data": ("data", "filepath"),
    "n_features": ("model", "n_features"),
    "n_isodepth": ("model", "n_isodepth"),
    "n_pcs": ("model", "n_pcs"),
    "hidden_dim": ("model", "hidden_dim"),
    "epochs": ("training", "epochs"),
    "learning_rate": ("training", "learning_rate"),
    "lasso_lambda": ("training", "lasso_lambda"),
    "weight_decay": ("training", "weight_decay"),
    "seed": ("training", "seed"),
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Values supplied through the CLI override config.yaml,
    making experiments reproducible without editing the file.
    """
    parser = argparse.ArgumentParser(description="MET-MAP on BC-515")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=MODEL_REGISTRY)
    parser.add_argument("--data", help="Override the data file in the config")
    parser.add_argument("--exp-name", default="baseline",
                        help="Experiment folder under results/, or results/<model>/ for a single model")
    parser.add_argument("--dummy", action="store_true", help="Use synthetic data instead of the .h5ad file")
    parser.add_argument("--n-features", type=int)
    parser.add_argument("--n-isodepth", type=int)
    parser.add_argument("--n-pcs", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lasso-lambda", type=float)
    parser.add_argument("--weight-decay", type=float, help="> 0 trains with AdamW")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cv", action="store_true",
                        help="Compare the configured models, or only the model selected with --model")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """
    Load config.yaml and apply any command-line overrides.

    CLI arguments always take precedence over configuration values.
    """
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Single run with a named model (--model): use that model's entry from the "models:" list,
    # so its params and training overrides apply here too, as they do in --cv.
    # Command-line flags below still win over the config.
    if args.model and not args.cv:
        entry = next((m for m in cfg.get("models", []) if m["type"] == args.model), None)
        if entry is not None:
            cfg["model"].update(entry.get("params") or {})
            cfg["training"].update(entry.get("training") or {})
    apply_overrides(cfg, args)
    return cfg


def apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    for arg_name, (section, key) in CLI_OVERRIDES.items():
        value = getattr(args, arg_name)
        if value is not None:
            cfg[section][key] = value


def validate_features(model_cfg: dict) -> None:
    n_features = model_cfg["n_features"]
    if n_features is not None and n_features < 1:
        raise SystemExit("--n-features must be positive.")
    if model_cfg["type"] == "self_attention" and n_features is None:
        raise SystemExit("Set --n-features for self_attention (start with 20).")


def set_deterministic(seed: int) -> None:
    """Same seed, same result: fixes the random seeds and forces deterministic operations.
    One thread is slower, but parallel threads can add up numbers in a different order."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def run_cv(
    cfg: dict,
    args: argparse.Namespace,
    out_dir: Path,
    coords: np.ndarray,
    annotation: np.ndarray,
    msi: np.ndarray,
    mz: np.ndarray,
) -> None:
    """Evaluation protocol: random 80/20 repeats and spatial k-fold, a fresh model per split.

    Runs every model in cfg["models"]. Each model gets its own folder; the boxplot compares them.
    Shared outputs (coverage, fold map, boxplot) go in out_dir.
    """
    eval_cfg = cfg["evaluation"]
    seed = cfg["training"]["seed"]
    n_spots = coords.shape[0]

    # 1. Regions and spatial folds. Coverage is checked before training, so a bad split is caught early
    regions = region_labels(annotation, coords, eval_cfg["interface_neighbors"])
    fold_ids = spatial_folds(coords, eval_cfg["n_spatial_folds"], seed=seed)
    table = coverage_table(regions, fold_ids)
    table.to_csv(out_dir / "coverage.csv")
    for warning in coverage_warnings(table, eval_cfg["min_fold_fraction"]):
        print(f"WARNING: {warning}")
    plot_spatial_folds(coords, fold_ids, regions, out_dir / "spatial_folds.png")

    # 2. Train/test splits: random 80/20 repeats, and each spatial fold held out once
    random_split_list = random_splits(n_spots, eval_cfg["n_random_repeats"], eval_cfg["test_fraction"], seed)
    spatial_split_list = [
        (np.flatnonzero(fold_ids != k), np.flatnonzero(fold_ids == k))
        for k in range(eval_cfg["n_spatial_folds"])
    ]

    entries = cfg["models"]
    if args.model:
        entry = next((m for m in entries if m["type"] == args.model), None)
        entries = [entry or {"name": args.model, "type": args.model}]

    all_spots = np.arange(n_spots)
    for entry in entries:
        model_cfg = {**cfg["model"], **(entry.get("params") or {}), "type": entry["type"]}
        if args.model is None and "n_features" in eval_cfg:
            model_cfg["n_features"] = eval_cfg["n_features"]
        train_cfg = {**cfg["training"], **(entry.get("training") or {})}
        run_cfg = {**cfg, "model": model_cfg, "training": train_cfg, "data": cfg["data"].copy()}
        apply_overrides(run_cfg, args)
        validate_features(model_cfg)

        model_dir = out_dir / entry["name"]
        model_dir.mkdir(exist_ok=True)
        with (model_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(run_cfg, f, sort_keys=False)
        print(f"\n=== {entry['name']} ===", flush=True)

        random_results, random_features = evaluate_splits(
            model_cfg, train_cfg, coords, msi, mz, random_split_list, regions, "random",
        )
        spatial_results, spatial_features = evaluate_splits(
            model_cfg, train_cfg, coords, msi, mz, spatial_split_list, regions, "spatial",
        )
        results = pd.concat([random_results, spatial_results], ignore_index=True)
        results.to_csv(model_dir / "cv_results.csv", index=False)
        pd.concat([random_features, spatial_features], ignore_index=True).to_csv(
            model_dir / "cv_feature_r2.csv", index=False,
        )
        summary = summarize(results)
        summary.to_csv(model_dir / "cv_summary.csv")
        print(summary.to_string())

        _, reconstruction = evaluate_splits(
            model_cfg, train_cfg, coords, msi, mz, [(all_spots, all_spots)], regions, "reconstruction",
        )
        reconstruction.to_csv(model_dir / "reconstruction_r2.csv", index=False)

    plot_entries = [{**entry, "measured_input": entry["type"] in MEASURED_INPUT_MODELS} for entry in entries]
    plot_model_boxplots(out_dir, plot_entries)
    print(f"CV results in {out_dir} (boxplot: boxplot_r2.png)")


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    model_type = cfg["model"]["type"]
    if not args.cv:
        validate_features(cfg["model"])
    out_dir = Path("results")
    if not args.cv or args.model:
        out_dir /= model_type
    out_dir /= args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    set_deterministic(cfg["training"]["seed"])

    # 1. Data (synthetic with --dummy). With --cv, the evaluation protocol takes over from here
    if args.dummy:
        coords, msi, annotation, mz = create_dummy_bc515()
    else:
        coords, msi, annotation, mz = load_bc515(cfg["data"]["filepath"])
    if args.cv:
        run_cv(cfg, args, out_dir, coords, annotation, msi, mz)
        return

    # 2. Preprocess: TIC normalization, optional top-n metabolites, standardization
    msi_log = tic_normalize(msi)
    n_features = cfg["model"]["n_features"]
    keep = np.arange(len(mz))
    if n_features is not None:
        keep = select_top_metabolites(msi_log, n_features)
        msi_log, mz = msi_log[:, keep], mz[keep]
    coords_t = torch.tensor(standardize(coords), dtype=torch.float32)
    targets_t = torch.tensor(standardize(msi_log), dtype=torch.float32)

    pcs_t = None
    if cfg["model"]["type"] == "pca":
        pca = fit_pca(targets_t, cfg["model"]["n_pcs"],)
        pcs_t = transform_pca(targets_t, pca,)

    print(f"Spots: {coords_t.shape[0]} | metabolites: {targets_t.shape[1]}")

    # 3. Model
    model = create_model(
        cfg["model"],
        targets_t.shape[1],
    )

    # 4. Train
    training_results = train_selected_model(
        cfg["model"]["type"],
        model,
        coords_t,
        targets_t,
        cfg["training"],
        pcs_t,
    )
    total_losses = training_results["total_losses"]
    l1_losses = training_results.get("l1_losses", None,)
    coord_losses = training_results.get("coord_losses", None,)
    met_losses = training_results.get("met_losses", None,)

    # 5. Evaluate
    prediction, isodepths = predict_model(
        cfg["model"]["type"],
        model,
        coords_t,
        targets_t,
        pcs_t,
    )
    prediction, isodepths = prediction.numpy(), isodepths.numpy()
    target = targets_t.numpy()
    r2 = reconstruction_r2(target, prediction)
    linear_outputs = {}
    loadings = None
    weights = model.decoder_weights()
    if weights is not None:
        loadings = scaled_loadings(weights, isodepths)
        linear_outputs = {"weights": weights, "l1_loss": l1_losses}
    ranking = rank_metabolites(mz, r2, loadings)
    ranking["original_feature_index"] = keep[ranking["feature_index"].to_numpy()]
    final_mse = float(np.mean((prediction - target) ** 2))

    # 6. Save tables, outputs and the trained weights
    ranking.to_csv(out_dir / "metabolite_ranking.csv", index=False)
    joint_losses = {} if coord_losses is None else {"coord_losses": coord_losses, "met_losses": met_losses}
    np.savez_compressed(
        out_dir / "outputs.npz",
        coords=coords,
        mz=mz,
        isodepths=isodepths,
        reconstruction=prediction,
        r2=r2,
        total_loss=total_losses,
        feature_index=keep,  # Which metabolites of the dataset the model used
        **joint_losses,
        **linear_outputs,
    )
    torch.save(model.state_dict(), out_dir / "model_state_dict.pt")
    summary = {"n_spots": len(coords), "n_features": len(mz), "final_mse": final_mse,
               "median_r2": float(np.nanmedian(r2))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # 7. Figures
    top_idx = ranking["feature_index"].to_numpy()[:3]
    plot_annotations(coords, annotation, out_dir / "annotations.png")
    plot_training_loss(total_losses, l1_losses, out_dir / "training_loss.png")
    plot_isodepths(coords, isodepths, out_dir / "isodepths.png")
    plot_top_metabolites(coords, msi_log, target, prediction, r2, top_idx, mz,
                         out_dir / "top_metabolites.png")
    if loadings is not None:
        best = int(top_idx[0])
        plot_gradient_profile(isodepths[:, int(loadings[best].argmax())], msi_log[:, best],
                              out_dir / "gradient_profile.png")

    print(f"Final MSE: {final_mse:.4f} | results in {out_dir}")


if __name__ == "__main__":
    main()
