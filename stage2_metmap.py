# %%
# BC-515 MET-MAP. AI-assisted implementation; see docs/GENAI_USE.md.
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import time

# Limit CPU threads before importing numerical libraries.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA = ROOT / "data/BC_515_Section_1.h5ad"
OUT = ROOT / "results/stage2_metmap"
EXPECTED_SHA256 = "1f96d6f1aaa36c117ccd7fb94b172a1c1435585d54279e8cffb93df0299c7204"
CONFIG = dict(seed=0, epochs=3000, K=3, hidden=[120, 120], lr=0.001,
              lasso_lambda=0.0001, end_factor=0.05, total_iters=12000,
              device="cpu", threads=1, batch_size=None)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

# %%
# Data: TIC normalization, log1p, then feature-wise standardization.
def load_data():
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256, "Input checksum differs from the course data."
    adata = ad.read_h5ad(DATA)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    raw = np.asarray(adata.uns["msi"], dtype=np.float64)
    mz = np.asarray(adata.uns["mz_features"]).reshape(-1).astype(str)
    spot_ids = adata.obs_names.to_numpy(dtype=str)
    assert raw.shape == (3508, 2086) and coords.shape == (3508, 2)
    assert len(mz) == raw.shape[1] and len(np.unique(spot_ids)) == len(coords)
    assert np.isfinite(raw).all() and (raw >= 0).all() and np.isfinite(coords).all()
    tic = raw.sum(axis=1)
    assert (tic > 0).all()
    log_signal = np.log1p(raw / tic[:, None] * tic.mean())
    spatial_scaler = StandardScaler().fit(coords)
    feature_scaler = StandardScaler().fit(log_signal)
    spatial = torch.tensor(spatial_scaler.transform(coords), dtype=torch.float32)
    target = torch.tensor(feature_scaler.transform(log_signal), dtype=torch.float32)
    print(f"Coordinates: {tuple(spatial.shape)}; MSI: {tuple(target.shape)}", flush=True)
    return dict(coords=coords, mz=mz, spot_ids=spot_ids, log_signal=log_signal,
                spatial=spatial, target=target, spatial_scaler=spatial_scaler,
                feature_scaler=feature_scaler, tic_mean=float(tic.mean()), sha256=digest)

# %%
# Model: coordinates -> three depths -> MSI reconstruction.
class METMAP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 120), nn.ReLU(),
            nn.Linear(120, 120), nn.ReLU(), nn.Linear(120, 3))
        self.decoder = nn.Linear(3, n_features)

    def forward(self, coords):
        return self.decoder(self.encoder(coords))

# %%
# Training: full-batch MSE plus decoder weight L1 penalty.
def train(spatial, target):
    torch.manual_seed(CONFIG["seed"])
    model = METMAP(target.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=CONFIG["end_factor"],
        total_iters=CONFIG["total_iters"])
    # Match the reference input-gradient path.
    spatial = spatial.detach().clone().requires_grad_(True)
    rows = []
    for epoch in range(CONFIG["epochs"]):
        optimizer.zero_grad()
        spatial.grad = None
        prediction = model(spatial)
        mse = nn.functional.mse_loss(prediction, target, reduction="mean")
        l1 = CONFIG["lasso_lambda"] * torch.norm(model.decoder.weight, 1)
        loss = mse + l1
        rows.append((epoch + 1, loss.item(), mse.item(), l1.item(),
                     optimizer.param_groups[0]["lr"]))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}: total={loss.item():.6f}, MSE={mse.item():.6f}", flush=True)
    history = pd.DataFrame(rows, columns=["epoch", "total_loss", "mse", "l1", "lr"])
    return model, history

# %%
# Outputs: training reconstruction only, not held-out evaluation.
def save_results(data, model, history, elapsed):
    OUT.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        depth = model.encoder(data["spatial"]).numpy()
        prediction = model(data["spatial"]).numpy()
    target = data["target"].numpy()
    assert np.isfinite(prediction).all() and np.isfinite(depth).all()
    observed64, predicted64 = target.astype(np.float64), prediction.astype(np.float64)
    residual = np.sum((predicted64 - observed64) ** 2, axis=0)
    total = np.sum((observed64 - observed64.mean(axis=0)) ** 2, axis=0)
    r2 = np.full(len(total), np.nan)
    valid = total > 0
    r2[valid] = 1 - residual[valid] / total[valid]
    metrics = pd.DataFrame(dict(feature_index=np.arange(len(total)), mz=data["mz"],
                                training_reconstruction_r2=r2))
    metrics.to_csv(OUT / "feature_metrics.csv", index=False)
    history.to_csv(OUT / "training_history.csv", index=False)
    np.savez_compressed(OUT / "outputs.npz", prediction=prediction, depth=depth,
                        coords=data["coords"], mz=data["mz"], spot_ids=data["spot_ids"])
    # Retain preprocessing parameters for model reuse.
    np.savez(OUT / "preprocessing.npz", tic_mean=data["tic_mean"],
             spatial_mean=data["spatial_scaler"].mean_, spatial_scale=data["spatial_scaler"].scale_,
             feature_mean=data["feature_scaler"].mean_, feature_scale=data["feature_scaler"].scale_)
    torch.save(model.state_dict(), OUT / "model_state.pt")
    summary = dict(config=CONFIG, input_sha256=data["sha256"],
                   final_training_mse=float(np.mean((prediction - target) ** 2)),
                   median_training_r2=float(np.nanmedian(r2)), elapsed_seconds=elapsed,
                   evaluation="Full-slice training reconstruction; no held-out evaluation",
                   python=platform.python_version(),
                   versions={p: importlib.metadata.version(p) for p in
                             ["torch", "numpy", "pandas", "scikit-learn", "anndata", "matplotlib"]},
                   course_commit="6352c2d6f7a24f729ceb30f4a89041f72fa785d0",
                   upstream_commit="43c9386ffe7d5b2f0f66d64d4820358c3cd8a11c")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ax = history.plot(x="epoch", y=["total_loss", "mse", "l1"], figsize=(7, 4))
    ax.set(xlabel="Epoch", ylabel="Loss", title="Full-slice training")
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / "training_loss.png", dpi=160)
    plt.close(ax.figure)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    scaled_depth = (depth - depth.min(axis=0)) / np.maximum(np.ptp(depth, axis=0), 1e-12)
    for k, ax in enumerate(axes):
        points = ax.scatter(*data["coords"].T, c=scaled_depth[:, k], s=5, vmin=0, vmax=1)
        ax.set(xlabel="x", ylabel="y", title=f"Learned depth {k+1}", aspect="equal")
        fig.colorbar(points, ax=ax, label="Display scale (0–1)", shrink=0.7)
    fig.savefig(OUT / "spatial_depths.png", dpi=160)
    plt.close(fig)

    # Display the three best fits, not a representative sample.
    top = metrics.sort_values("training_reconstruction_r2", ascending=False).head(3)
    predicted_log = data["feature_scaler"].inverse_transform(prediction)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for col, j in enumerate(top.feature_index):
        measured, fitted = data["log_signal"][:, j], predicted_log[:, j]
        lo, hi = min(measured.min(), fitted.min()), max(measured.max(), fitted.max())
        for row, (values, label) in enumerate([(measured, "Measured"), (fitted, "Reconstructed")]):
            ax = axes[row, col]
            points = ax.scatter(*data["coords"].T, c=values, s=5, vmin=lo, vmax=hi)
            ax.set(xlabel="x", ylabel="y", aspect="equal",
                   title=f"{label} | m/z {float(data['mz'][j]):.4f}")
            fig.colorbar(points, ax=ax, label="TIC log1p signal", shrink=0.65)
    fig.savefig(OUT / "reconstruction_examples.png", dpi=160)
    plt.close(fig)
    print(f"Training MSE: {summary['final_training_mse']:.6f}; saved to {OUT}", flush=True)

# %%
if __name__ == "__main__":
    data = load_data()
    started = time.perf_counter()
    model, history = train(data["spatial"], data["target"])
    save_results(data, model, history, time.perf_counter() - started)
