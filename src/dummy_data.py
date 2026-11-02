"""Small synthetic dataset with the same structure as BC-515, for running without the real file."""

from __future__ import annotations

import numpy as np


def create_dummy_bc515(
    n_spots: int = 200,
    n_metabolites: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (coords, msi, annotation, mz) shaped like load_bc515 output."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(n_spots, 2))
    # A smooth spatial trend per metabolite so the model has something to learn.
    trend = coords @ rng.normal(size=(2, n_metabolites)) / 100
    msi = np.exp(trend + rng.normal(scale=0.3, size=(n_spots, n_metabolites)))
    annotation = rng.choice(["Stroma", "Tumor", "Unlabelled"], size=n_spots, p=[0.77, 0.21, 0.02])
    mz = np.round(rng.uniform(100, 1000, size=n_metabolites), 4).astype(str)
    return coords, msi, annotation, mz
