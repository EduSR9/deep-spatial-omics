"""Train/test splits for evaluation: random repeats and spatial k-fold, plus region coverage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_splits(
    n_spots: int,
    n_repeats: int = 5,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Independent random train/test splits, one per repeat.

    Returns
    -------
    list of (train_idx, test_idx) pairs, each an np.ndarray of spot indices.
    """
    # TODO: n_test = round(n_spots * test_fraction)
    # TODO: for each repeat r, perm = np.random.default_rng(seed + r).permutation(n_spots)
    # TODO: append (np.sort(perm[n_test:]), np.sort(perm[:n_test]))
    raise NotImplementedError


def spatial_folds(coords: np.ndarray, n_folds: int = 5, seed: int = 0, n_iter: int = 100) -> np.ndarray:
    """Assign each spot to a spatial block with k-means on (x, y).

    Each fold is a contiguous patch of tissue, so a held-out fold tests prediction
    in an area the model never saw.

    Returns
    -------
    np.ndarray of int, shape (n_spots,), values in 0..n_folds-1
    """
    # TODO: pick n_folds random spots as initial centers (seeded rng)
    # TODO: loop up to n_iter: assign each spot to nearest center, recompute centers, stop when unchanged
    # TODO: return the fold label per spot
    raise NotImplementedError


def region_labels(annotation: np.ndarray, coords: np.ndarray, n_neighbors: int = 6) -> np.ndarray:
    """Tumor, Stroma, Interface or Unlabelled for each spot.

    A Tumor or Stroma spot is "Interface" if at least one of its n_neighbors nearest
    spots carries the other label. Used for coverage checks only, never for training.

    Returns
    -------
    np.ndarray of str, shape (n_spots,)
    """
    # TODO: find the n_neighbors nearest spots for each spot (exclude the spot itself)
    # TODO: mark Tumor spots with a Stroma neighbour, and Stroma spots with a Tumor neighbour, as "Interface"
    # TODO: all other labels stay as in annotation
    raise NotImplementedError


def coverage_table(regions: np.ndarray, fold_ids: np.ndarray) -> pd.DataFrame:
    """Spot counts per region in each fold. Rows are folds, columns are regions."""
    # TODO: pd.crosstab of fold_ids against regions
    raise NotImplementedError


def coverage_warnings(table: pd.DataFrame, min_fraction: float = 0.5) -> list[str]:
    """Warnings for regions that are under-represented in any fold.

    A region is flagged when a fold holds less than min_fraction of the count it would
    have under an even split. Unlabelled spots are ignored. A region with no spots at
    all is also flagged.
    """
    # TODO: for Tumor, Stroma and Interface (even if missing from table, count as 0)
    # TODO: if the region has no spots at all, add a warning and continue
    # TODO: expected = total / n_folds; warn for each fold with count < min_fraction * expected
    raise NotImplementedError
