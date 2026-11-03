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
    n_test = round(n_spots*test_fraction)

    splits = []
    # Run n_repeats
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r) # Getting a random seed
        perm = rng.permutation(n_spots) # Permutation
        train_idx = perm[n_test:]
        test_idx = perm[:n_test]
        splits.append((np.sort(train_idx),np.sort(test_idx)))

    return(splits)


def spatial_folds(coords: np.ndarray, n_folds: int = 5, seed: int = 0, n_iter: int = 100) -> np.ndarray:
    """Assign each spot to a spatial block with k-means on (x, y).

    Each fold is a contiguous patch of tissue, so a held-out fold tests prediction
    in an area the model never saw.

    Returns
    -------
    np.ndarray of int, shape (n_spots,), values in 0..n_folds-1
    """
    n_spots = coords.shape[0]
    rng = np.random.default_rng(seed)
    initial_indices = rng.choice(n_spots, size = n_folds, replace=False)
    centers = coords[initial_indices]
    for _ in range(n_iter):
        # Measuring the distance of every point to c (centers)
        dist = np.array([np.linalg.norm(coords - c, axis = 1) for c in centers])
        labels = np.argmin(dist, axis = 0)
        new_centers = np.empty_like(centers)
        for k in range(n_folds):
            new_centers[k] = coords[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
        if np.array_equal(new_centers, centers):
            break
        centers = new_centers
    return(labels)


def region_labels(annotation: np.ndarray, coords: np.ndarray, n_neighbors: int = 6) -> np.ndarray:
    """Tumor, Stroma, Interface or Unlabelled for each spot.

    A Tumor or Stroma spot is "Interface" if at least one of its n_neighbors nearest
    spots carries the other label. Used for coverage checks only, never for training.

    Returns
    -------
    np.ndarray of str, shape (n_spots,)
    """
    result = annotation.copy()

    for i in range(len(coords)):
        curr_spot = coords[i]

        # 1. Calculate distances and neighbors for each point
        dist = np.array(np.linalg.norm(coords - curr_spot, axis = 1))
        sort_dist = np.argsort(dist)
        neighbors_idx = sort_dist[1:n_neighbors + 1]

        # 2. Check tags with the original tags
        for j in neighbors_idx:
            if annotation[i] == "Tumor" and annotation[j] == "Stroma":
                result[i] = "Interface"
                break # After one is enough
            elif annotation[i] == "Stroma" and annotation[j] == "Tumor":
                result[i] = "Interface"
                break
    return result


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
