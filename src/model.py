"""Spatial models: coordinates -> isodepths -> metabolites.

ADDING A MODEL
1. Write the class below, under "NEW MODELS GO HERE". It needs:
   - __init__(n_features, hidden_dim=120, n_isodepth=3)
   - forward(coords) -> (prediction, isodepths)
   - penalty(lasso_lambda): extra loss term, or None
   - decoder_weights(): (n_features, n_isodepth) array, or None
   - class attribute needs_feature_subset = True/False
2. Add one line to MODEL_REGISTRY.
3. Add one entry under "models:" in config.yaml.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MetMapMLP(nn.Module):
    """Encode coordinates into isodepths and reconstruct signals with a linear decoder."""

    needs_feature_subset = False

    def __init__(self, n_features: int, hidden_dim: int = 120, n_isodepth: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_isodepth),
        )
        self.decoder = nn.Linear(n_isodepth, n_features)

    def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        isodepths = self.encoder(coords)
        reconstruction = self.decoder(isodepths)
        return reconstruction, isodepths

    def penalty(self, lasso_lambda: float) -> torch.Tensor:
        """L1 on the decoder weights (the MET-MAP lasso)."""
        return lasso_lambda * torch.norm(self.decoder.weight, 1)

    def decoder_weights(self) -> np.ndarray:
        return self.decoder.weight.detach().cpu().numpy()


# NEW MODELS GO HERE (see the top of this file)


class AttentionMetMap(MetMapMLP):
    """Single-head graph attention over the nearest training spots in isodepth space."""

    def __init__(self, n_features, hidden_dim=120, n_isodepth=3, k=16):
        super().__init__(n_features, hidden_dim, n_isodepth)
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k
        self.projection = nn.Linear(n_isodepth, n_isodepth, bias=False)
        self.edge_score = nn.Linear(2 * n_isodepth, 1, bias=False)

    def forward(self, coords, reference_coords, self_indices=None):
        depths = self.encoder(coords)
        reference = self.encoder(reference_coords)
        # Distances select graph edges; only selected edges receive learned scores.
        with torch.no_grad():
            distances = torch.cdist(depths, reference, compute_mode="donot_use_mm_for_euclid_dist")
            available = len(reference)
            if self_indices is not None:
                distances.scatter_(1, self_indices[:, None], float("inf"))
                available -= 1
            if available < 1:
                raise ValueError("Each query needs at least one other reference spot")
            indices = distances.topk(min(self.k, available), largest=False).indices

        neighbors = self.projection(reference)[indices]
        query = self.projection(depths).unsqueeze(1).expand_as(neighbors)
        scores = self.edge_score(torch.cat((query, neighbors), dim=-1)).squeeze(-1)
        weights = torch.softmax(nn.functional.leaky_relu(scores, negative_slope=0.2), dim=1)
        context = (weights.unsqueeze(-1) * neighbors).sum(dim=1)
        return self.decoder(depths + context), depths

    def penalty(self, lasso_lambda):
        return None

    def decoder_weights(self):
        # Linear readout acts on contextualized depths, not the raw isodepth axes.
        return None


# Maps the "type" name used in config.yaml to the class that builds it.
# Add one line per new model: "my_model": MyModel,
MODEL_REGISTRY = {
    "metmap": MetMapMLP,
    "attention": AttentionMetMap,
}
