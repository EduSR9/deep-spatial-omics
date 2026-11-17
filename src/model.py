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


# Maps the "type" name used in config.yaml to the class that builds it.
# Add one line per new model: "my_model": MyModel,
MODEL_REGISTRY = {
    "metmap": MetMapMLP,
}
