"""MET-MAP model: spatial encoder to isodepths, linear decoder to metabolites."""

from __future__ import annotations

import torch
import torch.nn as nn

class MetMapMLP(nn.Module):
    """Encode coordinates into isodepths and reconstruct signals."""

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