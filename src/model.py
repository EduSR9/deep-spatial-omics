"""MET-MAP model: spatial encoder to isodepths, linear decoder to metabolites."""

from __future__ import annotations

import torch
import torch.nn as nn


class MetMapMLP(nn.Module):
    """Encode (x, y) into isodepths, then decode linearly into metabolite intensities.

    Encoder: 2 -> hidden -> ReLU -> hidden -> ReLU -> n_isodepth (no final activation)
    Decoder: n_isodepth -> n_features, linear with bias

    Parameters
    ----------
    n_features : int
        Number of metabolites to reconstruct.
    hidden_dim : int
        Width of both hidden encoder layers.
    n_isodepth : int
        Number of latent spatial gradients.
    """

    def __init__(self, n_features: int, hidden_dim: int = 120, n_isodepth: int = 3) -> None:
        super().__init__()
        # TODO: self.encoder = nn.Sequential(Linear(2, hidden), ReLU, Linear(hidden, hidden), ReLU, Linear(hidden, n_isodepth))
        # TODO: self.decoder = nn.Linear(n_isodepth, n_features)  (keep the bias; only weights get L1)
        raise NotImplementedError

    def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, isodepths).

        Parameters
        ----------
        coords : torch.Tensor, shape (n_spots, 2)

        Returns
        -------
        reconstruction : torch.Tensor, shape (n_spots, n_features)
        isodepths : torch.Tensor, shape (n_spots, n_isodepth)
        """
        # TODO: isodepths = self.encoder(coords); reconstruction = self.decoder(isodepths)
        raise NotImplementedError
