from __future__ import annotations

import torch
import torch.nn as nn


class PCAMetMap(nn.Module):

    def __init__(
        self,
        n_pcs: int = 3, # number of principal components created from metabolites
        hidden_dim: int = 120, # hidden dimension of the encoder
        n_isodepth: int = 3, # number of isodepths
        n_features: int = 2086, # number of metabolites
    ) -> None:

        super().__init__()

        # Input size: 2 coordinates + number of principal components (3)
        input_dim = 2 + n_pcs

        # Encoder: (x,y,pcs) -> isodepths
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, n_isodepth),
        )

        # Decoder: isodepths -> metabolites
        self.decoder = nn.Linear(
            n_isodepth,
            n_features,
        )

    def forward(
        self,
        coords: torch.Tensor,
        pcs: torch.Tensor,
    ):
        
        # Concatenate coordinates and principal components
        x = torch.cat(
            [coords, pcs],
            dim=1,
        )

        # Encode to isodepths
        isodepths = self.encoder(x)

        # Reconstruct metabolites
        reconstruction = self.decoder(isodepths)

        # Model returns both the reconstruction and the isodepths
        return (
            reconstruction,
            isodepths,
        )