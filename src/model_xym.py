from __future__ import annotations

import torch
import torch.nn as nn


class XYMMetMap(nn.Module):
    """
    Encoder:
        (x,y,m) -> isodepths

    Decoder:
        isodepths -> metabolites
    """

    def __init__(
        self,
        n_features: int, # number of metabolites
        hidden_dim: int = 120, # hidden dimension of the encoder
        n_isodepth: int = 3, # number of isodepths
    ) -> None:

        super().__init__()

        # Input size: 2 coordinates + metabolite dimension
        input_dim = n_features + 2

        # Encoder: (x,y,metabolites) -> isodepths
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
        metabolites: torch.Tensor,
    ):

        # Concatenate coordinates and metabolites
        x = torch.cat(
            [coords, metabolites],
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