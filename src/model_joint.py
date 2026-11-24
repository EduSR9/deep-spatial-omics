from __future__ import annotations

import torch
import torch.nn as nn


class JointSpatialMetabolicModel(nn.Module):
    """
    Joint autoencoder.

    Encoder:
        (x,y,metabolites) -> isodepths
    Decoder 1:
        isodepths -> (x,y)
     Decoder 2:
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

        # Encoder: (x,y,metabolites) -> d
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, n_isodepth),
        )

        # Decoder for coordinates: d -> (x,y)
        self.coord_decoder = nn.Sequential(
            nn.Linear(n_isodepth, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 2),
        )

        # Decoder for metabolites: d -> metabolites
        self.met_decoder = nn.Sequential(
            nn.Linear(n_isodepth, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, n_features),
        )

    def forward(
        self,
        coords: torch.Tensor,
        metabolites: torch.Tensor,
    ):


        # concatenate: [x,y] + metabolite vector
        x = torch.cat(
            [coords, metabolites],
            dim=1,
        )

        # Encode to isodepths
        isodepths = self.encoder(x)

        # Reconstruct coordinates
        coords_recon = self.coord_decoder(isodepths)
        # Reconstruct metabolites
        metabolites_recon = self.met_decoder(isodepths)

        # Model returns 2 reconstructions and the isodepths
        return (
            coords_recon,
            metabolites_recon,
            isodepths,
        )