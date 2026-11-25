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
    # Encode coordinates into isodepths and reconstruct signals with a linear decoder.

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
        # L1 on the decoder weights (the MET-MAP lasso).
        return lasso_lambda * torch.norm(self.decoder.weight, 1)

    def decoder_weights(self) -> np.ndarray:
        return self.decoder.weight.detach().cpu().numpy()


# NEW MODELS GO HERE (see the top of this file)


class AttentionMetMap(MetMapMLP):
    def __init__(self, n_features, hidden_dim=120, n_isodepth=3, k=64):
        super().__init__(n_features, hidden_dim, n_isodepth)
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k
        self.projection = nn.Linear(n_isodepth, n_isodepth, bias=False)
        self.edge_score = nn.Linear(2 * n_isodepth, 1, bias=False)
        self.register_buffer("reference_coords", None, persistent=False)

    def forward(self, coords):
        # Keep this fold's training spots as the reference set for evaluation.
        if self.training:
            self.reference_coords = coords.detach()
        if self.reference_coords is None:
            raise RuntimeError("Training coordinates are required before prediction")

        depths = self.encoder(coords)
        reference_depths = self.encoder(self.reference_coords)

        with torch.no_grad():
            distances = torch.cdist(
                depths, reference_depths, compute_mode="donot_use_mm_for_euclid_dist"
            )
            # Exclude each spot from its own neighborhood during training and reconstruction.
            same_spots = torch.equal(coords, self.reference_coords)
            if same_spots:
                distances.fill_diagonal_(float("inf"))
            available = len(reference_depths) - int(same_spots)
            if available < 1:
                raise ValueError("Each query needs at least one other reference spot")
            indices = distances.topk(min(self.k, available), largest=False).indices

        neighbors = self.projection(reference_depths)[indices]
        query = self.projection(depths).unsqueeze(1).expand_as(neighbors)
        scores = self.edge_score(torch.cat((query, neighbors), dim=-1)).squeeze(-1)
        weights = torch.softmax(nn.functional.leaky_relu(scores, negative_slope=0.2), dim=1)
        context = (weights.unsqueeze(-1) * neighbors).sum(dim=1)
        return self.decoder(depths + context), depths

    def penalty(self, lasso_lambda):
        # AdamW handles regularization for Attention.
        return None

    def decoder_weights(self):
        # Context mixing prevents direct loadings on the raw isodepth axes.
        return None


# Maps the "type" name used in config.yaml to the class that builds it.
# Add one line per new model: "my_model": MyModel,
MODEL_REGISTRY = {
    "metmap": MetMapMLP,
    "attention": AttentionMetMap,
}
