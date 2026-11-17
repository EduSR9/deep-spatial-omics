"""Models: spatial encoder to isodepths, then a decoder from isodepths to metabolites.

Every model in this file follows the same contract, so training, evaluation and plotting
never need to know which model they are running.

HOW TO ADD A NEW MODEL
======================

Only this file and config.yaml need to change.

STEP 1. Write your class in the section marked "NEW MODELS GO HERE" below
        (after MetMapMLP, before MODEL_REGISTRY). It must have these 4 methods:

  a) __init__(self, n_features, hidden_dim=120, n_isodepth=3)
       builds your layers. Reuse the MetMapMLP encoder if you want the isodepths
       computed the same way.

  b) forward(self, coords) -> (prediction, isodepths)
       prediction: shape (n_spots, n_features)
       isodepths:  shape (n_spots, n_isodepth)

  c) penalty(self, lasso_lambda)
       return an extra loss term (a torch scalar), or None if you have none.

  d) decoder_weights(self)
       return a numpy array of shape (n_features, n_isodepth) if you have a
       linear decoder, or None if you don't.

  and one class attribute:
       needs_feature_subset = True   if your model can't run on all 2,086 metabolites
                                     (for example, it uses too much memory)
       needs_feature_subset = False  if it can

  Copying a class from elsewhere is NOT enough: it must have (a)-(d) and the
  attribute above, or training will crash.

  Example skeleton (replace the "..." with your code):

    class MyModel(nn.Module):
        needs_feature_subset = True

        def __init__(self, n_features, hidden_dim=120, n_isodepth=3):
            super().__init__()
            ...  # your layers

        def forward(self, coords):
            ...  # your computation
            return prediction, isodepths

        def penalty(self, lasso_lambda):
            return None

        def decoder_weights(self):
            return None

STEP 2. Register it in MODEL_REGISTRY at the bottom of this file. Add one line:
            "my_model": MyModel,
        "my_model" is the name you will use in config.yaml.

STEP 3. Add one entry under "models:" in config.yaml (see the comments there).

Then test it:
    python src/main.py --dummy --model my_model --exp-name test_my_model
    python src/main.py --dummy --cv --exp-name test_cv
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


# ==============================================================================================
# NEW MODELS GO HERE (see "HOW TO ADD A NEW MODEL" at the top of this file)
# ==============================================================================================


# Maps the "type" name used in config.yaml to the class that builds it.
# Add one line per new model: "my_model": MyModel,
MODEL_REGISTRY = {
    "metmap": MetMapMLP,
}
