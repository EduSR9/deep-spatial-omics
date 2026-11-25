from __future__ import annotations

import torch
import numpy as np
from torchgen import model

from src.model import MetMapMLP
from src.attention import AttentionMetMap

from src.model_joint import JointSpatialMetabolicModel
from src.model_xym import XYMMetMap
from src.model_pca import PCAMetMap

from src.train import train_model
from src.train_joint import train_joint_model
from src.train_xym import train_xym_model
from src.train_pca import train_pca_model

MODEL_REGISTRY = {
    "metmap": MetMapMLP,
    "attention": AttentionMetMap,
    "joint": JointSpatialMetabolicModel,
    "xym": XYMMetMap,
    "pca": PCAMetMap,
}

"""
Master file for choosing model specific parameters and outputs.

create_model: assigning model parameters based on model type
train_selected_model: selection of model training
predict_model: 
fit_pca: precalculation of PCA to be used as model input and training
"""

# =====================================================
# MODEL CREATION
# =====================================================

def create_model(
    model_cfg: dict,
    n_features: int,
):

    model_type = model_cfg["type"]

    if model_type == "metmap":
        return MetMapMLP(
            n_features=n_features,
            hidden_dim=model_cfg["hidden_dim"],
            n_isodepth=model_cfg["n_isodepth"],
        )

    elif model_type == "attention":
        return AttentionMetMap(
            n_features=n_features,
            hidden_dim=model_cfg["hidden_dim"],
            n_isodepth=model_cfg["n_isodepth"],
        )

    elif model_type == "joint":
        return JointSpatialMetabolicModel(
            n_features=n_features,
            hidden_dim=model_cfg["hidden_dim"],
            n_isodepth=model_cfg["n_isodepth"],
        )

    elif model_type == "xym":
        return XYMMetMap(
            n_features=n_features,
            hidden_dim=model_cfg["hidden_dim"],
            n_isodepth=model_cfg["n_isodepth"],
        )

    elif model_type == "pca":
        return PCAMetMap(
            n_pcs=model_cfg["n_pcs"],
            n_features=n_features,
            hidden_dim=model_cfg["hidden_dim"],
            n_isodepth=model_cfg["n_isodepth"],
        )

    raise ValueError(
        f"Unknown model type: {model_type}"
    )


# =====================================================
# TRAINING
# =====================================================

def train_selected_model(
    model_type,
    model,
    coords_t,
    targets_t,
    train_cfg,
    pcs_t=None,
):
    
    if model_type in ("metmap", "attention", ):
        total_losses, l1_losses = train_model(
            model,
            coords_t,
            targets_t,
            train_cfg,
        )
        return {
            "total_losses": total_losses,
            "l1_losses": l1_losses,
        }

    elif model_type == "joint":
        return train_joint_model(
            model,
            coords_t,
            targets_t,
            train_cfg,
        )

    elif model_type == "xym":
        return train_xym_model(
            model,
            coords_t,
            targets_t,
            train_cfg,
        )

    elif model_type == "pca":
        return train_pca_model(
            model,
            coords_t,
            pcs_t,
            targets_t,
            train_cfg,
        )

    raise ValueError(
        f"Training not implemented for {model_type}"
    )

# =====================================================
# PREDICTION
# =====================================================

def predict_model(
    model_type,
    model,
    coords_t,
    targets_t=None,
    pcs_t=None,
):
    model.eval()

    with torch.no_grad():

        if model_type in ("metmap","attention",):
            prediction, isodepths = model(coords_t,)

        elif model_type == "xym":
            prediction, isodepths = model(coords_t, targets_t,)

        elif model_type == "joint":
            (coords_recon, prediction, isodepths,) = model(coords_t,targets_t,)

        elif model_type == "pca":
            prediction, isodepths = model(coords_t, pcs_t,)

        else:
            raise ValueError(f"Prediction not implemented for {model_type}")

    return (prediction,isodepths,)

# =====================================================
# PCA COMPUTATION
# =====================================================

def fit_pca(metabolites, n_pcs,):

    X = metabolites.numpy()
    X = X - X.mean(axis=0, keepdims=True,)
    U, S, Vt = np.linalg.svd(X, full_matrices=False,)

    return {
        "mean": X.mean(axis=0,keepdims=True,),
        "components": Vt[:n_pcs],
    }

def transform_pca(metabolites, pca,):

    X = metabolites.numpy()
    X = X - pca["mean"]
    pcs = X @ pca["components"].T

    return torch.tensor(
        pcs,
        dtype=torch.float32,
    )