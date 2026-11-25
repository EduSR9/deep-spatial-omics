from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

def compute_loss(
    pred,
    target,
    decoder_weight,
    lasso_lambda,
):
    mse = F.mse_loss(pred, target,)
    l1 = (lasso_lambda * torch.norm(decoder_weight, 1,))
    return (mse + l1, l1,)

def train_xym_model(
    model,
    coords,
    metabolites,
    cfg,
):

    # Initialize the optimizer: 
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
    )


    # Track losses (used for plotting and analysis)
    losses = []
    l1_losses = []

    for epoch in range(cfg["epochs"]):

        # Zero the gradients for the optimizer
        optimizer.zero_grad()

        # Execute the model -> get the reconstruction and isodepths
        (reconstruction, isodepths,) = model(coords, metabolites,)

        # Compute metabolite reconstruction loss
        loss, l1 = compute_loss(reconstruction, metabolites, model.decoder.weight, cfg["lasso_lambda"],)

        # Backpropagate the loss to compute gradients
        loss.backward()

        # Update the model parameters based on the computed gradients
        optimizer.step()

        # Add loss to the list of losses
        losses.append(loss.item())
        l1_losses.append(l1.item())

        # Print progress every 500 epochs
        mse = loss.item() - l1.item()
        if (epoch + 1) % 500 == 0:
            print(
                f"Epoch {epoch + 1}: "
                f"total={loss.item():.6f}, "
                f"MSE={mse:.6f}",
                flush=True,
            )

    return {
        "total_losses": np.array(losses),
        "l1_losses": np.array(l1_losses),
    }
