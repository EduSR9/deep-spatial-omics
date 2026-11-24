from __future__ import annotations

import torch
import torch.nn.functional as F


def train_joint_model(
    model,
    coords,
    metabolites,
    cfg,
):

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
    )

    losses = []
    coord_losses = []
    met_losses = []

    for epoch in range(cfg["epochs"]):

        # Zero the gradients for the optimizer
        optimizer.zero_grad()

        # Execute the model -> get the reconstruction and isodepths
        (coords_recon, metabolites_recon, isodepths, ) = model(coords, metabolites, )

        # Compute reconstruction losses for coordinates and metabolites
        coords_loss = F.mse_loss(coords_recon, coords, )
        met_loss = F.mse_loss(metabolites_recon, metabolites, )
        loss = (0.01 * coords_loss + met_loss)

        # Backpropagate the loss to compute gradients
        loss.backward()

        # Update the model parameters based on the computed gradients
        optimizer.step()

        # Add loss to respective lists of losses
        losses.append(loss.item())
        coord_losses.append(coords_loss.item())
        met_losses.append(met_loss.item())

        # Print progress every 500 epochs
        if (epoch + 1) % 500 == 0:
            print(
                f"Epoch {epoch + 1}: "
                f"total={loss.item():.6f}, "
                f"coord={coords_loss.item():.6f}, "
                f"met={met_loss.item():.6f}",
                flush=True,
            )

    return {
        "total_losses": losses,
        "coord_losses": coord_losses,
        "met_losses": met_losses,
    }