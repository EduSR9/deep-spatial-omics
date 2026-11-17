"""Coordinate-only MET-MAP with a metabolite self-attention decoder."""

import torch
from torch import nn

from src.model import MetMapMLP


class AttentionMetMap(nn.Module):
    def __init__(self, n_features, hidden_dim=120, n_isodepth=3):
        super().__init__()
        self.encoder = MetMapMLP(n_features, hidden_dim, n_isodepth).encoder
        self.depth_projection = nn.Linear(n_isodepth, 32)
        self.feature_embeddings = nn.Parameter(torch.randn(n_features, 32) * 0.02)
        self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
        self.norm1 = nn.LayerNorm(32)
        self.feedforward = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.norm2 = nn.LayerNorm(32)
        self.output = nn.Linear(32, 1)

    def forward(self, coords):
        depths = self.encoder(coords)
        tokens = self.depth_projection(depths)[:, None, :] + self.feature_embeddings[None, :, :]
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.feedforward(tokens))
        return self.output(tokens).squeeze(-1), depths
