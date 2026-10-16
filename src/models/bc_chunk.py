from __future__ import annotations

import torch.nn as nn


class BCChunkMLP(nn.Module):
    """Action-chunk behavior-cloning policy: state(6D) -> [a_t, ..., a_{t+k-1}] (6D*k)."""

    def __init__(
        self,
        in_dim: int = 6,
        action_dim: int = 6,
        chunk_size: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim * chunk_size),
        )

    def forward(self, x):
        return self.net(x)
