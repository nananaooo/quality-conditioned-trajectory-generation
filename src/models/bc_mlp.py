from __future__ import annotations

import torch.nn as nn


class BCMLP(nn.Module):
    """Frame-wise behavior-cloning policy: observation.state (6D) -> action (6D).

    Deliberately small — RQ2 measures whether synthetic data helps a downstream
    policy, not absolute policy quality.
    """

    def __init__(self, in_dim: int = 6, out_dim: int = 6, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
