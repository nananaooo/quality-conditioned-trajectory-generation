from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LSTMVAEConfig:
    feature_dim: int = 12
    hidden_dim: int = 128
    latent_dim: int = 32
    num_layers: int = 1
    decode_delta: bool = False  # decode per-step deltas + integrate (structural smoothness)


class LSTMVAE(nn.Module):
    """A minimal sequence VAE over fixed-length joint trajectory windows.

    Encoder: LSTM -> last hidden -> (mu, logvar).
    Decoder: z initializes the hidden state and is fed (repeated) at every step,
    producing the full window. This is a sanity-check baseline (Step "baseline"),
    not the final generation model.
    """

    def __init__(self, config: LSTMVAEConfig | None = None) -> None:
        super().__init__()
        cfg = config or LSTMVAEConfig()
        self.cfg = cfg

        self.encoder = nn.LSTM(cfg.feature_dim, cfg.hidden_dim, cfg.num_layers, batch_first=True)
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        self.fc_h = nn.Linear(cfg.latent_dim, cfg.hidden_dim * cfg.num_layers)
        self.decoder = nn.LSTM(cfg.latent_dim, cfg.hidden_dim, cfg.num_layers, batch_first=True)
        self.output = nn.Linear(cfg.hidden_dim, cfg.feature_dim)
        # delta mode: a separate head decodes the initial frame from z; the LSTM output
        # is interpreted as per-step deltas that are integrated (cumsum) into a trajectory.
        self.fc_x0 = nn.Linear(cfg.latent_dim, cfg.feature_dim) if cfg.decode_delta else None

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h, _) = self.encoder(x)
        h_last = h[-1]
        return self.fc_mu(h_last), self.fc_logvar(h_last)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch = z.size(0)
        dec_in = z.unsqueeze(1).expand(batch, seq_len, self.cfg.latent_dim)
        h0 = torch.tanh(self.fc_h(z)).view(batch, self.cfg.num_layers, self.cfg.hidden_dim)
        h0 = h0.transpose(0, 1).contiguous()
        c0 = torch.zeros_like(h0)
        out, _ = self.decoder(dec_in, (h0, c0))
        raw = self.output(out)
        if self.cfg.decode_delta:
            x0 = self.fc_x0(z).unsqueeze(1)  # [B, 1, F] initial frame
            return x0 + torch.cumsum(raw, dim=1)  # raw = per-step deltas -> integrate
        return raw

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, num_samples: int, seq_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Draw z ~ N(0, I) and decode -> [num_samples, seq_len, feature_dim] (normalized space)."""
        z = torch.randn(num_samples, self.cfg.latent_dim, device=device)
        return self.decode(z, seq_len)


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    lambda_vel: float = 0.0,
    lambda_acc: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss + optional velocity/acceleration matching (smoothness/pace fidelity).

    The velocity term matches the recon's 1st temporal difference to the target's,
    and the acceleration term matches the 2nd difference. These shape the decoder to
    reproduce the data's pace (addresses RQ2 diagnosis: prior samples too jerky).
    The returned ``recon_loss`` is pure MSE so it stays comparable across configs.
    """
    recon_loss = nn.functional.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl
    if lambda_vel > 0:
        vel = nn.functional.mse_loss(recon[:, 1:] - recon[:, :-1], target[:, 1:] - target[:, :-1])
        total = total + lambda_vel * vel
    if lambda_acc > 0:
        acc_r = recon[:, 2:] - 2 * recon[:, 1:-1] + recon[:, :-2]
        acc_t = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
        total = total + lambda_acc * nn.functional.mse_loss(acc_r, acc_t)
    return total, recon_loss, kl
