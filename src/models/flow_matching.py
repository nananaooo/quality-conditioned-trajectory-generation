from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FlowMatchingConfig:
    feature_dim: int = 12  # channels (state 6D + action 6D)
    seq_len: int = 60
    hidden_dim: int = 128
    num_blocks: int = 4  # residual 1D-conv blocks
    kernel_size: int = 5
    time_embed_dim: int = 64


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map continuous t in [0,1] (shape [B]) to [B, dim] sinusoidal features."""
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1))
    )
    args = t[:, None].float() * freqs[None, :] * 1000.0
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # pad odd dims
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _ConvBlock(nn.Module):
    """Residual 1D-conv block with FiLM-style time conditioning (over time axis)."""

    def __init__(self, channels: int, kernel_size: int, time_dim: int) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.time_proj = nn.Linear(time_dim, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None]  # broadcast time signal over the sequence
        h = self.act(self.norm2(self.conv2(h)))
        return x + h


class FlowMatchingVelocity(nn.Module):
    """Velocity field v_theta(x_t, t) for rectified flow over [B, seq_len, feature_dim] windows.

    The window is treated as [B, feature_dim, seq_len] (channels x time) so 1D convs
    capture temporal structure (smoothness) directly, unlike a flattened MLP. This is
    the structural reason flow matching is expected to fix the VAE's prior-sample
    jerkiness: it denoises in data space along a temporal-convolutional field.
    """

    def __init__(self, config: FlowMatchingConfig | None = None) -> None:
        super().__init__()
        cfg = config or FlowMatchingConfig()
        self.cfg = cfg
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim),
        )
        self.in_proj = nn.Conv1d(cfg.feature_dim, cfg.hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [_ConvBlock(cfg.hidden_dim, cfg.kernel_size, cfg.time_embed_dim) for _ in range(cfg.num_blocks)]
        )
        self.out_proj = nn.Conv1d(cfg.hidden_dim, cfg.feature_dim, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, feature_dim], t: [B] in [0,1] -> velocity [B, seq_len, feature_dim]."""
        t_emb = self.time_mlp(sinusoidal_time_embedding(t, self.cfg.time_embed_dim))
        h = self.in_proj(x.transpose(1, 2))  # [B, hidden, seq_len]
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(h).transpose(1, 2)


@dataclass
class ConditionalFlowConfig:
    action_dim: int = 6  # generated channels (action chunk)
    state_dim: int = 6  # conditioning current state
    chunk_size: int = 8
    hidden_dim: int = 128
    num_blocks: int = 4
    kernel_size: int = 3  # short chunk -> smaller kernel
    embed_dim: int = 64
    quality_dim: int = 0  # 0 = disabled (state-only, backward compatible); >0 enables quality conditioning


class ConditionalFlowField(nn.Module):
    """Conditional rectified-flow field over action chunks, conditioned on the current state.

    Predicts the CLEAN target x1 (sample/x1-prediction) rather than the velocity. With the
    linear path x_t=(1-t)x0+t*x1 and white-noise x0, velocity-prediction leaves a
    high-frequency residual that integrates into jerk (the unconditional chunk-1 failure
    mode); x1-prediction targets the smooth clean chunk directly. Conditioning (time +
    state [+ quality]) is injected per block as a FiLM-style additive signal.

    When ``cfg.quality_dim > 0`` an extra quality signal is fused alongside state, turning the
    filter (keep/drop a segment) of the prior quality-aware work into a generation DIAL: at
    inference the same state can be sampled at a chosen quality level. With quality_dim=0 the
    field is identical to the state-only version (chunks 1-7) -- backward compatible.
    """

    def __init__(self, config: ConditionalFlowConfig | None = None) -> None:
        super().__init__()
        cfg = config or ConditionalFlowConfig()
        self.cfg = cfg
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim), nn.SiLU(), nn.Linear(cfg.embed_dim, cfg.embed_dim)
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.embed_dim), nn.SiLU(), nn.Linear(cfg.embed_dim, cfg.embed_dim)
        )
        self.quality_mlp = (
            nn.Sequential(nn.Linear(cfg.quality_dim, cfg.embed_dim), nn.SiLU(), nn.Linear(cfg.embed_dim, cfg.embed_dim))
            if cfg.quality_dim > 0
            else None
        )
        self.fuse = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.in_proj = nn.Conv1d(cfg.action_dim, cfg.hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [_ConvBlock(cfg.hidden_dim, cfg.kernel_size, cfg.embed_dim) for _ in range(cfg.num_blocks)]
        )
        self.out_proj = nn.Conv1d(cfg.hidden_dim, cfg.action_dim, 1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, state: torch.Tensor, quality: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: [B, chunk, action_dim], t: [B], state: [B, state_dim], quality: [B, quality_dim] or None
        -> x1_hat [B, chunk, action_dim]."""
        signal = self.time_mlp(sinusoidal_time_embedding(t, self.cfg.embed_dim)) + self.state_mlp(state)
        if self.quality_mlp is not None:
            if quality is None:
                raise ValueError("quality_dim>0 but no quality passed to forward().")
            signal = signal + self.quality_mlp(quality)
        cond = self.fuse(signal)
        h = self.in_proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(h).transpose(1, 2)


def conditional_flow_loss(
    model: ConditionalFlowField,
    x1: torch.Tensor,
    state: torch.Tensor,
    lambda_smooth: float = 0.0,
    quality: torch.Tensor | None = None,
) -> torch.Tensor:
    """x1-prediction training: regress the clean action chunk from a point on the linear path.

    ``lambda_smooth`` adds velocity+acceleration matching between the predicted chunk and the
    real chunk (temporal 1st/2nd differences along the chunk axis). This directly targets the
    residual jerkiness of sampled chunks (chunk-2 finding: jerk plateaus ~2.4x real, field-limited)
    without harming fidelity, since the target is the real chunk's own pace.

    ``quality`` ([B, quality_dim]) is the per-chunk conditioning quality signal when the field
    has quality_dim>0; None keeps the state-only behaviour.
    """
    b = x1.size(0)
    x0 = torch.randn_like(x1)
    t = torch.rand(b, device=x1.device)
    x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
    x1_hat = model(x_t, t, state, quality)
    loss = nn.functional.mse_loss(x1_hat, x1)
    if lambda_smooth > 0:
        vel = nn.functional.mse_loss(x1_hat[:, 1:] - x1_hat[:, :-1], x1[:, 1:] - x1[:, :-1])
        acc = nn.functional.mse_loss(
            x1_hat[:, 2:] - 2 * x1_hat[:, 1:-1] + x1_hat[:, :-2],
            x1[:, 2:] - 2 * x1[:, 1:-1] + x1[:, :-2],
        )
        loss = loss + lambda_smooth * (vel + acc)
    return loss


@torch.no_grad()
def conditional_sample(
    model: ConditionalFlowField,
    state: torch.Tensor,
    num_steps: int = 50,
    eps: float = 1e-3,
    quality: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the ODE from x0~N(0,I) to t=1, conditioned on state [+ quality].

    For the linear path with x1-prediction, the velocity is v=(x1_hat - x_t)/(1-t).
    The (1-t) denominator is clamped by eps near t=1 for stability.
    ``quality`` ([B, quality_dim]) selects the generation quality level when the field has
    quality_dim>0; None keeps the state-only behaviour.
    Returns [B, chunk, action_dim] in normalized space (B = state.size(0)).
    """
    model.eval()
    b = state.size(0)
    x = torch.randn(b, model.cfg.chunk_size, model.cfg.action_dim, device=state.device)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        tval = i * dt
        t = torch.full((b,), tval, device=state.device)
        x1_hat = model(x, t, state, quality)
        v = (x1_hat - x) / max(1.0 - tval, eps)
        x = x + v * dt
    return x


def flow_matching_loss(model: FlowMatchingVelocity, x1: torch.Tensor) -> torch.Tensor:
    """Rectified-flow (conditional-OT) training loss.

    Linear path x_t = (1-t) x0 + t x1 with x0 ~ N(0,I); the constant target velocity
    along this path is (x1 - x0). No noise schedule to tune -> stable training.
    """
    b = x1.size(0)
    x0 = torch.randn_like(x1)
    t = torch.rand(b, device=x1.device)
    x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
    target = x1 - x0
    pred = model(x_t, t)
    return nn.functional.mse_loss(pred, target)


@torch.no_grad()
def sample(
    model: FlowMatchingVelocity,
    num_samples: int,
    seq_len: int,
    num_steps: int = 50,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Integrate dx/dt = v_theta(x, t) from x0 ~ N(0,I) over t in [0,1] (Euler).

    Returns [num_samples, seq_len, feature_dim] in normalized space.
    """
    model.eval()
    x = torch.randn(num_samples, seq_len, model.cfg.feature_dim, device=device)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((num_samples,), i * dt, device=device)
        x = x + model(x, t) * dt
    return x
