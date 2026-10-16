from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.normalization import denormalize, normalize
from src.models.lstm_vae import LSTMVAE, LSTMVAEConfig, vae_loss


def load_windows(windows_path: str, stats_path: str):
    windows = np.load(windows_path).astype(np.float32)
    stats = np.load(stats_path, allow_pickle=True)
    window_episode = stats["window_episode"] if "window_episode" in stats else np.zeros(len(windows), dtype=np.int64)
    return windows, window_episode


def episode_split(window_episode: np.ndarray, val_frac: float, seed: int):
    """Split window indices by EPISODE (no overlapping-window leakage)."""
    episodes = np.unique(window_episode)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(round(len(episodes) * val_frac)))
    val_eps = set(episodes[:n_val].tolist())
    is_val = np.array([e in val_eps for e in window_episode])
    return np.where(~is_val)[0], np.where(is_val)[0], len(episodes) - n_val, n_val


def normalization_from(windows: np.ndarray):
    mean = windows.mean(axis=(0, 1), keepdims=True)
    std = windows.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def kl_beta(epoch: int, warmup: int, beta_max: float) -> float:
    if warmup <= 0:
        return beta_max
    return beta_max * min(1.0, (epoch + 1) / warmup)


def run_epoch(model, loader, beta, device, optimizer=None, lambda_vel=0.0, lambda_acc=0.0):
    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
    for (batch,) in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl = vae_loss(recon, batch, mu, logvar, beta, lambda_vel, lambda_acc)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        bs = batch.size(0)
        totals["loss"] += loss.item() * bs
        totals["recon"] += recon_loss.item() * bs
        totals["kl"] += kl.item() * bs
        totals["n"] += bs
    n = max(totals["n"], 1)
    return totals["loss"] / n, totals["recon"] / n, totals["kl"] / n


@torch.no_grad()
def sanity_check(model, mean, std, ref_windows_raw, seq_len, device, n_samples=64):
    model.eval()
    samples = denormalize(model.sample(n_samples, seq_len, device=device).cpu().numpy(), mean, std)
    train_min, train_max = ref_windows_raw.min(axis=(0, 1)), ref_windows_raw.max(axis=(0, 1))
    inside = ((samples >= train_min) & (samples <= train_max)).mean()
    flat = samples.reshape(n_samples, -1)
    pdist = np.linalg.norm(flat[:, None] - flat[None], axis=-1)
    diversity = pdist[np.triu_indices(n_samples, k=1)].mean()
    return {
        "inside": float(inside),
        "gen_step": float(np.abs(np.diff(samples, axis=1)).mean()),
        "real_step": float(np.abs(np.diff(ref_windows_raw, axis=1)).mean()),
        "diversity": float(diversity),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate an LSTM-VAE on joint trajectory windows.")
    parser.add_argument("--windows", required=True, nargs="+", help="One or more window .npy files (pooled).")
    parser.add_argument(
        "--stats", required=True, nargs="+", help="Matching npz files with window_episode (parallel to --windows)."
    )
    parser.add_argument("--epochs", type=int, default=40, help="0 = eval-only (requires --init-checkpoint).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--decode-delta", action="store_true", help="Decode per-step deltas + integrate (smoother samples).")
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--beta-warmup", type=int, default=20)
    parser.add_argument("--lambda-vel", type=float, default=0.0, help="Velocity-matching weight (smoothness).")
    parser.add_argument("--lambda-acc", type=float, default=0.0, help="Acceleration-matching weight (smoothness).")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Fraction of EPISODES held out for val.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-checkpoint", default=None, help="Load weights before training (fine-tuning / RQ1).")
    parser.add_argument("--output", default="outputs/checkpoints/lstm_vae.pt")
    parser.add_argument("--tag", default="", help="Optional label printed with the final result line.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(args.windows) != len(args.stats):
        raise ValueError("--windows and --stats must have the same number of entries.")

    # Per-dataset normalization, then pool. Consistent with the transfer design:
    # each source is standardized to ~N(0,1) by its own TRAIN stats (no val leakage);
    # physicai fine-tuning later computes its own stats. For multi-dataset pretraining
    # the per-source stats differ, so the saved checkpoint stats are just dataset[0]'s
    # (used only for the sanity-sample viz; fine-tuning recomputes from physicai).
    train_chunks, val_chunks = [], []
    ref_mean = ref_std = ref_train_raw = None
    for wpath, spath in zip(args.windows, args.stats):
        raw, wep = load_windows(wpath, spath)
        tr_idx, va_idx, n_tr_ep, n_va_ep = episode_split(wep, args.val_frac, args.seed)
        mean_i, std_i = normalization_from(raw[tr_idx])
        norm = normalize(raw, mean_i, std_i).astype(np.float32)
        train_chunks.append(norm[tr_idx])
        val_chunks.append(norm[va_idx])
        if ref_mean is None:
            ref_mean, ref_std, ref_train_raw = mean_i, std_i, raw[tr_idx]
        print(f"  {Path(wpath).name}: {raw.shape} ep-split {n_tr_ep}tr/{n_va_ep}va ({len(tr_idx)}/{len(va_idx)} win)")

    X_train = np.concatenate(train_chunks, axis=0)
    X_val = np.concatenate(val_chunks, axis=0)
    seq_len, feat = X_train.shape[1], X_train.shape[2]
    mean, std = ref_mean, ref_std
    print(f"Pooled: {X_train.shape[0]} train / {X_val.shape[0]} val windows | device={device}")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val)), batch_size=args.batch_size, shuffle=False)

    cfg = LSTMVAEConfig(
        feature_dim=feat, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        num_layers=args.num_layers, decode_delta=args.decode_delta,
    )
    model = LSTMVAE(cfg).to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if ckpt["config"] != cfg.__dict__:
            raise ValueError(f"Arch mismatch.\n  ckpt: {ckpt['config']}\n  now:  {cfg.__dict__}")
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded init weights from: {args.init_checkpoint}")
    if args.epochs == 0 and not args.init_checkpoint:
        raise ValueError("epochs=0 (eval-only) requires --init-checkpoint.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        beta = kl_beta(epoch, args.beta_warmup, args.beta)
        tr = run_epoch(model, train_loader, beta, device, optimizer, args.lambda_vel, args.lambda_acc)
        va = run_epoch(model, val_loader, beta, device, None)
        print(
            f"epoch {epoch + 1:3d}/{args.epochs} | beta={beta:.4f} | "
            f"train recon={tr[1]:.4f} kl={tr[2]:.4f} | val recon={va[1]:.4f} kl={va[2]:.4f}"
        )

    # Final held-out reconstruction (pure MSE, beta=0) — the common RQ1 metric.
    val_recon = run_epoch(model, val_loader, 0.0, device, None)[1]
    s = sanity_check(model, mean, std, ref_train_raw, seq_len, device)
    print("\n===== RESULT =====")
    tag = f"[{args.tag}] " if args.tag else ""
    print(f"{tag}held-out recon MSE (normalized): {val_recon:.5f}")
    print(
        f"prior samples: inside-range={s['inside'] * 100:.1f}%  "
        f"smoothness gen={s['gen_step']:.3f}/real={s['real_step']:.3f}  diversity={s['diversity']:.1f}"
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "mean": mean,
            "std": std,
            "stats_path": args.stats,
            "windows_path": args.windows,
            "val_recon": val_recon,
            "args": vars(args),
        },
        out,
    )
    print(f"Saved checkpoint: {out}")


if __name__ == "__main__":
    main()
