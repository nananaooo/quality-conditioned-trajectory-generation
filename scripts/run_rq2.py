from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from _bootstrap import add_project_root

add_project_root()

from src.data.lerobot_reader import (  # noqa: E402
    find_dataset_roots,
    group_episode_frames,
    read_data_frame,
    sort_frames,
    stack_feature_matrix,
)
from src.data.windowing import slice_episode  # noqa: E402
from src.models.bc_mlp import BCMLP  # noqa: E402
from src.models.lstm_vae import LSTMVAE, LSTMVAEConfig  # noqa: E402
from src.train.train_lstm_vae import episode_split  # noqa: E402

FEATURES = ["observation.state", "action"]


def load_physicai_episodes(root: str):
    """Return per-episode [T, 12] matrices in the same global order build uses."""
    episodes = []
    for ds in find_dataset_roots(root):
        df = read_data_frame(ds)
        for _, ep_df in group_episode_frames(sort_frames(df)):
            episodes.append(stack_feature_matrix(ep_df, FEATURES))
    return episodes  # list of [T,12]


def normalize_stats(frames: np.ndarray):
    mean = frames.mean(0, keepdims=True)
    std = frames.std(0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _load_vae(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = LSTMVAE(LSTMVAEConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"]


@torch.no_grad()
def synth_frames(ckpt_path, n_windows, seq_len, seed, mode="prior", train_windows=None):
    """Generate synthetic (state,action) frames from a VAE.

    mode='prior': z ~ N(0, I) (diverse but can be jerky/out-of-range).
    mode='aggregate': z ~ aggregate posterior (encode train windows, resample a random
      one's posterior) -> samples land near the real manifold (smoother, in-range).
    """
    torch.manual_seed(seed + 1000)
    model, mean, std = _load_vae(ckpt_path)
    if mode == "aggregate":
        if train_windows is None or len(train_windows) == 0:
            raise ValueError("aggregate mode requires train_windows.")
        x = torch.from_numpy(((train_windows - mean) / std).astype(np.float32))
        mu, logvar = model.encode(x)
        idx = torch.randint(0, mu.size(0), (n_windows,))
        z = model.reparameterize(mu[idx], logvar[idx])
        norm = model.decode(z, seq_len).numpy()
    else:
        norm = model.sample(n_windows, seq_len).numpy()
    denorm = norm * std + mean
    return denorm.reshape(-1, denorm.shape[2]).astype(np.float32)


def train_bc(x: np.ndarray, y: np.ndarray, xt: np.ndarray, yt: np.ndarray, epochs: int, seed: int) -> float:
    torch.manual_seed(seed)
    model = BCMLP(in_dim=x.shape[1], out_dim=y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=256, shuffle=True
    )
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss_fn(model(bx), by).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(xt))
        return float(loss_fn(pred, torch.from_numpy(yt)).item())


def main() -> None:
    p = argparse.ArgumentParser(description="RQ2: does VAE-synthetic data help downstream behavior cloning?")
    p.add_argument("--physicai-root", default="data_raw/physicai")
    p.add_argument("--vae-template", default="outputs/checkpoints/sweep/exp2multi_s{seed}.pt",
                   help="VAE checkpoint per seed; '{seed}' is substituted.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--synth-windows", type=int, default=500)
    p.add_argument("--synth-mode", choices=["prior", "aggregate"], default="aggregate")
    p.add_argument("--seq-len-build", type=int, default=60, help="Window length used when building train windows.")
    p.add_argument("--stride-build", type=int, default=30)
    p.add_argument("--bc-epochs", type=int, default=60)
    p.add_argument("--target", choices=["action", "delta"], default="action",
                   help="action: predict action[t]; delta: predict action[t]-state[t].")
    args = p.parse_args()

    episodes = load_physicai_episodes(args.physicai_root)
    ep_ids = np.arange(len(episodes))
    print(f"physicai episodes: {len(episodes)} | target={args.target} | synth-windows={args.synth_windows}\n")

    real_only, real_synth = [], []
    for seed in args.seeds:
        # episode-level split identical to RQ1 (same seed/val_frac on the same ep ordering)
        win_ep = np.repeat(ep_ids, [len(e) for e in episodes])  # frame->episode id
        tr_idx, te_idx, n_tr, n_te = episode_split(win_ep, args.val_frac, seed)
        all_frames = np.concatenate(episodes, axis=0)
        tr, te = all_frames[tr_idx], all_frames[te_idx]

        def split_xy(frames):
            state, action = frames[:, :6], frames[:, 6:]
            y = action - state if args.target == "delta" else action
            return state, y

        # normalization from REAL TRAIN only
        s_mean, s_std = normalize_stats(tr[:, :6])
        xtr, ytr = split_xy(tr)
        xte, yte = split_xy(te)
        y_mean, y_std = normalize_stats(ytr)
        nx = lambda a: (a - s_mean) / s_std
        ny = lambda a: (a - y_mean) / y_std

        # synthetic frames from this seed's fine-tuned VAE (trained on the same train episodes)
        train_eps = sorted(set(win_ep[tr_idx].tolist()))
        train_windows = np.asarray(
            [w for e in train_eps for w in slice_episode(episodes[e], args.seq_len_build, args.stride_build)],
            dtype=np.float32,
        )
        syn = synth_frames(
            args.vae_template.format(seed=seed), args.synth_windows, args.seq_len, seed,
            mode=args.synth_mode, train_windows=train_windows,
        )
        sx, sy = split_xy(syn)

        e_real = train_bc(nx(xtr), ny(ytr), nx(xte), ny(yte), args.bc_epochs, seed)
        e_both = train_bc(
            np.concatenate([nx(xtr), nx(sx)]), np.concatenate([ny(ytr), ny(sy)]),
            nx(xte), ny(yte), args.bc_epochs, seed,
        )
        real_only.append(e_real)
        real_synth.append(e_both)
        print(f"seed {seed}: {n_tr}tr/{n_te}te ep | real-only test MSE={e_real:.4f} | real+synth={e_both:.4f} "
              f"| {'better' if e_both < e_real else 'worse'} ({(1 - e_both / e_real) * 100:+.1f}%)")

    a, b = np.array(real_only), np.array(real_synth)
    d = a - b
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 and d.std() > 0 else float("nan")
    print("\n===== RQ2 RESULT (downstream BC test MSE) =====")
    print(f"real-only:   {a.mean():.4f} ± {a.std(ddof=1):.4f}")
    print(f"real+synth:  {b.mean():.4f} ± {b.std(ddof=1):.4f}")
    print(f"mean improvement {(1 - b.mean() / a.mean()) * 100:+.1f}% | real+synth better in {int((b < a).sum())}/{len(a)} seeds | paired t={t:.2f}")


if __name__ == "__main__":
    main()
