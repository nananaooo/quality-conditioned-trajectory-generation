from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from _bootstrap import add_project_root

add_project_root()

from src.models.bc_chunk import BCChunkMLP  # noqa: E402
from src.models.flow_matching import (  # noqa: E402
    ConditionalFlowConfig,
    ConditionalFlowField,
    conditional_flow_loss,
    conditional_sample,
)
from src.train.train_lstm_vae import episode_split  # noqa: E402

# reuse the exact episode loader RQ2 used (per-episode [T,12], global order)
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("_rq2", Path(__file__).with_name("run_rq2.py"))
_rq2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rq2)
load_physicai_episodes = _rq2.load_physicai_episodes


def build_pairs(episodes, ep_indices, k):
    """From episodes (list of [T,12]) build (state[t], action[t:t+k]) for the given episode ids."""
    states, chunks = [], []
    for e in ep_indices:
        ep = episodes[e]
        for t in range(len(ep) - k):
            states.append(ep[t, :6])
            chunks.append(ep[t : t + k, 6:])
    return np.asarray(states, np.float32), np.asarray(chunks, np.float32)


def norm_stats(a):
    m = a.reshape(-1, a.shape[-1]).mean(0)
    s = a.reshape(-1, a.shape[-1]).std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return m.astype(np.float32), s.astype(np.float32)


def jerk(chunks):  # raw units, per-step |Δaction| along the chunk time axis
    return float(np.abs(np.diff(chunks, axis=1)).mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Chunk 2: conditional flow-matching (state->action chunk) vs chunk-BC.")
    p.add_argument("--config", default="configs/conditional_flow_physicai.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lambda-smooth", type=float, default=None, help="Override config smoothness weight.")
    p.add_argument("--output", default="outputs/checkpoints/conditional_flow_physicai.pt")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.lambda_smooth is not None:
        cfg["lambda_smooth"] = args.lambda_smooth
    seed = args.seed if args.seed is not None else cfg["seed"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    k = cfg["chunk_size"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episodes = load_physicai_episodes(cfg["physicai_root"])
    win_ep = np.repeat(np.arange(len(episodes)), [len(e) for e in episodes])
    tr_idx, _, n_tr, n_te = episode_split(win_ep, cfg["val_frac"], seed)
    train_eps = sorted(set(win_ep[tr_idx].tolist()))
    test_eps = [e for e in range(len(episodes)) if e not in set(train_eps)]
    print(f"physicai {len(episodes)} ep | {n_tr}tr/{n_te}te | k={k} | seed={seed} | device={device}")

    s_tr, c_tr = build_pairs(episodes, train_eps, k)
    s_te, c_te = build_pairs(episodes, test_eps, k)
    s_mean, s_std = norm_stats(s_tr)            # condition normalization (train only)
    a_mean, a_std = norm_stats(c_tr)            # action-chunk normalization (train only)
    nS = lambda x: (x - s_mean) / s_std
    nA = lambda x: (x - a_mean) / a_std
    dA = lambda x: x * a_std + a_mean
    print(f"pairs: {len(s_tr)} train / {len(s_te)} test")

    sT = torch.from_numpy(nS(s_tr)); cT = torch.from_numpy(nA(c_tr))
    loader = DataLoader(TensorDataset(sT, cT), batch_size=cfg["batch_size"], shuffle=True)

    # ---- conditional flow-matching ----
    fm = ConditionalFlowField(ConditionalFlowConfig(
        action_dim=6, state_dim=6, chunk_size=k, hidden_dim=cfg["hidden_dim"],
        num_blocks=cfg["num_blocks"], kernel_size=cfg["kernel_size"], embed_dim=cfg["embed_dim"],
    )).to(device)
    opt = torch.optim.Adam(fm.parameters(), lr=cfg["lr"])
    lambda_smooth = cfg.get("lambda_smooth", 0.0)
    for ep in range(epochs):
        fm.train(); tot = n = 0.0
        for bs, bc in loader:
            bs, bc = bs.to(device), bc.to(device)
            loss = conditional_flow_loss(fm, bc, bs, lambda_smooth=lambda_smooth)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 5.0); opt.step()
            tot += loss.item() * bs.size(0); n += bs.size(0)
        if (ep + 1) % max(1, epochs // 8) == 0 or ep == 0:
            print(f"  [FM]  epoch {ep+1:3d}/{epochs} | x1-pred mse={tot/max(n,1):.5f}")

    # ---- chunk-BC baseline (deterministic) ----
    bc_model = BCChunkMLP(in_dim=6, action_dim=6, chunk_size=k, hidden_dim=cfg["hidden_dim"]).to(device)
    optb = torch.optim.Adam(bc_model.parameters(), lr=cfg["lr"])
    lossf = torch.nn.MSELoss()
    for ep in range(epochs):
        bc_model.train()
        for bs, bc in loader:
            bs, bc = bs.to(device), bc.to(device)
            optb.zero_grad()
            lossf(bc_model(bs), bc.reshape(bc.size(0), -1)).backward()
            optb.step()

    # ---- evaluate on held-out conditions ----
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(s_te))[: cfg["eval_conditions"]]
    s_eval_raw, c_eval_raw = s_te[idx], c_te[idx]
    cond = torch.from_numpy(nS(s_eval_raw)).to(device)
    real_jerk = jerk(c_eval_raw)
    amin, amax = c_tr.reshape(-1, 6).min(0), c_tr.reshape(-1, 6).max(0)

    def chunk_metrics(gen_raw):
        inside = ((gen_raw >= amin) & (gen_raw <= amax)).mean()
        anchor = np.abs(gen_raw[:, 0, :] - s_eval_raw).mean()  # first action vs conditioning state
        return jerk(gen_raw), float(inside), float(anchor)

    with torch.no_grad():
        fm_gen = dA(conditional_sample(fm, cond, num_steps=cfg["sample_steps"]).cpu().numpy())
        bc_gen = dA(bc_model(cond).reshape(-1, k, 6).cpu().numpy())
    real_anchor = float(np.abs(c_eval_raw[:, 0, :] - s_eval_raw).mean())
    fm_j, fm_in, fm_an = chunk_metrics(fm_gen)
    bc_j, bc_in, bc_an = chunk_metrics(bc_gen)
    fm_mse = float(((fm_gen - c_eval_raw) ** 2).mean())
    bc_mse = float(((bc_gen - c_eval_raw) ** 2).mean())

    # diversity + best-of-N fidelity on a subset
    n_div = min(cfg["diversity_conditions"], len(idx))
    N = cfg["diversity_samples"]
    cond_d = cond[:n_div]
    with torch.no_grad():
        stacks = np.stack([dA(conditional_sample(fm, cond_d, num_steps=cfg["sample_steps"]).cpu().numpy())
                           for _ in range(N)], 0)  # [N, n_div, k, 6]
    diversity = float(stacks.std(0).mean())
    best_of_n = float(((stacks - c_eval_raw[:n_div][None]) ** 2).mean(axis=(2, 3)).min(0).mean())
    bc_mse_d = float(((bc_gen[:n_div] - c_eval_raw[:n_div]) ** 2).mean())

    print("\n===== CONDITIONAL FLOW-MATCHING vs CHUNK-BC (held-out, raw units) =====")
    print(f"real action-chunk jerk={real_jerk:.3f}  anchor|a0-s|={real_anchor:.3f}")
    print(f"  FM : jerk={fm_j:.3f} ({fm_j/real_jerk:.2f}x) inside={fm_in*100:.1f}% anchor={fm_an:.3f} chunkMSE={fm_mse:.3f}")
    print(f"  BC : jerk={bc_j:.3f} ({bc_j/real_jerk:.2f}x) inside={bc_in*100:.1f}% anchor={bc_an:.3f} chunkMSE={bc_mse:.3f}")
    print(f"  multimodality (n={n_div}, N={N}): FM diversity={diversity:.3f} | "
          f"FM best-of-N MSE={best_of_n:.3f} vs BC MSE={bc_mse_d:.3f}")
    print("\nGO(primary): FM jerk <=~1.5x real AND inside-range high (fixes unconditional 4.54x).")
    print("GO(bonus):   FM diversity>0 AND best-of-N MSE <= BC MSE (multimodal coverage).")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fm_state": fm.state_dict(), "fm_config": fm.cfg.__dict__,
                "s_mean": s_mean, "s_std": s_std, "a_mean": a_mean, "a_std": a_std,
                "seed": seed, "metrics": {"real_jerk": real_jerk, "fm_jerk": fm_j, "bc_jerk": bc_j,
                "fm_inside": fm_in, "bc_inside": bc_in, "diversity": diversity,
                "best_of_n": best_of_n, "bc_mse_d": bc_mse_d}, "source_config": cfg}, out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
