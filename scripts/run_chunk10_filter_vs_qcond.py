"""Chunk 10 (KEY experiment): filter-then-generate (A) vs quality-conditioned-on-all-data (B).

The heart of the contribution. Teleop data has many low-quality (idle) chunks. The field's
standard move is to FILTER them out and train on the usable remainder (A). Our claim (LACON,
ported to robot trajectories): DON'T discard -- keep all data, condition on a quality dial, and
generate at high quality (B). B should match or beat A at producing realistic, diverse USABLE
trajectories, and its edge should GROW as collected data gets scarcer (because filtering throws
away an ever-larger share).

Falsification-first: if B does not beat A, report it honestly. controllability (chunks 8-9) is
only the premise; THIS is whether the quality dial is actually useful.

Split: usable = chunk activity >= train-median activity (A keeps top ~50%, discards bottom ~50%).
Eval target = real held-out USABLE chunks. Headline metrics = distribution coverage/fidelity
(Wasserstein, precision/recall, best-of-N, diversity); chunkMSE reported but mode-averaging-biased.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from _bootstrap import add_project_root

add_project_root()

from src.eval.distribution_metrics import per_joint_wasserstein, precision_recall  # noqa: E402
from src.models.flow_matching import (  # noqa: E402
    ConditionalFlowConfig,
    ConditionalFlowField,
    conditional_flow_loss,
    conditional_sample,
)
from src.train.train_lstm_vae import episode_split  # noqa: E402
from run_conditional_flow import jerk, norm_stats  # noqa: E402
from run_chunk6_external import build_pairs_from_windows  # noqa: E402
from run_chunk4_eval import ci95  # noqa: E402


def activity(chunks: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(chunks, axis=1)).sum(axis=2).mean(axis=1).astype(np.float32)


def train_fm(loader, cfg, k, epochs, lam, device, seed, quality_dim, action_dim=6, state_dim=6):
    """Train a conditional FM. quality_dim=0 -> state-only (filter baseline A);
    quality_dim=1 -> quality-conditioned (B). Loader yields (state, chunk[, quality])."""
    torch.manual_seed(seed)
    fm = ConditionalFlowField(ConditionalFlowConfig(
        action_dim=action_dim, state_dim=state_dim, chunk_size=k, hidden_dim=cfg["hidden_dim"],
        num_blocks=cfg["num_blocks"], kernel_size=cfg["kernel_size"], embed_dim=cfg["embed_dim"],
        quality_dim=quality_dim,
    )).to(device)
    opt = torch.optim.Adam(fm.parameters(), lr=cfg["lr"])
    for _ in range(epochs):
        fm.train()
        for batch in loader:
            if quality_dim > 0:
                bs, bc, bq = (t.to(device) for t in batch)
            else:
                bs, bc = (t.to(device) for t in batch); bq = None
            loss = conditional_flow_loss(fm, bc, bs, lambda_smooth=lam, quality=bq)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 5.0); opt.step()
    fm.eval()
    return fm


def evaluate(gen, real_usable, c_ev_states_chunks, amin, amax, rj, pr_k):
    """gen, real_usable: [N,k,6] raw. Returns dict of metrics (gen vs real usable pool)."""
    m = len(real_usable)
    out = {
        "jerk_ratio": jerk(gen) / rj,
        "inside": float(((gen >= amin) & (gen <= amax)).mean()),
        "wass": per_joint_wasserstein(gen[:m], real_usable),
    }
    pr, rc = precision_recall(real_usable.reshape(m, -1), gen[:m].reshape(m, -1), k=pr_k)
    out["precision"] = pr; out["recall"] = rc
    return out


def main():
    p = argparse.ArgumentParser(description="Chunk 10: filter-then-generate vs quality-conditioned-on-all.")
    p.add_argument("--config", default="configs/conditional_flow_physicai.yaml")
    p.add_argument("--windows", default="data_processed/windows/being_so101_stateaction_w60_s30.npy")
    p.add_argument("--stats", default="data_processed/stats/being_so101_stateaction_w60_s30_stats.npz")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--pr-k", type=int, default=3)
    p.add_argument("--ep-fracs", type=float, nargs="+", default=[1.0, 0.5, 0.25],
                   help="scarcity sweep: fraction of TRAIN episodes kept (data B uses all of; A filters within).")
    p.add_argument("--n-best", type=int, default=8, help="N samples for best-of-N fidelity.")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    k = cfg["chunk_size"]; lam = cfg.get("lambda_smooth", 0.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows = np.load(args.windows).astype(np.float32)
    stats = np.load(args.stats, allow_pickle=True)
    win_ep = stats["window_episode"]
    print(f"{Path(args.windows).stem} | {windows.shape} | k={k} lambda={lam} epochs={epochs} "
          f"seeds={args.seeds} ep_fracs={args.ep_fracs} | {device}\n")

    # results[frac][metric][A|B] = list over seeds
    metrics = ["wass", "precision", "recall", "best_of_N", "diversity", "jerk_ratio", "inside"]
    results = {f: {mn: {"A": [], "B": []} for mn in metrics} for f in args.ep_fracs}

    for seed in args.seeds:
        np.random.seed(seed); torch.manual_seed(seed)
        tr_w, te_w, _, _ = episode_split(win_ep, cfg["val_frac"], seed)
        tr_eps_all = np.unique(win_ep[tr_w])
        rng = np.random.default_rng(seed)

        # held-out usable target is fixed across fracs (depends only on the test split + threshold)
        s_te, c_te = build_pairs_from_windows(windows[te_w], k)

        for frac in args.ep_fracs:
            keep = rng.choice(tr_eps_all, size=max(1, int(round(len(tr_eps_all) * frac))), replace=False)
            tr_w_f = tr_w[np.isin(win_ep[tr_w], keep)]
            s_tr, c_tr = build_pairs_from_windows(windows[tr_w_f], k)
            s_mean, s_std = norm_stats(s_tr); a_mean, a_std = norm_stats(c_tr)
            nS = lambda x: (x - s_mean) / s_std
            dA = lambda x: x * a_std + a_mean

            # quality axis = activity; usable threshold = median activity of THIS train pool
            A_tr = activity(c_tr)
            thr = float(np.median(A_tr))
            usable = A_tr >= thr
            Am, As_ = float(A_tr.mean()), float(A_tr.std() + 1e-8)
            nA_ = lambda v: (v - Am) / As_
            usable_q_pool = nA_(A_tr[usable])  # B draws its generation quality from here (no target leak)

            # ----- A: filter-then-generate (train on usable subset only, state-only) -----
            loaderA = DataLoader(TensorDataset(torch.from_numpy(nS(s_tr[usable])),
                                               torch.from_numpy((c_tr[usable] - a_mean) / a_std)),
                                 batch_size=cfg["batch_size"], shuffle=True)
            fmA = train_fm(loaderA, cfg, k, epochs, lam, device, seed, quality_dim=0)

            # ----- B: quality-conditioned on ALL data -----
            loaderB = DataLoader(TensorDataset(torch.from_numpy(nS(s_tr)),
                                               torch.from_numpy((c_tr - a_mean) / a_std),
                                               torch.from_numpy(nA_(A_tr)[:, None].astype(np.float32))),
                                 batch_size=cfg["batch_size"], shuffle=True)
            fmB = train_fm(loaderB, cfg, k, epochs, lam, device, seed, quality_dim=1)

            # ----- eval target: real held-out USABLE chunks (activity >= same raw threshold) -----
            te_usable = activity(c_te) >= thr
            idx = np.where(te_usable)[0]
            idx = rng.permutation(idx)[: cfg["eval_conditions"]]
            s_ev, c_ev = s_te[idx], c_te[idx]
            cond = torch.from_numpy(nS(s_ev)).to(device)
            rj = jerk(c_ev); amin, amax = c_tr.reshape(-1, 6).min(0), c_tr.reshape(-1, 6).max(0)
            qB = torch.from_numpy(rng.choice(usable_q_pool, size=cond.size(0))[:, None].astype(np.float32)).to(device)

            with torch.no_grad():
                genA = dA(conditional_sample(fmA, cond, num_steps=cfg["sample_steps"]).cpu().numpy())
                genB = dA(conditional_sample(fmB, cond, num_steps=cfg["sample_steps"], quality=qB).cpu().numpy())

            mA = evaluate(genA, c_ev, None, amin, amax, rj, args.pr_k)
            mB = evaluate(genB, c_ev, None, amin, amax, rj, args.pr_k)

            # diversity + best-of-N (multi-sample) on a subset
            n_div = min(cfg["diversity_conditions"], len(idx)); N = args.n_best
            cond_d = cond[:n_div]
            qB_d = torch.from_numpy(rng.choice(usable_q_pool, size=(N, n_div)).astype(np.float32)).to(device)
            with torch.no_grad():
                stA = np.stack([dA(conditional_sample(fmA, cond_d, num_steps=cfg["sample_steps"]).cpu().numpy())
                                for _ in range(N)], 0)
                stB = np.stack([dA(conditional_sample(fmB, cond_d, num_steps=cfg["sample_steps"],
                                                      quality=qB_d[i][:, None]).cpu().numpy()) for i in range(N)], 0)
            mA["diversity"] = float(stA.std(0).mean()); mB["diversity"] = float(stB.std(0).mean())
            mA["best_of_N"] = float(((stA - c_ev[:n_div][None]) ** 2).mean(axis=(2, 3)).min(0).mean())
            mB["best_of_N"] = float(((stB - c_ev[:n_div][None]) ** 2).mean(axis=(2, 3)).min(0).mean())

            for mn in metrics:
                results[frac][mn]["A"].append(mA[mn]); results[frac][mn]["B"].append(mB[mn])
            print(f"seed {seed} frac {frac:.2f} | usable {usable.mean()*100:.0f}% of {len(s_tr)} | "
                  f"recall A={mA['recall']:.3f} B={mB['recall']:.3f} | wass A={mA['wass']:.3f} B={mB['wass']:.3f} | "
                  f"bestN A={mA['best_of_N']:.2f} B={mB['best_of_N']:.2f}")

    # ---- report ----
    def fmt(frac, mn, mdl):
        v = np.array(results[frac][mn][mdl]); return f"{v.mean():.3f}+/-{ci95(v):.3f}"

    print("\n===== CHUNK 10: filter(A) vs quality-conditioned-on-all(B) =====")
    print("higher=better: recall, precision, diversity, inside | lower=better: wass, best_of_N, |jerk-1|")
    for frac in args.ep_fracs:
        print(f"\n-- train-episode fraction = {frac:.2f} --")
        print(f"{'metric':<12}{'A(filter)':>20}{'B(q-cond all)':>20}{'winner':>10}")
        for mn in metrics:
            a = np.array(results[frac][mn]["A"]); b = np.array(results[frac][mn]["B"])
            higher = mn in ("recall", "precision", "diversity", "inside")
            win = "B" if ((b.mean() > a.mean()) == higher) else "A"
            print(f"{mn:<12}{fmt(frac,mn,'A'):>20}{fmt(frac,mn,'B'):>20}{win:>10}")
    print("\nGO = B matches/beats A on coverage (recall/wass/best_of_N), and the B-advantage GROWS "
          "as frac shrinks (filtering hurts more when data is scarce). NO-GO = A >= B -> discarding is fine.")


if __name__ == "__main__":
    main()
