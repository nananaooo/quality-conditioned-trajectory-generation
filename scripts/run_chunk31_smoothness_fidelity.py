"""Chunk 31: honest smoothness-fidelity metrics for the A-vs-B comparison.

WHY THIS EXISTS
---------------
The metric the codebase calls `jerk` is

    jerk(chunks) = mean_{t,j} |Delta a|          # FIRST difference

which is identical to the `activity` quality axis up to the joint count:

    activity(chunk) = mean_t sum_j |Delta a| = D * jerk(chunk)

Two consequences, both verified 2026-09-16:
  1. It is NOT jerk (jerk is the third difference). The name is wrong everywhere.
  2. In the A-vs-B comparison it is CIRCULAR FOR B: arm B is conditioned to generate at the
     usable activity level and is then scored on how closely its activity matches the usable
     activity. Arm A receives no such instruction. Not an apples-to-apples fidelity measure.

So this script reports, alongside the usual metrics, fidelity measures that are NOT the
conditioning variable (arms in chunk10/29 use quality_dim=1 = activity only):

    motion_ratio    mean|d1 a| gen / real     -- the old "jerk_ratio", renamed. DIAL-DEPENDENT for B.
    accel_ratio     mean|d2 a| gen / real     -- second difference. Not conditioned here.
    jerk_ratio      mean|d3 a| gen / real     -- TRUE jerk (third difference). Not conditioned.
    rough_ratio     (|d2|/|d1|) gen / real    -- scale-free roughness per unit motion.
                                                 Independent of the activity dial by construction.

All ratios: 1.0 = matches real held-out usable data. <1 = generated is smoother than real.

Per-seed lines are printed so scripts/make_table2_paired.py can compute paired statistics.
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
from src.models.flow_matching import conditional_sample  # noqa: E402
from src.train.train_lstm_vae import episode_split  # noqa: E402
from run_conditional_flow import norm_stats  # noqa: E402
from run_chunk6_external import build_pairs_from_windows  # noqa: E402
from run_chunk4_eval import ci95  # noqa: E402
from run_chunk10_filter_vs_qcond import activity, train_fm  # noqa: E402

ARMS = ["A", "B"]
METRICS = ["recall", "prec", "div", "wass", "bestN",
           "motion", "accel", "jerk3", "rough", "inside"]


def _absdiff(x: np.ndarray, order: int) -> float:
    """Mean |n-th temporal difference| over a chunk batch [N, k, D]."""
    for _ in range(order):
        x = np.diff(x, axis=1)
    return float(np.abs(x).mean())


def smoothness_profile(chunks: np.ndarray) -> dict[str, float]:
    d1 = _absdiff(chunks, 1)
    d2 = _absdiff(chunks, 2)
    d3 = _absdiff(chunks, 3)
    return {"d1": d1, "d2": d2, "d3": d3, "rough": d2 / max(d1, 1e-12)}


def main():
    p = argparse.ArgumentParser(description="Chunk 31: true smoothness fidelity, A vs B.")
    p.add_argument("--config", default="configs/conditional_flow_physicai.yaml")
    p.add_argument("--windows", default="data_processed/windows/being_so101_stateaction_w60_s30.npy")
    p.add_argument("--stats", default="data_processed/stats/being_so101_stateaction_w60_s30_stats.npz")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--pr-k", type=int, default=3)
    p.add_argument("--ep-fracs", type=float, nargs="+", default=[1.0])
    p.add_argument("--n-best", type=int, default=8)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    k = cfg["chunk_size"]; lam = cfg.get("lambda_smooth", 0.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows = np.load(args.windows).astype(np.float32)
    stats = np.load(args.stats, allow_pickle=True)
    win_ep = stats["window_episode"]
    D = windows.shape[2] // 2
    print(f"{Path(args.windows).stem} | {windows.shape} | D={D} k={k} lambda={lam} epochs={epochs} "
          f"seeds={args.seeds} ep_fracs={args.ep_fracs} | {device}\n")

    results = {f: {m: {a: [] for a in ARMS} for m in METRICS} for f in args.ep_fracs}

    for seed in args.seeds:
        np.random.seed(seed); torch.manual_seed(seed)
        tr_w, te_w, _, _ = episode_split(win_ep, cfg["val_frac"], seed)
        tr_eps_all = np.unique(win_ep[tr_w])
        rng = np.random.default_rng(seed)
        s_te, c_te = build_pairs_from_windows(windows[te_w], k, state_dim=D)

        for frac in args.ep_fracs:
            keep = rng.choice(tr_eps_all, size=max(1, int(round(len(tr_eps_all) * frac))), replace=False)
            tr_w_f = tr_w[np.isin(win_ep[tr_w], keep)]
            s_tr, c_tr = build_pairs_from_windows(windows[tr_w_f], k, state_dim=D)
            s_mean, s_std = norm_stats(s_tr); a_mean, a_std = norm_stats(c_tr)
            nS = lambda x: (x - s_mean) / s_std
            dA = lambda x: x * a_std + a_mean

            A_tr = activity(c_tr)
            thr = float(np.median(A_tr))
            usable = A_tr >= thr
            Am, As_ = float(A_tr.mean()), float(A_tr.std() + 1e-8)
            nA_ = lambda v: (v - Am) / As_
            usable_q_pool = nA_(A_tr[usable])

            loaderA = DataLoader(TensorDataset(torch.from_numpy(nS(s_tr[usable])),
                                               torch.from_numpy((c_tr[usable] - a_mean) / a_std)),
                                 batch_size=cfg["batch_size"], shuffle=True)
            fmA = train_fm(loaderA, cfg, k, epochs, lam, device, seed,
                           quality_dim=0, action_dim=D, state_dim=D)

            loaderB = DataLoader(TensorDataset(torch.from_numpy(nS(s_tr)),
                                               torch.from_numpy((c_tr - a_mean) / a_std),
                                               torch.from_numpy(nA_(A_tr)[:, None].astype(np.float32))),
                                 batch_size=cfg["batch_size"], shuffle=True)
            fmB = train_fm(loaderB, cfg, k, epochs, lam, device, seed,
                           quality_dim=1, action_dim=D, state_dim=D)

            te_usable = activity(c_te) >= thr
            idx = rng.permutation(np.where(te_usable)[0])[: cfg["eval_conditions"]]
            s_ev, c_ev = s_te[idx], c_te[idx]
            cond = torch.from_numpy(nS(s_ev)).to(device)
            amin, amax = c_tr.reshape(-1, D).min(0), c_tr.reshape(-1, D).max(0)
            ref = smoothness_profile(c_ev)
            qB = torch.from_numpy(rng.choice(usable_q_pool, size=cond.size(0))[:, None]
                                  .astype(np.float32)).to(device)

            with torch.no_grad():
                gen = {"A": dA(conditional_sample(fmA, cond, num_steps=cfg["sample_steps"]).cpu().numpy()),
                       "B": dA(conditional_sample(fmB, cond, num_steps=cfg["sample_steps"],
                                                  quality=qB).cpu().numpy())}

            m = {}
            n = len(c_ev)
            for a in ARMS:
                g = gen[a]
                sp = smoothness_profile(g)
                pr, rc = precision_recall(c_ev.reshape(n, -1), g[:n].reshape(n, -1), k=args.pr_k)
                m[a] = {
                    "recall": rc, "prec": pr,
                    "wass": per_joint_wasserstein(g[:n], c_ev),
                    "inside": float(((g >= amin) & (g <= amax)).mean()),
                    "motion": sp["d1"] / ref["d1"],
                    "accel": sp["d2"] / ref["d2"],
                    "jerk3": sp["d3"] / ref["d3"],
                    "rough": sp["rough"] / ref["rough"],
                }

            n_div = min(cfg["diversity_conditions"], len(idx)); N = args.n_best
            cond_d = cond[:n_div]
            qB_d = torch.from_numpy(rng.choice(usable_q_pool, size=(N, n_div)).astype(np.float32)).to(device)
            with torch.no_grad():
                st = {"A": np.stack([dA(conditional_sample(fmA, cond_d, num_steps=cfg["sample_steps"]).cpu().numpy())
                                     for _ in range(N)], 0),
                      "B": np.stack([dA(conditional_sample(fmB, cond_d, num_steps=cfg["sample_steps"],
                                                           quality=qB_d[i][:, None]).cpu().numpy())
                                     for i in range(N)], 0)}
            for a in ARMS:
                m[a]["div"] = float(st[a].std(0).mean())
                m[a]["bestN"] = float(((st[a] - c_ev[:n_div][None]) ** 2).mean(axis=(2, 3)).min(0).mean())
                for mn in METRICS:
                    results[frac][mn][a].append(m[a][mn])

            print(f"seed {seed} frac {frac:.2f} | " + " | ".join(
                f"{mn} A={m['A'][mn]:.4f} B={m['B'][mn]:.4f}"
                for mn in ("recall", "prec", "div", "motion", "accel", "jerk3", "rough")))

    def fmt(frac, mn, a):
        v = np.array(results[frac][mn][a]); return f"{v.mean():.4f}+/-{ci95(v):.4f}"

    print("\n===== CHUNK 31: smoothness fidelity, A(filter) vs B(q-cond all) =====")
    print("ratios: 1.0 = matches real held-out usable. <1 = smoother than real.")
    print("motion = 1st diff (DIAL-DEPENDENT for B -- do not use as an A/B fidelity claim)")
    print("accel / jerk3 / rough = not the conditioning variable -> fair comparison\n")
    for frac in args.ep_fracs:
        print(f"-- train-episode fraction = {frac:.2f} --")
        print(f"{'metric':<10}{'A(filter)':>20}{'B(q-cond all)':>20}{'paired B-A':>22}{'sign':>8}")
        for mn in METRICS:
            a = np.array(results[frac][mn]["A"]); b = np.array(results[frac][mn]["B"])
            d = b - a
            if mn in ("motion", "accel", "jerk3", "rough"):
                # closeness to 1.0 is what matters
                closer = int((np.abs(b - 1) < np.abs(a - 1)).sum())
                tag = f"{closer}/{len(d)} closer to 1"
            else:
                higher = mn in ("recall", "prec", "div", "inside")
                tag = f"{int((d > 0).sum() if higher else (d < 0).sum())}/{len(d)} B"
            print(f"{mn:<10}{fmt(frac,mn,'A'):>20}{fmt(frac,mn,'B'):>20}"
                  f"{d.mean():>+14.4f} +/-{ci95(d):<6.4f}{tag:>18}")
        print()


if __name__ == "__main__":
    main()
