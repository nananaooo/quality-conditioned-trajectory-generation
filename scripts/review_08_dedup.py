"""Review 08: effect of duplicated chunks (overlapping w60/s30 windows) on Table 1 counts and P/R.

build_pairs_from_windows takes every start t in 0..51 of each 60-frame window; consecutive windows
overlap by 30 frames, so starts 30..51 (22 of 52 = 42%) reappear in the next window. Nothing in the
campaign removes them. Here:
  (1) unique (state, chunk) count per corpus vs Table 1 count;
  (2) per campaign cell, re-evaluate precision/recall of the saved T=1 samples (A, C, B2)
      with the eval target set as used (original) and with duplicate targets dropped (keep first
      occurrence, drop the paired generated sample too). Verifies original == stored first.
Analysis only; uses saved samples, no generation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from _bootstrap import add_project_root

add_project_root()

from src.eval.distribution_metrics import precision_recall  # noqa: E402
from src.train.train_lstm_vae import episode_split  # noqa: E402
from run_chunk6_external import build_pairs_from_windows  # noqa: E402

CORP = ["being_so101", "rayenghali", "koch_pickplace5", "aloha_tape"]
OUT = Path("outputs/campaign")


def first_unique(x):
    _, first = np.unique(x.reshape(len(x), -1), axis=0, return_index=True)
    return np.sort(first)


def main():
    cfg = yaml.safe_load(open("configs/conditional_flow_physicai.yaml", encoding="utf-8"))
    cc = yaml.safe_load(open("configs/review_campaign.yaml", encoding="utf-8"))
    k, pr_k = cfg["chunk_size"], cc["pr_k"]

    print("(1) unique chunks per corpus")
    for c in CORP + ["physicai"]:
        W = np.load(f"data_processed/windows/{c}_stateaction_w60_s30.npy").astype(np.float32)
        D = W.shape[2] // 2
        s, ch = build_pairs_from_windows(W, k, state_dim=D)
        sc = np.concatenate([s, ch.reshape(len(ch), -1)], 1)
        u = len(first_unique(sc))
        print(f"  {c:<18} chunks={len(ch):>6} unique(state,chunk)={u:>6} ratio={len(ch)/u:.3f}")

    print("\n(2) P/R with and without duplicate eval targets (T=1 saved samples)")
    rows = {}
    for c in CORP:
        W = np.load(f"data_processed/windows/{c}_stateaction_w60_s30.npy").astype(np.float32)
        we = np.load(f"data_processed/stats/{c}_stateaction_w60_s30_stats.npz", allow_pickle=True)["window_episode"]
        D = W.shape[2] // 2
        for sc in [1.0, 0.25]:
            for seed in range(5):
                tag = f"{c}_f{sc:.2f}_s{seed}"
                res = json.load(open(OUT / "results" / f"{tag}.json"))
                _, te_w, _, _ = episode_split(we, cfg["val_frac"], seed)
                _, c_te = build_pairs_from_windows(W[te_w], k, state_dim=D)
                for arm in ["A", "C", "B2"]:
                    z = np.load(OUT / "samples" / tag / f"{arm}.npz")
                    idx, g = z["state_idx"], z["gen"]
                    real = c_te[idx]; m = len(real)
                    p0, r0 = precision_recall(real.reshape(m, -1), g[:m].reshape(m, -1), k=pr_k)
                    st = res["arms"][arm]["main"]
                    assert abs(p0 - st["precision"]) < 1e-6 and abs(r0 - st["recall"]) < 1e-6, (tag, arm, p0, st["precision"])
                    keep = first_unique(real)
                    p1, r1 = precision_recall(real[keep].reshape(len(keep), -1), g[keep].reshape(len(keep), -1), k=pr_k)
                    rows.setdefault((c, sc), {}).setdefault(arm, []).append((p0, r0, p1, r1, 1 - len(keep) / m))
    for (c, sc), d in rows.items():
        A, B = np.array(d["A"]), np.array(d["B2"])
        dup = A[:, 4].mean()
        line = f"  {c:<16} {sc:<4} dup-targets={dup*100:4.1f}% |"
        for j, name in [(1, "recall"), (0, "precision")]:
            d0 = B[:, j] - A[:, j]; d1 = B[:, j + 2] - A[:, j + 2]
            p1 = stats.ttest_1samp(d1, 0).pvalue
            line += (f" {name}: F {A[:, j].mean():.3f}->{A[:, j+2].mean():.3f}, QC-M {B[:, j].mean():.3f}->{B[:, j+2].mean():.3f},"
                     f" Δ {d0.mean():+.3f}->{d1.mean():+.3f}{'*' if p1 < 0.05 else ''} |")
        print(line)


if __name__ == "__main__":
    main()
