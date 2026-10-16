"""Review task A5: full 3x3 defect-injection matrix on all four main corpora, 5 draws each.

Protocol identical to the paper's RQ1 procedure (eq 4.2, 4.3), only extended: every corpus, every
(defect, axis) cell, draws 0-4. Per draw: 2,000 query chunks and a disjoint 6,000-chunk reference pool
(whole corpus), action noise = 0.5 x per-joint action std, swap = actions of another query chunk,
idle = first frame + 5% of the motion. AUC with the clean chunk as positive, ties averaged.

    .venv/Scripts/python.exe scripts/review_A5_defect_matrix.py
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from _bootstrap import add_project_root

add_project_root()

from run_chunk6_external import build_pairs_from_windows  # noqa: E402
from run_conditional_flow import norm_stats  # noqa: E402

CORPORA = ["being_so101", "rayenghali", "koch_pickplace5", "aloha_tape"]
DEFECTS = ["noise", "swap", "idle"]
AXES = ["activity", "smoothness", "consistency"]


def activity(c):
    return np.abs(np.diff(c, axis=1)).sum(2).mean(1)


def smoothness(c):
    return -np.abs(np.diff(c, axis=1, n=2)).sum(2).mean(1)


def knn(q, r, k, block=500):
    out = np.empty((len(q), k), np.int64)
    for i in range(0, len(q), block):
        d = ((q[i:i + block, None, :] - r[None]) ** 2).sum(-1)
        out[i:i + block] = np.argpartition(d, k, axis=1)[:, :k]
    return out


def consistency(act_n, nbr, ref_flat):
    qa = act_n.reshape(len(act_n), -1)
    return -np.sqrt(((qa[:, None, :] - ref_flat[nbr]) ** 2).sum(-1)).mean(1)


def auc(clean, corrupt):
    s = np.concatenate([clean, corrupt]); r = rankdata(s); n1, n0 = len(clean), len(corrupt)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    allres = {}
    for name in CORPORA:
        W = np.load(f"data_processed/windows/{name}_stateaction_w60_s30.npy").astype(np.float32)
        D = W.shape[2] // 2
        s_all, c_all = build_pairs_from_windows(W, 8, state_dim=D)
        s_m, s_s = norm_stats(s_all); a_m, a_s = norm_stats(c_all)
        nS = lambda x: (x - s_m) / s_s
        nA = lambda x: (x - a_m) / a_s
        res = np.zeros((5, 3, 3))
        for draw in range(5):
            rng = np.random.default_rng(draw)
            perm = rng.permutation(len(s_all)); ref, qry = perm[:6000], perm[6000:8000]
            ref_flat = nA(c_all[ref]).reshape(len(ref), -1)
            nbr = knn(nS(s_all[qry]), nS(s_all[ref]), 10)
            q = c_all[qry]
            corrupt = {"noise": q + rng.normal(0, 1, q.shape).astype(np.float32) * (0.5 * a_s.reshape(1, 1, D)),
                       "swap": q[rng.permutation(len(q))],
                       "idle": q[:, :1, :] + (q - q[:, :1, :]) * 0.05}
            score = {"activity": activity, "smoothness": smoothness,
                     "consistency": lambda c: consistency(nA(c), nbr, ref_flat)}
            for i, dfc in enumerate(DEFECTS):
                for j, ax in enumerate(AXES):
                    res[draw, i, j] = auc(score[ax](q), score[ax](corrupt[dfc]))
        allres[name] = res
        mu, sd = res.mean(0), res.std(0, ddof=1)
        print(f"===== {name} (D={D}, 2,000 chunks x 5 draws) — AUC mean ± sd (clean = positive)")
        print(f"{'defect':<8}" + "".join(f"{a:>20}" for a in AXES))
        for i, dfc in enumerate(DEFECTS):
            print(f"{dfc:<8}" + "".join(f"{mu[i, j]:>13.3f} ± {sd[i, j]:.3f}" for j in range(3)))
        print()
    pooled = np.concatenate([allres[n] for n in CORPORA], 0)  # 20 x 3 x 3
    print("===== pooled over 4 corpora x 5 draws — AUC mean ± sd")
    print(f"{'defect':<8}" + "".join(f"{a:>20}" for a in AXES))
    for i, dfc in enumerate(DEFECTS):
        print(f"{dfc:<8}" + "".join(f"{pooled[:, i, j].mean():>13.3f} ± {pooled[:, i, j].std(ddof=1):.3f}" for j in range(3)))
    print("\nDiagonal check (target axis = noise->smoothness, swap->consistency, idle->activity is the farthest from 0.5 in its row):")
    target = {"noise": 1, "swap": 2, "idle": 0}
    for n in CORPORA + ["pooled"]:
        m = pooled.mean(0) if n == "pooled" else allres[n].mean(0)
        row_ok = [int(np.argmax(np.abs(m[i] - 0.5))) == target[d] for i, d in enumerate(DEFECTS)]
        print(f"  {n:<16} " + " ".join(f"{d}:{'yes' if ok else 'NO'}" for d, ok in zip(DEFECTS, row_ok)))


if __name__ == "__main__":
    main()
