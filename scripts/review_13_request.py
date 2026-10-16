"""Report 13: request (q-draw) refinement for QC-S without retraining — pre-registered in
outputs/campaign_dedup/rq13/prereg.md (written before any result).

Candidates: K_full in {8, 16, 32, 64} x {uniform, distance-weighted (Gaussian, h = median neighbour distance)}.
  val     : selection on the 1/2-scale models (report 11b) with validation states from the training episodes NOT in the
            1/2 subset (de-duplicated, usable under the full-pool tau, <= 1,500); K_val = K_full / 2. R is not touched.
  confirm : the selected candidate and the current QC-S (K 32 uniform), freshly generated on R with the full models.

    .venv/Scripts/python.exe scripts/review_13_request.py --stage val
    .venv/Scripts/python.exe scripts/review_13_request.py --stage confirm --k 32 --weighted 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from _bootstrap import add_project_root

add_project_root()

from src.eval.distribution_metrics import _pairwise_dist  # noqa: E402
from src.train.train_lstm_vae import episode_split  # noqa: E402
from run_chunk6_external import build_pairs_from_windows  # noqa: E402
from review_dedup_reeval import first_unique  # noqa: E402
from review_11_rq import Cell, evaluate, gen, load_fm, subset_eps  # noqa: E402

OUT = Path("outputs/campaign_dedup/rq13")
D9 = Path("outputs/campaign_dedup")
KS_FULL = [8, 16, 32, 64]
KEEP = ("recall", "precision", "diversity", "best_of_N", "inside", "lowq_G")


def knn_dist(x, ref, k, block=512):
    idx = np.empty((len(x), k), np.int64); dist = np.empty((len(x), k), np.float32)
    for i in range(0, len(x), block):
        d = _pairwise_dist(x[i:i + block], ref)
        p = np.argpartition(d, k, axis=1)[:, :k]
        idx[i:i + block] = p; dist[i:i + block] = np.take_along_axis(d, p, 1)
    return idx, dist


def make_pools(cell, states, K, weighted):
    """(q, prob) per state from usable chunks among the K nearest training states; fallback = Q uniform."""
    ax = cell.ax
    nbr, dist = knn_dist(cell.nS(states), cell.nS(cell.s_tr), K)
    pools, fb = [], 0
    for i in range(len(states)):
        ok = ax.usable_tr[nbr[i]]
        if not ok.any():
            fb += 1; pools.append((ax.Q, None)); continue
        q = ax.q_tr[nbr[i][ok]]
        if weighted:
            h = float(np.median(dist[i]))
            d2 = dist[i][ok].astype(np.float64) ** 2   # shift by the min: same normalised weights, no underflow
            w = np.exp(-(d2 - d2.min()) / (2 * h * h)) if h > 0 else np.ones(ok.sum())
            pools.append((q, w / w.sum()))
        else:
            pools.append((q, None))
    return pools, fb / len(states)


def q_fn(pools, rs, device):
    def fn(shape):
        n_states = shape[-1]; lead = shape[:-1]
        out = np.empty((*shape, 2), np.float32)
        for i in range(n_states):
            q, p = pools[i]
            pick = q[rs.choice(len(q), size=lead or (1,), p=p)]
            if lead: out[:, i] = pick
            else: out[i] = pick[0]
        return torch.from_numpy(out).to(device)
    return fn


def run_val(corpus, seed, cfg, cc, device):
    tag = f"{corpus}_s{seed}"; out_p = OUT / "val" / f"{tag}.json"
    if out_p.exists():
        print(f"[skip] {tag}"); return
    t0 = time.time(); steps, N = cfg["sample_steps"], cc["n_best"]
    half = Cell(corpus, 0.5, seed, cfg); tau_full = Cell(corpus, 1.0, seed, cfg).ax.tau
    W = np.load(f"data_processed/windows/{corpus}_stateaction_w60_s30.npy").astype(np.float32)
    we = np.load(f"data_processed/stats/{corpus}_stateaction_w60_s30_stats.npz", allow_pickle=True)["window_episode"]
    tr_w, _, _, _ = episode_split(we, cfg["val_frac"], seed)
    tr_eps = np.unique(we[tr_w])
    rest = np.setdiff1d(tr_eps, subset_eps(tr_eps, 0.5, seed))
    assert len(np.intersect1d(rest, half.keep)) == 0
    s_v, c_v = build_pairs_from_windows(W[tr_w[np.isin(we[tr_w], rest)]], half.k, state_dim=half.D)
    from run_review_campaign import act_raw
    u = first_unique(s_v, c_v); u = u[act_raw(c_v[u]) >= tau_full]
    rs = np.random.default_rng([seed, 13, 1]); torch.manual_seed(13_000 + seed)
    idx = rs.permutation(u)[: cfg["eval_conditions"]]
    c_ev = c_v[idx]; cond = torch.from_numpy(half.nS(s_v[idx])).to(device)
    n_div = min(cfg["diversity_conditions"], len(idx))
    ck = Path("outputs/campaign_dedup/rq11/ckpt") / f"{corpus}_f0.50_s{seed}"
    fA, fQ = load_fm(ck / "A.pt", cfg, half, 0, device), load_fm(ck / "B2.pt", cfg, half, 2, device)
    res = {"corpus": corpus, "seed": seed, "n_val": int(len(idx)), "n_rest_eps": int(len(rest)), "cand": {}}
    m = evaluate(*gen(fA, cond, None, steps, N, n_div, half), c_ev, half, half.ax, cc["pr_k"])
    res["F"] = {k: m[k] for k in KEEP}
    for Kf in KS_FULL:
        for wt in (0, 1):
            pools, fb = make_pools(half, s_v[idx], Kf // 2, wt)
            m = evaluate(*gen(fQ, cond, q_fn(pools, rs, device), steps, N, n_div, half), c_ev, half, half.ax, cc["pr_k"])
            res["cand"][f"K{Kf}_w{wt}"] = {**{k: m[k] for k in KEEP}, "fallback": fb}
    res["seconds"] = time.time() - t0
    out_p.write_text(json.dumps(res), encoding="utf-8")
    c = res["cand"]
    print(f"[val] {tag} n={len(idx)} {res['seconds'] / 60:.1f} min | " + " ".join(
        f"{k}:R{v['recall']:.3f}/P{v['precision']:.3f}" for k, v in c.items()) + f" | F R{res['F']['recall']:.3f}/P{res['F']['precision']:.3f}", flush=True)


def run_confirm(corpus, seed, cfg, cc, device, K, wt):
    tag = f"{corpus}_f1.00_s{seed}"; out_p = OUT / "confirm" / f"{tag}_K{K}_w{wt}.json"
    if out_p.exists():
        print(f"[skip] {tag}"); return
    t0 = time.time(); steps, N = cfg["sample_steps"], cc["n_best"]
    cell = Cell(corpus, 1.0, seed, cfg)
    idx = np.load(D9 / "samples" / tag / "A.npz")["state_idx"]
    c_ev = cell.c_te[idx]; cond = torch.from_numpy(cell.nS(cell.s_te[idx])).to(device)
    n_div = min(cfg["diversity_conditions"], len(idx))
    fQ = load_fm(Path(cc["out_dir"]) / "ckpt" / tag / "B2.pt", cfg, cell, 2, device)
    res = {"corpus": corpus, "seed": seed, "selected": f"K{K}_w{wt}", "arms": {}}
    for name, (k_, w_) in (("current", (32, 0)), ("selected", (K, wt))):
        rs = np.random.default_rng([seed, 13, 2]); torch.manual_seed(13_500 + seed)   # same stream for both
        pools, fb = make_pools(cell, cell.s_te[idx], k_, w_)
        m = evaluate(*gen(fQ, cond, q_fn(pools, rs, device), steps, N, n_div, cell), c_ev, cell, cell.ax, cc["pr_k"])
        res["arms"][name] = {**{kk: m[kk] for kk in KEEP}, "fallback": fb}
    res["seconds"] = time.time() - t0
    out_p.write_text(json.dumps(res), encoding="utf-8")
    a = res["arms"]
    print(f"[confirm] {tag} | current R{a['current']['recall']:.3f}/P{a['current']['precision']:.3f} "
          f"selected R{a['selected']['recall']:.3f}/P{a['selected']['precision']:.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/review_campaign.yaml")
    p.add_argument("--stage", choices=["val", "confirm"], required=True)
    p.add_argument("--k", type=int, default=32); p.add_argument("--weighted", type=int, default=0)
    p.add_argument("--corpora", nargs="+"); p.add_argument("--seeds", type=int, nargs="+")
    a = p.parse_args()
    cc = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(cc["base_config"]).read_text(encoding="utf-8"))
    for d in ("val", "confirm"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for corpus in a.corpora or cc["corpora"]:
        for seed in a.seeds if a.seeds is not None else cc["seeds"]:
            if a.stage == "val":
                run_val(corpus, seed, cfg, cc, device)
            else:
                run_confirm(corpus, seed, cfg, cc, device, a.k, a.weighted)


if __name__ == "__main__":
    main()
