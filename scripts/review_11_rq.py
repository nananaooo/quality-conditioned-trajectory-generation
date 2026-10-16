"""Report 11 (v20 RQ restructure): evaluation inputs for stage 1 (re-analysis, report 11a) and stage 2
(RQ3 scale experiment, report 11b). Reuses the report-09 code path, config, seeds, split, normalisation,
checkpoints and saved samples; metric definitions and evaluation settings are unchanged
(k_NN = 3, |G| = |R| = 1,500, 256 states x N = 8, T = 1, Euler 50, 32 state neighbours + fallback).

Stages
  full    : full-data cells. REPLAYS the report-09 main-arm generation with the identical RNG stream (verified
            against the saved A/C/B2/QCs samples), which recovers the Activity-Conditioned samples that were not
            saved; then, with a separate RNG, generates AC-S (Activity-Conditioned with q from the same 32 state
            neighbours as QC-S) and QC-S at the action-consistency target (replayed from the filter-variant run).
            Low-quality fraction (activity < tau) and activity/smoothness z means for F, FU, AC, AC-S, QC-M, QC-S.
  common  : scales 0.25 / 0.5 / 0.125 evaluated on the COMMON target R = the full-data target of the same seed
            (held-out split + full-training-pool tau; identical 1,500 states at every scale). Training-side tau and
            the usable pool Q (Filtered's criterion, QC-S's q pool) come from the subset, as in the campaign.
            0.25 reuses the campaign + filter-variant checkpoints and evaluates every stage-1 quantity;
            0.5 / 0.125 train Filtered and Quality-Conditioned on nested episode subsets (1/8 c 1/4 c 1/2 c full).

    .venv/Scripts/python.exe scripts/review_11_rq.py --stage full
    .venv/Scripts/python.exe scripts/review_11_rq.py --stage common --scales 0.25
    .venv/Scripts/python.exe scripts/review_11_rq.py --stage common --scales 0.5 0.125
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

from src.eval.distribution_metrics import precision_recall  # noqa: E402
from src.train.train_lstm_vae import episode_split  # noqa: E402
from run_conditional_flow import norm_stats  # noqa: E402
from run_chunk6_external import build_pairs_from_windows  # noqa: E402
from run_chunk31_smoothness_fidelity import smoothness_profile  # noqa: E402
from run_review_campaign import QDIM, Axes, act_raw, jerk_raw, knn_idx, main_metrics, make_fm, sample, train  # noqa: E402
from review_dedup_reeval import ARMS, K_STATE, TEMPS, first_unique  # noqa: E402
from review_dedup_filter_variants import N_REF, consistency, pair_ids  # noqa: E402

OUT = Path("outputs/campaign_dedup/rq11")
DEDUP = Path("outputs/campaign_dedup")
SKEY = {1.0: 100, 0.5: 50, 0.25: 25, 0.125: 12}      # integer key of a scale in seeds / tags
TAG = {1.0: "1.00", 0.5: "0.50", 0.25: "0.25", 0.125: "0.125"}
KEEP = ("recall", "precision", "diversity", "best_of_N", "inside")


# ----------------------------------------------------------------------------- data
def subset_eps(tr_eps, scale, seed):
    """Nested episode subsets. 1 and 1/4 are the campaign draws; 1/2 = 1/4 + 25% of the training episodes drawn
    from the rest; 1/8 = half of the 1/4 subset."""
    n = len(tr_eps)
    if scale == 1.0:
        return np.random.default_rng([seed, 100]).choice(tr_eps, size=n, replace=False)
    q = np.random.default_rng([seed, 25]).choice(tr_eps, size=max(1, int(round(n * 0.25))), replace=False)
    if scale == 0.25:
        return q
    if scale == 0.5:
        rest = np.setdiff1d(tr_eps, q)
        return np.concatenate([q, np.random.default_rng([seed, 50, 11]).choice(rest, size=int(round(n * 0.5)) - len(q), replace=False)])
    if scale == 0.125:
        return np.random.default_rng([seed, 12, 11]).choice(q, size=max(1, int(round(len(q) / 2))), replace=False)
    raise ValueError(scale)


class Cell:
    def __init__(self, corpus, scale, seed, cfg):
        self.corpus, self.scale, self.seed, self.k = corpus, scale, seed, cfg["chunk_size"]
        W = np.load(f"data_processed/windows/{corpus}_stateaction_w60_s30.npy").astype(np.float32)
        we = np.load(f"data_processed/stats/{corpus}_stateaction_w60_s30_stats.npz", allow_pickle=True)["window_episode"]
        self.D = D = W.shape[2] // 2
        tr_w, te_w, _, _ = episode_split(we, cfg["val_frac"], seed)
        tr_eps = np.unique(we[tr_w])
        self.keep = np.sort(subset_eps(tr_eps, scale, seed)); self.n_eps_total = len(tr_eps)
        tr_w = tr_w[np.isin(we[tr_w], self.keep)]
        self.s_tr, self.c_tr = build_pairs_from_windows(W[tr_w], self.k, state_dim=D)
        self.s_te, self.c_te = build_pairs_from_windows(W[te_w], self.k, state_dim=D)
        self.s_mean, self.s_std = norm_stats(self.s_tr); self.a_mean, self.a_std = norm_stats(self.c_tr)
        self.ax = Axes(self.c_tr, lambda c: c)
        self.amin, self.amax = self.c_tr.reshape(-1, D).min(0), self.c_tr.reshape(-1, D).max(0)
        self.bs = min(cfg["batch_size"], max(8, len(self.s_tr) // 4))

    def nS(self, x): return ((x - self.s_mean) / self.s_std).astype(np.float32)
    def nA(self, x): return ((x - self.a_mean) / self.a_std).astype(np.float32)
    def dA(self, x): return x * self.a_std + self.a_mean


def load_fm(path, cfg, cell, qd, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    assert np.allclose(ck["a_mean"], cell.a_mean) and np.allclose(ck["s_std"], cell.s_std), f"{path}: data rebuild mismatch"
    fm = make_fm(cfg, cell.D, qd).to(device); fm.load_state_dict(ck["state"]); fm.eval()
    return fm


# ----------------------------------------------------------------------------- generation helpers
def gen(fm, cond, qfn, steps, N, n_div, cell, T=1.0):
    """Identical call order to review_dedup_reeval.gen (q for G, sample G, q for the stack, N stack samples)."""
    g = cell.dA(sample(fm, cond, None if qfn is None else qfn((len(cond),)), steps, T).cpu().numpy())
    qs = None if qfn is None else qfn((N, n_div))
    st = np.stack([cell.dA(sample(fm, cond[:n_div], None if qs is None else qs[i], steps, T).cpu().numpy())
                   for i in range(N)], 0)
    return g, st


def q_dist(Q, qd, rs, device):
    def fn(shape):
        return torch.from_numpy(Q[rs.integers(0, len(Q), shape)][..., :qd].astype(np.float32)).to(device)
    return fn


def state_pools(cell, states, ax):
    """q pools from the usable chunks among the K_STATE nearest training states (normalised); fallback = Q."""
    nbr = knn_idx(cell.nS(states), cell.nS(cell.s_tr), K_STATE)
    pools, fb = [], 0
    for i in range(len(states)):
        u = nbr[i][ax.usable_tr[nbr[i]]]
        if len(u) == 0:
            fb += 1; pools.append(ax.Q)
        else:
            pools.append(ax.q_tr[u])
    return pools, fb / len(states)


def q_state(pools, qd, rs, device):
    def fn(shape):
        n_states = shape[-1]; lead = shape[:-1]
        out_q = np.empty((*shape, qd), np.float32)
        for i in range(n_states):
            pick = pools[i][rs.integers(0, len(pools[i]), lead or (1,))][..., :qd]
            if lead: out_q[:, i] = pick
            else: out_q[i] = pick[0]
        return torch.from_numpy(out_q).to(device)
    return fn


def q_per_state(Q, rs, device):
    def fn(shape):
        base = Q[rs.integers(0, len(Q), shape[-1])]
        return torch.from_numpy(np.broadcast_to(base, (*shape, 2)).astype(np.float32).copy()).to(device)
    return fn


def q_global(Q, device):
    med = np.median(Q, axis=0).astype(np.float32)
    def fn(shape):
        return torch.from_numpy(np.broadcast_to(med, (*shape, 2)).astype(np.float32).copy()).to(device)
    return fn


def lowq(g, st, axR):
    """Low-quality fraction (activity < tau of the target definition) and q z-means under the target's axis."""
    flat = st.reshape(-1, *st.shape[2:])
    qg, qd = axR.q_of(act_raw(g), jerk_raw(g)), axR.q_of(act_raw(flat), jerk_raw(flat))
    return {"lowq_G": float((act_raw(g) < axR.tau).mean()), "lowq_div": float((act_raw(flat) < axR.tau).mean()),
            "zact_G": float(qg[:, 0].mean()), "zsmooth_G": float(qg[:, 1].mean()),
            "zact_div": float(qd[:, 0].mean()), "zsmooth_div": float(qd[:, 1].mean())}


def evaluate(g, st, c_ev, cell, axR, pr_k):
    m = main_metrics(g, c_ev, st, cell.amin, cell.amax, smoothness_profile(c_ev), pr_k)
    return {**{kk: m[kk] for kk in KEEP}, **lowq(g, st, axR)}


def output_filtered(fm, cond, c_ev, cell, ax, steps, N, n_div, tries, pr_k):
    """Report-09 Output-Filtered: rejection on Full-Unconditioned with the training tau, max `tries` draws;
    PR on the states with an accepted sample, diversity / best-of-N on the states that filled N."""
    k, D = cell.k, cell.D
    acc = np.full((len(cond), k, D), np.nan, np.float32); have = np.zeros(len(cond), bool); n_try = n_acc = 0
    for _ in range(tries):
        todo = np.where(~have)[0]
        if len(todo) == 0: break
        x = cell.dA(sample(fm, cond[todo], None, steps).cpu().numpy())
        ok = act_raw(x) >= ax.tau; n_try += len(todo); n_acc += int(ok.sum())
        acc[todo[ok]] = x[ok]; have[todo[ok]] = True
    div_acc = [[] for _ in range(n_div)]
    for _ in range(tries):
        todo = np.array([i for i in range(n_div) if len(div_acc[i]) < N])
        if len(todo) == 0: break
        x = cell.dA(sample(fm, cond[todo], None, steps).cpu().numpy())
        ok = act_raw(x) >= ax.tau
        for j, i in enumerate(todo):
            if ok[j]: div_acc[i].append(x[j])
    full = [i for i in range(n_div) if len(div_acc[i]) >= N]
    sel = np.where(have)[0]
    stO = np.stack([np.stack(div_acc[i][:N], 0) for i in full], 1)
    pr, rc = precision_recall(c_ev[sel].reshape(len(sel), -1), acc[sel].reshape(len(sel), -1), k=pr_k)
    return {"recall": rc, "precision": pr, "diversity": float(stO.std(0).mean()),
            "best_of_N": float(((stO - c_ev[full][None]) ** 2).mean(axis=(2, 3)).min(0).mean()),
            "inside": float(((acc[sel] >= cell.amin) & (acc[sel] <= cell.amax)).mean()),
            "accept_rate_pr": n_acc / max(n_try, 1), "frac_states_no_accept": float(1 - have.mean()),
            "frac_div_states_short": float(1 - len(full) / n_div)}


def con_target(cell, seed, scale, cfg, cc):
    """Replays review_dedup_filter_variants' RNG stream up to the consistency-target draw (same cell)."""
    rs = np.random.default_rng([seed, int(round(scale * 100)), 34])
    _, tr_id = pair_ids(cell.s_tr, cell.c_tr)
    ref = rs.permutation(len(cell.s_tr))[: min(N_REF, len(cell.s_tr))]
    rS, rA, r_id = cell.nS(cell.s_tr[ref]), cell.nA(cell.c_tr[ref]).reshape(len(ref), -1), tr_id[ref]
    te_first, _ = pair_ids(cell.s_te, cell.c_te)
    con_te_u = consistency(cell.nS(cell.s_te[te_first]), cell.nA(cell.c_te[te_first]).reshape(len(te_first), -1),
                           np.full(len(te_first), -1), rS, rA, r_id)
    rs.permutation(np.where(cell.ax.usable(cell.c_te))[0])          # val_idx draw of the original run
    thr = float(np.median(con_te_u))
    idx = rs.permutation(te_first[con_te_u >= thr])[: cfg["eval_conditions"]]
    return idx, thr


def check_con(idx, thr, cell, fv_json):
    fv = json.loads(fv_json.read_text(encoding="utf-8"))
    ok = abs(fv["con_threshold_heldout_unique_median"] - thr) < 1e-5 and \
        abs(fv["con_target_act_usable_frac"] - float(cell.ax.usable(cell.c_te[idx]).mean())) < 1e-9
    assert ok, f"{fv_json}: consistency-target replay mismatch"
    return fv


# ----------------------------------------------------------------------------- stage: full
def run_full(corpus, seed, cfg, cc, device):
    tag = f"{corpus}_f1.00_s{seed}"; out_p = OUT / "full" / f"{tag}.json"
    if out_p.exists():
        print(f"[skip] {tag}"); return
    t0 = time.time()
    steps, N = cfg["sample_steps"], cc["n_best"]
    cell = Cell(corpus, 1.0, seed, cfg)
    axes = {"raw": cell.ax, "z": Axes(cell.c_tr, cell.nA)}
    fms = {arm: load_fm(Path(cc["out_dir"]) / "ckpt" / tag / f"{arm}.pt", cfg, cell, QDIM[arm], device) for arm in ARMS}

    # ---- replay of review_dedup_reeval.run_cell up to the end of the main arms (identical RNG stream)
    rs = np.random.default_rng([seed, 100, 23])
    torch.manual_seed(1_000 + seed * 10 + 4)
    uniq = first_unique(cell.s_te, cell.c_te)
    is_u = np.zeros(len(cell.c_te), bool); is_u[uniq] = True
    tgt = {}
    for n, ax in axes.items():
        idx = rs.permutation(np.where(ax.usable(cell.c_te) & is_u)[0])[: cfg["eval_conditions"]]
        tgt[n] = dict(idx=idx, cond=torch.from_numpy(cell.nS(cell.s_te[idx])).to(device))
    n_div = min(cfg["diversity_conditions"], cfg["eval_conditions"])
    idx = tgt["raw"]["idx"]; c_ev = cell.c_te[idx]; cond = tgt["raw"]["cond"]
    pools, fb = state_pools(cell, cell.s_te[idx], cell.ax)
    samples = {}
    for arm in ARMS:
        n = "z" if arm.endswith("_z") else "raw"
        qfn = q_dist(axes[n].Q, QDIM[arm], rs, device) if QDIM[arm] else None
        samples[arm] = gen(fms[arm], tgt[n]["cond"], qfn, steps, N, n_div, cell)
    samples["QCs"] = gen(fms["B2"], cond, q_state(pools, 2, rs, device), steps, N, n_div, cell)

    saved = DEDUP / "samples" / tag
    dev = {}
    for arm in ("A", "C", "B2", "QCs"):
        z = np.load(saved / f"{arm}.npz")
        assert np.array_equal(z["state_idx"], idx)
        dev[arm] = float(max(np.abs(z["gen"] - samples[arm][0]).max(), np.abs(z["div"] - samples[arm][1]).max()))
    old = json.loads((DEDUP / "results" / f"{tag}.json").read_text(encoding="utf-8"))
    b1 = main_metrics(*samples["B1"][:1], c_ev, samples["B1"][1], cell.amin, cell.amax, smoothness_profile(c_ev), cc["pr_k"])
    dev["B1_recall_vs_json"] = abs(b1["recall"] - old["arms"]["B1"]["main"]["recall"])
    dev["B1_diversity_vs_json"] = abs(b1["diversity"] - old["arms"]["B1"]["main"]["diversity"])
    np.savez_compressed(OUT / "samples" / f"{tag}_B1.npz", gen=samples["B1"][0].astype(np.float32),
                        div=samples["B1"][1].astype(np.float32), state_idx=idx)

    res = {"corpus": corpus, "scale": 1.0, "seed": seed, "target": "R_full", "tau_R": cell.ax.tau,
           "replay_max_abs_dev": dev, "state_cond_fallback_frac": fb, "arms": {}}
    for arm in ("A", "C", "B1", "B2", "QCs"):
        res["arms"][arm] = evaluate(*samples[arm], c_ev, cell, cell.ax, cc["pr_k"])

    # ---- new draws (separate RNG): AC-S, QC-S at the consistency target
    rs2 = np.random.default_rng([seed, 100, 111]); torch.manual_seed(11_000 + seed * 10 + 4)
    g, st = gen(fms["B1"], cond, q_state(pools, 1, rs2, device), steps, N, n_div, cell)
    res["arms"]["ACs"] = evaluate(g, st, c_ev, cell, cell.ax, cc["pr_k"])
    np.savez_compressed(OUT / "samples" / f"{tag}_ACs.npz", gen=g.astype(np.float32), div=st.astype(np.float32), state_idx=idx)

    cidx, thr = con_target(cell, seed, 1.0, cfg, cc)
    check_con(cidx, thr, cell, DEDUP / "filter_variants" / f"{tag}.json")
    cpools, cfb = state_pools(cell, cell.s_te[cidx], cell.ax)
    g, st = gen(fms["B2"], torch.from_numpy(cell.nS(cell.s_te[cidx])).to(device), q_state(cpools, 2, rs2, device), steps, N, n_div, cell)
    res["con_target"] = {"QCs": evaluate(g, st, cell.c_te[cidx], cell, cell.ax, cc["pr_k"]), "fallback": cfb}

    # ---- references: held-out chunks and the target R
    first, _ = pair_ids(cell.s_te, cell.c_te)
    qR = cell.ax.q_of(act_raw(c_ev), jerk_raw(c_ev))
    res["reference"] = {"heldout_lowq_all": float((act_raw(cell.c_te) < cell.ax.tau).mean()),
                        "heldout_lowq_unique": float((act_raw(cell.c_te[first]) < cell.ax.tau).mean()),
                        "R_lowq": float((act_raw(c_ev) < cell.ax.tau).mean()),
                        "R_zact": float(qR[:, 0].mean()), "R_zsmooth": float(qR[:, 1].mean())}
    res["seconds"] = time.time() - t0
    out_p.write_text(json.dumps(res), encoding="utf-8")
    print(f"[done] {tag} {res['seconds'] / 60:.1f} min | replay dev {max(v for k_, v in dev.items() if not k_.startswith('B1')):.2e} "
          f"B1 drec {dev['B1_recall_vs_json']:.1e} | lowq F {res['arms']['A']['lowq_G']:.3f} QCs {res['arms']['QCs']['lowq_G']:.3f} "
          f"| ACs rec {res['arms']['ACs']['recall']:.3f} AC {res['arms']['B1']['recall']:.3f}", flush=True)


# ----------------------------------------------------------------------------- stage: common target, sub-scales
def common_target(corpus, seed, cfg):
    """Full-data target R of report 09 (same seed): first rs draw of review_dedup_reeval at scale 1."""
    full = Cell(corpus, 1.0, seed, cfg)
    rs = np.random.default_rng([seed, 100, 23])
    is_u = np.zeros(len(full.c_te), bool); is_u[first_unique(full.s_te, full.c_te)] = True
    idx = rs.permutation(np.where(full.ax.usable(full.c_te) & is_u)[0])[: cfg["eval_conditions"]]
    saved = np.load(DEDUP / "samples" / f"{corpus}_f1.00_s{seed}" / "A.npz")["state_idx"]
    assert np.array_equal(idx, saved), "common target differs from the report-09 full-data target"
    return idx, full.ax


def train_arm(cell, arm, cfg, device, seed):
    """Campaign training of Filtered (A) or Quality-Conditioned (B2) on the cell's subset."""
    k, D, ax = cell.k, cell.D, cell.ax
    all_s, all_c, all_q = torch.from_numpy(cell.nS(cell.s_tr)), torch.from_numpy(cell.nA(cell.c_tr)), torch.from_numpy(ax.q_tr)
    if arm == "A":
        u = torch.from_numpy(ax.usable_tr); tensors = (all_s[u], all_c[u], all_q[u])
    else:
        tensors = (all_s, all_c, all_q)
    vr = np.random.default_rng([seed, SKEY[cell.scale], 7])
    vidx = vr.permutation(np.where(ax.usable(cell.c_te))[0])[:2048]
    vq = ax.q_of(act_raw(cell.c_te[vidx]), jerk_raw(cell.c_te[vidx]))
    g = torch.Generator().manual_seed(10_000 + seed)
    val = (torch.from_numpy(cell.nS(cell.s_te[vidx])).to(device), torch.from_numpy(cell.nA(cell.c_te[vidx])).to(device),
           torch.from_numpy(vq).to(device), torch.randn(len(vidx), k, D, generator=g).to(device), torch.rand(len(vidx), generator=g).to(device))
    t0 = time.time()
    fm, h = train(cfg, D, QDIM[arm], tensors, cell.bs, cfg["epochs"], seed, cfg.get("lambda_smooth", 0.0), device, val)
    h["epochs"], h["train_seconds"] = cfg["epochs"], time.time() - t0
    return fm, h


def run_common(corpus, scale, seed, cfg, cc, device):
    tag = f"{corpus}_f{TAG[scale]}_s{seed}"; out_p = OUT / "common" / f"{tag}.json"
    if out_p.exists():
        print(f"[skip] {tag}"); return
    t0 = time.time()
    steps, N, sk = cfg["sample_steps"], cc["n_best"], SKEY[scale]
    cell = Cell(corpus, scale, seed, cfg)
    idx, axR = common_target(corpus, seed, cfg)
    c_ev = cell.c_te[idx]; cond = torch.from_numpy(cell.nS(cell.s_te[idx])).to(device)
    n_div = min(cfg["diversity_conditions"], cfg["eval_conditions"])
    res = {"corpus": corpus, "scale": scale, "seed": seed, "target": "R_full", "tau_R": axR.tau, "tau_train": cell.ax.tau,
           "n_eps": int(len(cell.keep)), "n_eps_total": int(cell.n_eps_total), "episodes": cell.keep.tolist(),
           "n_train_pairs": int(len(cell.s_tr)), "batch_size": int(cell.bs),
           "usable_frac_train": float(cell.ax.usable_tr.mean()), "R_usable_under_train_tau": float(cell.ax.usable(c_ev).mean()),
           "arms": {}, "train": {}}

    fms = {}
    if scale == 0.25:
        ck = Path(cc["out_dir"]) / "ckpt" / f"{corpus}_f0.25_s{seed}"
        for arm in ("A", "A_step", "C", "B1", "B2"):
            fms[arm] = load_fm(ck / f"{arm}.pt", cfg, cell, QDIM[arm], device)
            h = torch.load(ck / f"{arm}.pt", map_location="cpu", weights_only=False)["hist"]
            res["train"][arm] = {"epochs": h["epochs"], "steps": h["steps"]}
        for arm in ("A25", "A75", "A_con"):
            fms[arm] = load_fm(DEDUP / "ckpt_fv" / f"{corpus}_f0.25_s{seed}" / f"{arm}.pt", cfg, cell, 0, device)
    else:
        ck = OUT / "ckpt" / tag; ck.mkdir(parents=True, exist_ok=True)
        for arm in ("A", "B2"):
            p = ck / f"{arm}.pt"
            if p.exists():
                fms[arm] = load_fm(p, cfg, cell, QDIM[arm], device); h = torch.load(p, map_location="cpu", weights_only=False)["hist"]
            else:
                fm, h = train_arm(cell, arm, cfg, device, seed)
                torch.save({"state": fm.state_dict(), "hist": h, "qdim": QDIM[arm], "D": cell.D, "s_mean": cell.s_mean,
                            "s_std": cell.s_std, "a_mean": cell.a_mean, "a_std": cell.a_std, "axes": "raw",
                            "episodes": cell.keep}, p)
                fms[arm] = fm
            res["train"][arm] = {"epochs": h["epochs"], "steps": h["steps"], "seconds": h.get("train_seconds"),
                                 "final_val": h["val"][-1], "n_train": int(cell.ax.usable_tr.sum() if arm == "A" else len(cell.s_tr))}

    rs = np.random.default_rng([seed, sk, 111]); torch.manual_seed(11_000 + seed * 10 + sk)
    pools, fb = state_pools(cell, cell.s_te[idx], cell.ax)
    res["state_cond_fallback_frac"] = fb
    S = {"A": gen(fms["A"], cond, None, steps, N, n_div, cell),
         "QCs": gen(fms["B2"], cond, q_state(pools, 2, rs, device), steps, N, n_div, cell)}
    if scale == 0.25:
        S["A_step"] = gen(fms["A_step"], cond, None, steps, N, n_div, cell)
        S["C"] = gen(fms["C"], cond, None, steps, N, n_div, cell)
        S["B1"] = gen(fms["B1"], cond, q_dist(cell.ax.Q, 1, rs, device), steps, N, n_div, cell)
        S["ACs"] = gen(fms["B1"], cond, q_state(pools, 1, rs, device), steps, N, n_div, cell)
        S["B2"] = gen(fms["B2"], cond, q_dist(cell.ax.Q, 2, rs, device), steps, N, n_div, cell)
        S["mode:per_state_fixed"] = gen(fms["B2"], cond, q_per_state(cell.ax.Q, rs, device), steps, N, n_div, cell)
        S["mode:global_median"] = gen(fms["B2"], cond, q_global(cell.ax.Q, device), steps, N, n_div, cell)
        for arm in ("A25", "A75", "A_con"):
            S[arm] = gen(fms[arm], cond, None, steps, N, n_div, cell)
    for arm, (g, st) in S.items():
        res["arms"][arm] = evaluate(g, st, c_ev, cell, axR, cc["pr_k"])
    np.savez_compressed(OUT / "samples" / f"{tag}.npz", state_idx=idx,
                        **{f"{a}_gen": S[a][0].astype(np.float32) for a in ("A", "QCs")},
                        **{f"{a}_div": S[a][1].astype(np.float32) for a in ("A", "QCs")})

    if scale == 0.25:
        res["arms"]["OutFilt"] = output_filtered(fms["C"], cond, c_ev, cell, cell.ax, steps, N, n_div, cc["reject_max_tries"], cc["pr_k"])
        curve = {}
        for T in TEMPS:
            m = res["arms"]["A"] if T == 1.0 else evaluate(*gen(fms["A"], cond, None, steps, N, n_div, cell, T), c_ev, cell, axR, cc["pr_k"])
            curve[str(T)] = {kk: m[kk] for kk in KEEP}
        res["temperature_curve_F"] = curve
        # QC-S at this cell's consistency target (the filter-variant run's 1/4 target)
        cidx, thr = con_target(cell, seed, scale, cfg, cc)
        check_con(cidx, thr, cell, DEDUP / "filter_variants" / f"{tag}.json")
        cpools, cfb = state_pools(cell, cell.s_te[cidx], cell.ax)
        g, st = gen(fms["B2"], torch.from_numpy(cell.nS(cell.s_te[cidx])).to(device), q_state(cpools, 2, rs, device), steps, N, n_div, cell)
        res["con_target"] = {"QCs": evaluate(g, st, cell.c_te[cidx], cell, cell.ax, cc["pr_k"]), "fallback": cfb}

    res["seconds"] = time.time() - t0
    out_p.write_text(json.dumps(res), encoding="utf-8")
    a, q = res["arms"]["A"], res["arms"]["QCs"]
    print(f"[done] {tag} {res['seconds'] / 60:.1f} min | eps {res['n_eps']}/{res['n_eps_total']} pairs {res['n_train_pairs']} "
          f"| rec F {a['recall']:.3f} QCs {q['recall']:.3f} | div F {a['diversity']:.4f} QCs {q['diversity']:.4f} "
          f"| lowq F {a['lowq_G']:.3f} QCs {q['lowq_G']:.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/review_campaign.yaml")
    p.add_argument("--stage", choices=["full", "common"], required=True)
    p.add_argument("--corpora", nargs="+"); p.add_argument("--seeds", type=int, nargs="+"); p.add_argument("--scales", type=float, nargs="+")
    a = p.parse_args()
    cc = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(cc["base_config"]).read_text(encoding="utf-8"))
    for d in ("full", "common", "samples", "ckpt"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    (OUT / f"config_used_{a.stage}.yaml").write_text(yaml.safe_dump({"campaign": cc, "base": cfg, "cli": vars(a), "temps": TEMPS,
                                                                     "k_state": K_STATE}), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} | stage={a.stage}", flush=True)
    for corpus in a.corpora or cc["corpora"]:
        for seed in a.seeds if a.seeds is not None else cc["seeds"]:
            if a.stage == "full":
                run_full(corpus, seed, cfg, cc, device)
            else:
                for scale in a.scales or [0.25]:
                    run_common(corpus, scale, seed, cfg, cc, device)


if __name__ == "__main__":
    main()
