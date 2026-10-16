"""Report 12 (request 12): additional analyses on the full-data cells, no training.

B1  two-axis target. Tremor (pre-fixed): within the training activity-usable chunks, 10 activity-decile bins; per bin the
    lower-25% quantile of smoothness z is the tremor threshold (sensitivity: 50%). Two-axis usable = activity >= tau and
    not tremor. Bin edges / thresholds from the training pool are applied to held-out and generated chunks (activity
    outside the training range -> nearest end bin). Target R2 = 1,500 unique held-out two-axis-usable chunks. Request
    pool = two-axis-usable chunks among the 32 nearest training states (fallback: all two-axis-usable training chunks).
    QC-S draws (activity, smoothness), AC-S draws activity from the same pool, Filtered has no request.
B2  consistency target of report 10/11a (replayed). QC-S request pool = chunks among the 32 nearest training states whose
    action-consistency >= training median (= the chunks the consistency-50% filter keeps); fallback: all such chunks.
6.1 below-tau generations of F / FU / AC-S / QC-S on the main target R (saved report-09/11a samples), split at the
    training activity p25 ("stop level") and tau = p50 ("boundary").
6.2 QC-S + post-generation check: OF rule (resample noise and q until activity >= tau, max 64 draws, states that never
    pass are dropped) on G and on the diversity set; same instrumented rule re-run for OF (Full-Unconditioned).
Outputs: outputs/campaign_dedup/rq12/cells/<cell>.json

    .venv/Scripts/python.exe scripts/review_12_lowq_axes.py
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
from run_review_campaign import act_raw, jerk_raw, knn_idx, sample  # noqa: E402
from review_dedup_reeval import K_STATE, first_unique  # noqa: E402
from review_dedup_filter_variants import N_REF, consistency, pair_ids  # noqa: E402
from review_11_rq import Cell, con_target, check_con, evaluate, gen, load_fm, q_state, state_pools  # noqa: E402

OUT = Path("outputs/campaign_dedup/rq12")
D9 = Path("outputs/campaign_dedup")
KEEP = ("recall", "precision", "diversity", "best_of_N", "inside")


class Tremor:
    """Pre-fixed tremor rule on the training pool (activity deciles of the usable chunks, per-bin smoothness quantile)."""

    def __init__(self, ax, frac):
        u = ax.usable_tr
        a, s = ax.act_tr[u], ax.q_tr[u, 1]
        self.ax, self.frac = ax, frac
        self.edges = np.quantile(a, np.arange(1, 10) / 10)
        b = np.searchsorted(self.edges, a, side="right")
        self.thr = np.array([np.quantile(s[b == k], frac) for k in range(10)])

    def classify(self, c):
        """0 = activity < tau, 1 = tremor, 2 = two-axis usable."""
        a, j = act_raw(c), jerk_raw(c)
        s = self.ax.q_of(a, j)[:, 1]
        trem = s < self.thr[np.searchsorted(self.edges, a, side="right")]
        return np.where(a < self.ax.tau, 0, np.where(trem, 1, 2))


def class_fracs(g, st, tr):
    cg = tr.classify(g); cd = tr.classify(st.reshape(-1, *st.shape[2:]))
    return {"below_tau_G": float((cg == 0).mean()), "tremor_G": float((cg == 1).mean()),
            "below_tau_div": float((cd == 0).mean()), "tremor_div": float((cd == 1).mean()),
            "tremor_among_usable_G": float((cg == 1).sum() / max((cg >= 1).sum(), 1))}


def nbr_pools(cell, states, ok_tr, q_all):
    """Pools of q from training chunks among the K_STATE nearest training states that satisfy ok_tr; fallback q_all[ok_tr]."""
    nbr = knn_idx(cell.nS(states), cell.nS(cell.s_tr), K_STATE)
    fb_pool = q_all[ok_tr]; pools, fb = [], 0
    for i in range(len(states)):
        u = nbr[i][ok_tr[nbr[i]]]
        if len(u) == 0:
            fb += 1; pools.append(fb_pool)
        else:
            pools.append(q_all[u])
    return pools, fb / len(states)


def b1(cell, fms, frac, seed, cfg, cc, device, rs, steps, N, n_div):
    ax = cell.ax; tr = Tremor(ax, frac)
    cls_tr = tr.classify(cell.c_tr); two_tr = cls_tr == 2
    first = first_unique(cell.s_te, cell.c_te)
    cls_te = tr.classify(cell.c_te)
    us_te_u = first[cls_te[first] >= 1]
    cand = first[cls_te[first] == 2]
    idx = rs.permutation(cand)[: cfg["eval_conditions"]]
    c_ev = cell.c_te[idx]; cond = torch.from_numpy(cell.nS(cell.s_te[idx])).to(device)
    pools, fb = nbr_pools(cell, cell.s_te[idx], two_tr, ax.q_tr)
    qa = ax.q_tr
    out = {"frac": frac, "n_R2_available": int(len(cand)), "n_R2": int(len(idx)), "fallback": fb,
           "edges": tr.edges.tolist(), "thr": tr.thr.tolist(),
           "heldout_tremor_among_usable_unique": float((cls_te[us_te_u] == 1).mean()),
           "train_tremor_among_usable": float((cls_tr[ax.usable_tr] == 1).mean()),
           "pool_means": {"act_pool": [float(x) for x in qa[ax.usable_tr].mean(0)], "two_axis_pool": [float(x) for x in qa[two_tr].mean(0)]},
           "R2_z": [float(x) for x in ax.q_of(act_raw(c_ev), jerk_raw(c_ev)).mean(0)], "arms": {}}
    S = {"A": gen(fms["A"], cond, None, steps, N, n_div, cell),
         "QCs": gen(fms["B2"], cond, q_state(pools, 2, rs, device), steps, N, n_div, cell),
         "ACs": gen(fms["B1"], cond, q_state(pools, 1, rs, device), steps, N, n_div, cell)}
    for arm, (g, st) in S.items():
        m = evaluate(g, st, c_ev, cell, ax, cc["pr_k"])
        out["arms"][arm] = {**{k: m[k] for k in KEEP + ("zact_G", "zsmooth_G")}, **class_fracs(g, st, tr)}
    return out


def b2(cell, fms, seed, cfg, cc, device, rs, steps, N, n_div, tag):
    idx, thr = con_target(cell, seed, 1.0, cfg, cc)
    check_con(idx, thr, cell, D9 / "filter_variants" / f"{tag}.json")
    r_ = np.random.default_rng([seed, 100, 34])                       # same reference draw as the filter-variant run
    _, tr_id = pair_ids(cell.s_tr, cell.c_tr)
    ref = r_.permutation(len(cell.s_tr))[: min(N_REF, len(cell.s_tr))]
    con_tr = consistency(cell.nS(cell.s_tr), cell.nA(cell.c_tr).reshape(len(cell.c_tr), -1), tr_id,
                         cell.nS(cell.s_tr[ref]), cell.nA(cell.c_tr[ref]).reshape(len(ref), -1), tr_id[ref])
    keep = con_tr >= np.median(con_tr)
    fv = json.loads((D9 / "filter_variants" / f"{tag}.json").read_text(encoding="utf-8"))
    assert abs(fv["kept"]["A_con"] - keep.mean()) < 1e-9, "consistency filter mask replay mismatch"
    pools, fb = nbr_pools(cell, cell.s_te[idx], keep, cell.ax.q_tr)
    c_ev = cell.c_te[idx]; cond = torch.from_numpy(cell.nS(cell.s_te[idx])).to(device)
    g, st = gen(fms["B2"], cond, q_state(pools, 2, rs, device), steps, N, n_div, cell)
    m = evaluate(g, st, c_ev, cell, cell.ax, cc["pr_k"])
    act_pool_fb = json.loads((D9 / "rq11" / "full" / f"{tag}.json").read_text(encoding="utf-8"))["con_target"]["fallback"]
    return {"QCs_conpool": {k: m[k] for k in KEEP}, "fallback": fb, "fallback_actpool_11a": act_pool_fb,
            "con_pool_act_usable_frac": float(cell.ax.usable_tr[keep].mean())}


def below_tau(cell, tag):
    ax = cell.ax; p25 = float(np.quantile(ax.act_tr, 0.25))
    S = {a: np.load(D9 / "samples" / tag / f"{a}.npz")["gen"] for a in ("A", "C", "QCs")}
    S["ACs"] = np.load(D9 / "rq11" / "samples" / f"{tag}_ACs.npz")["gen"]
    out = {"p25": p25, "tau": ax.tau, "arms": {}}
    for a, g in S.items():
        x = act_raw(g)
        out["arms"][a] = {"stop_G": float((x < p25).mean()), "boundary_G": float(((x >= p25) & (x < ax.tau)).mean())}
    x = act_raw(cell.c_te)
    out["heldout"] = {"stop": float((x < p25).mean()), "boundary": float(((x >= p25) & (x < ax.tau)).mean())}
    return out


def checked(fm, cond, c_ev, cell, qpick, steps, N, n_div, tries, pr_k):
    """OF rule with instrumentation. qpick(rows) -> q tensor for those state rows (None for unconditioned)."""
    ax, k, D = cell.ax, cell.k, cell.D
    m_ = len(cond)
    acc = np.full((m_, k, D), np.nan, np.float32); have = np.zeros(m_, bool); draws = np.zeros(m_, int)
    for _ in range(tries):
        todo = np.where(~have)[0]
        if len(todo) == 0: break
        x = cell.dA(sample(fm, cond[todo], qpick(todo), steps).cpu().numpy())
        ok = act_raw(x) >= ax.tau; draws[todo] += 1
        acc[todo[ok]] = x[ok]; have[todo[ok]] = True
    div_acc = [[] for _ in range(n_div)]; ddraws = 0
    for _ in range(tries):
        todo = np.array([i for i in range(n_div) if len(div_acc[i]) < N])
        if len(todo) == 0: break
        x = cell.dA(sample(fm, cond[todo], qpick(todo), steps).cpu().numpy())
        ok = act_raw(x) >= ax.tau; ddraws += len(todo)
        for j, i in enumerate(todo):
            if ok[j]: div_acc[i].append(x[j])
    full = [i for i in range(n_div) if len(div_acc[i]) >= N]
    sel = np.where(have)[0]
    stO = np.stack([np.stack(div_acc[i][:N], 0) for i in full], 1)
    pr, rc = precision_recall(c_ev[sel].reshape(len(sel), -1), acc[sel].reshape(len(sel), -1), k=pr_k)
    n_div_acc = sum(min(len(a), N) for a in div_acc)
    return {"recall": rc, "precision": pr, "diversity": float(stO.std(0).mean()),
            "best_of_N": float(((stO - c_ev[full][None]) ** 2).mean(axis=(2, 3)).min(0).mean()),
            "inside": float(((acc[sel] >= cell.amin) & (acc[sel] <= cell.amax)).mean()),
            "G_frac_resampled": float((draws > 1).mean()), "G_draws_per_accepted": float(draws[have].mean()),
            "G_states_dropped": int((~have).sum()), "div_draws_per_accepted": float(ddraws / max(n_div_acc, 1)),
            "div_states_dropped": int(n_div - len(full))}


def run_cell(corpus, seed, cfg, cc, device):
    tag = f"{corpus}_f1.00_s{seed}"; out_p = OUT / "cells" / f"{tag}.json"
    if out_p.exists():
        print(f"[skip] {tag}"); return
    t0 = time.time()
    steps, N = cfg["sample_steps"], cc["n_best"]; n_div = min(cfg["diversity_conditions"], cfg["eval_conditions"])
    cell = Cell(corpus, 1.0, seed, cfg)
    ck = Path(cc["out_dir"]) / "ckpt" / tag
    fms = {"A": load_fm(ck / "A.pt", cfg, cell, 0, device), "C": load_fm(ck / "C.pt", cfg, cell, 0, device),
           "B1": load_fm(ck / "B1.pt", cfg, cell, 1, device), "B2": load_fm(ck / "B2.pt", cfg, cell, 2, device)}
    rs = np.random.default_rng([seed, 100, 120]); torch.manual_seed(12_000 + seed * 10)
    res = {"corpus": corpus, "seed": seed}
    res["B1_25"] = b1(cell, fms, 0.25, seed, cfg, cc, device, rs, steps, N, n_div)
    res["B1_50"] = b1(cell, fms, 0.50, seed, cfg, cc, device, rs, steps, N, n_div)
    res["B2"] = b2(cell, fms, seed, cfg, cc, device, rs, steps, N, n_div, tag)
    res["below_tau"] = below_tau(cell, tag)

    # 6.2 post-generation check on the main target R (same states as report 09)
    idx = np.load(D9 / "samples" / tag / "A.npz")["state_idx"]
    c_ev = cell.c_te[idx]; cond = torch.from_numpy(cell.nS(cell.s_te[idx])).to(device)
    pools, _ = state_pools(cell, cell.s_te[idx], cell.ax)

    def q_rows(rows):
        return torch.from_numpy(np.stack([pools[i][rs.integers(0, len(pools[i]))] for i in rows]).astype(np.float32)).to(device)

    res["check"] = {"QCs_checked": checked(fms["B2"], cond, c_ev, cell, q_rows, steps, N, n_div, cc["reject_max_tries"], cc["pr_k"]),
                    "OF_rerun": checked(fms["C"], cond, c_ev, cell, lambda rows: None, steps, N, n_div, cc["reject_max_tries"], cc["pr_k"])}
    res["seconds"] = time.time() - t0
    out_p.write_text(json.dumps(res), encoding="utf-8")
    a, b_ = res["B1_25"]["arms"], res["check"]
    print(f"[done] {tag} {res['seconds'] / 60:.1f} min | R2 {res['B1_25']['n_R2']} fb {res['B1_25']['fallback']:.3f} "
          f"| R2 rec QCs {a['QCs']['recall']:.3f} ACs {a['ACs']['recall']:.3f} trem QCs {a['QCs']['tremor_G']:.3f} ACs {a['ACs']['tremor_G']:.3f} "
          f"| B2 fb {res['B2']['fallback']:.3f} rec {res['B2']['QCs_conpool']['recall']:.3f} "
          f"| chk resamp {b_['QCs_checked']['G_frac_resampled']:.3f} OF {b_['OF_rerun']['G_frac_resampled']:.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/review_campaign.yaml")
    p.add_argument("--corpora", nargs="+"); p.add_argument("--seeds", type=int, nargs="+")
    a = p.parse_args()
    cc = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(cc["base_config"]).read_text(encoding="utf-8"))
    (OUT / "cells").mkdir(parents=True, exist_ok=True)
    (OUT / "config_used.yaml").write_text(yaml.safe_dump({"campaign": cc, "base": cfg, "cli": vars(a), "k_state": K_STATE,
                                                          "tremor_fracs": [0.25, 0.5]}), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}", flush=True)
    for corpus in a.corpora or cc["corpora"]:
        for seed in a.seeds if a.seeds is not None else cc["seeds"]:
            run_cell(corpus, seed, cfg, cc, device)


if __name__ == "__main__":
    main()
