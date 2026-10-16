"""Report 14 (request 14): re-evaluate QC / AC with neighbour count K = 64 (full data; proportional K at 1/2, 1/4, 1/8:
32, 16, 8). No retraining. Only QC and AC are regenerated; every comparison arm is read from the earlier reports.
ALL outputs go to outputs/campaign_dedup/rq14/ (nothing earlier is overwritten).

full   (per full-data cell, target R and the report-12 targets)
  QC K64 at T = 1   : the SAME RNG stream as the report-13 confirmation run (asserted equal), samples saved
  QC K64 T curve    : T in TEMPS (T = 1 reused)
  AC K64            : Activity-Conditioned with q from the same K64 pools
  R2 (report 12 B1) : replayed target; QC / AC with two-axis-usable K64 pools; tremor classes
  consistency target: QC with the activity pool and with the consistency pool, both K64
  post-check        : QC K64 + OF rule (report 12 6.2 code)
scales (1/2, 1/4, 1/8 on the common target R): QC with proportional K

    .venv/Scripts/python.exe scripts/review_14_k64.py --stage full
    .venv/Scripts/python.exe scripts/review_14_k64.py --stage scales
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

from run_review_campaign import act_raw, jerk_raw  # noqa: E402
from review_dedup_reeval import TEMPS, first_unique  # noqa: E402
from review_dedup_filter_variants import N_REF, consistency, pair_ids  # noqa: E402
from review_11_rq import SKEY, TAG, Cell, check_con, common_target, con_target, evaluate, gen, load_fm  # noqa: E402
from review_12_lowq_axes import Tremor, checked, class_fracs  # noqa: E402
from review_13_request import knn_dist, make_pools, q_fn  # noqa: E402

OUT = Path("outputs/campaign_dedup/rq14")
D9 = Path("outputs/campaign_dedup")
KFULL = 64
KSCALE = {1.0: 64, 0.5: 32, 0.25: 16, 0.125: 8}
KEEP = ("recall", "precision", "diversity", "best_of_N", "inside", "lowq_G")


def mask_pools(cell, states, ok_tr, K):
    """q pools from training chunks satisfying ok_tr among the K nearest training states; fallback = all ok_tr chunks."""
    nbr, _ = knn_dist(cell.nS(states), cell.nS(cell.s_tr), K)
    fbp = cell.ax.q_tr[ok_tr]; pools, fb = [], 0
    for i in range(len(states)):
        u = nbr[i][ok_tr[nbr[i]]]
        if len(u) == 0:
            fb += 1; pools.append(fbp)
        else:
            pools.append(cell.ax.q_tr[u])
    return pools, fb / len(states)


def q_int(pools, qd, rs, device):
    def fn(shape):
        n_states = shape[-1]; lead = shape[:-1]
        out = np.empty((*shape, qd), np.float32)
        for i in range(n_states):
            pick = pools[i][rs.integers(0, len(pools[i]), lead or (1,))][..., :qd]
            if lead: out[:, i] = pick
            else: out[i] = pick[0]
        return torch.from_numpy(out).to(device)
    return fn


def stop_boundary(g, ax):
    p25 = float(np.quantile(ax.act_tr, 0.25)); x = act_raw(g)
    return {"stop_G": float((x < p25).mean()), "boundary_G": float(((x >= p25) & (x < ax.tau)).mean())}


def run_full(corpus, seed, cfg, cc, device):
    tag = f"{corpus}_f1.00_s{seed}"; out_p = OUT / "full" / f"{tag}.json"
    if out_p.exists():
        print(f"[skip] {tag}"); return
    t0 = time.time(); steps, N = cfg["sample_steps"], cc["n_best"]
    cell = Cell(corpus, 1.0, seed, cfg); ax = cell.ax
    ck = Path(cc["out_dir"]) / "ckpt" / tag
    fQ, fAC = load_fm(ck / "B2.pt", cfg, cell, 2, device), load_fm(ck / "B1.pt", cfg, cell, 1, device)
    idx = np.load(D9 / "samples" / tag / "A.npz")["state_idx"]
    c_ev = cell.c_te[idx]; cond = torch.from_numpy(cell.nS(cell.s_te[idx])).to(device)
    n_div = min(cfg["diversity_conditions"], len(idx))
    res = {"corpus": corpus, "seed": seed, "K": KFULL, "arms": {}}

    # ---- QC K64, T = 1: identical stream to report 13 'selected'
    rs = np.random.default_rng([seed, 13, 2]); torch.manual_seed(13_500 + seed)
    pools, fb = make_pools(cell, cell.s_te[idx], KFULL, 0)
    g, st = gen(fQ, cond, q_fn(pools, rs, device), steps, N, n_div, cell)
    m = evaluate(g, st, c_ev, cell, ax, cc["pr_k"])
    r13 = json.loads((D9 / "rq13" / "confirm" / f"{tag}_K64_w0.json").read_text(encoding="utf-8"))["arms"]["selected"]
    res["replay_dev_vs_report13"] = max(abs(m[k] - r13[k]) for k in ("recall", "precision", "diversity", "best_of_N"))
    res["arms"]["QC"] = {**{k: m[k] for k in KEEP}, **stop_boundary(g, ax), "fallback": fb}
    np.savez_compressed(OUT / "samples" / f"{tag}_QC.npz", gen=g.astype(np.float32), div=st.astype(np.float32), state_idx=idx)

    # ---- new stream for everything else
    rs = np.random.default_rng([seed, 14, 1]); torch.manual_seed(14_000 + seed)
    curve = {"1.0": {k: res["arms"]["QC"][k] for k in ("recall", "precision", "diversity", "best_of_N", "inside")}}
    qfn = q_fn(pools, rs, device)
    for T in TEMPS:
        if T == 1.0: continue
        m = evaluate(*gen(fQ, cond, qfn, steps, N, n_div, cell, T), c_ev, cell, ax, cc["pr_k"])
        curve[str(T)] = {k: m[k] for k in ("recall", "precision", "diversity", "best_of_N", "inside")}
    res["temperature_curve_QC"] = curve

    upools = [p[0] for p in pools]                      # uniform pools (K64) as plain arrays for qd = 1
    g, st = gen(fAC, cond, q_int(upools, 1, rs, device), steps, N, n_div, cell)
    m = evaluate(g, st, c_ev, cell, ax, cc["pr_k"])
    res["arms"]["AC"] = {**{k: m[k] for k in KEEP}, **stop_boundary(g, ax)}
    np.savez_compressed(OUT / "samples" / f"{tag}_AC.npz", gen=g.astype(np.float32), div=st.astype(np.float32), state_idx=idx)

    # ---- R2 (report 12 B1, 25%): replay of the target draw
    tr = Tremor(ax, 0.25); cls_tr = tr.classify(cell.c_tr)
    first = first_unique(cell.s_te, cell.c_te); cls_te = tr.classify(cell.c_te)
    cand = first[cls_te[first] == 2]
    idx2 = np.random.default_rng([seed, 100, 120]).permutation(cand)[: cfg["eval_conditions"]]
    r12 = json.loads((D9 / "rq12" / "cells" / f"{tag}.json").read_text(encoding="utf-8"))["B1_25"]
    c2 = cell.c_te[idx2]
    assert len(idx2) == r12["n_R2"] and np.allclose(ax.q_of(act_raw(c2), jerk_raw(c2)).mean(0), r12["R2_z"]), "R2 replay mismatch"
    cond2 = torch.from_numpy(cell.nS(cell.s_te[idx2])).to(device)
    p2, fb2 = mask_pools(cell, cell.s_te[idx2], cls_tr == 2, KFULL)
    res["R2"] = {"fallback": fb2, "arms": {}}
    for arm, fm, qd in (("QC", fQ, 2), ("AC", fAC, 1)):
        g, st = gen(fm, cond2, q_int(p2, qd, rs, device), steps, N, n_div, cell)
        m = evaluate(g, st, c2, cell, ax, cc["pr_k"])
        res["R2"]["arms"][arm] = {**{k: m[k] for k in KEEP}, "zact_G": m["zact_G"], "zsmooth_G": m["zsmooth_G"], **class_fracs(g, st, tr)}

    # ---- consistency target: activity pool and consistency pool, K64
    cidx, thr = con_target(cell, seed, 1.0, cfg, cc)
    check_con(cidx, thr, cell, D9 / "filter_variants" / f"{tag}.json")
    r_ = np.random.default_rng([seed, 100, 34]); _, tr_id = pair_ids(cell.s_tr, cell.c_tr)
    ref = r_.permutation(len(cell.s_tr))[: min(N_REF, len(cell.s_tr))]
    con_tr = consistency(cell.nS(cell.s_tr), cell.nA(cell.c_tr).reshape(len(cell.c_tr), -1), tr_id,
                         cell.nS(cell.s_tr[ref]), cell.nA(cell.c_tr[ref]).reshape(len(ref), -1), tr_id[ref])
    keep = con_tr >= np.median(con_tr)
    fv = json.loads((D9 / "filter_variants" / f"{tag}.json").read_text(encoding="utf-8"))
    assert abs(fv["kept"]["A_con"] - keep.mean()) < 1e-9
    cc_ev = cell.c_te[cidx]; ccond = torch.from_numpy(cell.nS(cell.s_te[cidx])).to(device)
    res["con"] = {}
    for name, ok in (("QC_actpool", ax.usable_tr), ("QC_conpool", keep)):
        pc, fbc = mask_pools(cell, cell.s_te[cidx], ok, KFULL)
        m = evaluate(*gen(fQ, ccond, q_int(pc, 2, rs, device), steps, N, n_div, cell), cc_ev, cell, ax, cc["pr_k"])
        res["con"][name] = {**{k: m[k] for k in KEEP}, "fallback": fbc}

    # ---- post-generation check (report 12 6.2), K64 pools
    def q_rows(rows):
        return torch.from_numpy(np.stack([upools[i][rs.integers(0, len(upools[i]))] for i in rows]).astype(np.float32)).to(device)
    res["QC_checked"] = checked(fQ, cond, c_ev, cell, q_rows, steps, N, n_div, cc["reject_max_tries"], cc["pr_k"])

    res["seconds"] = time.time() - t0
    out_p.write_text(json.dumps(res), encoding="utf-8")
    q = res["arms"]["QC"]
    print(f"[full] {tag} {res['seconds'] / 60:.1f} min | replay dev {res['replay_dev_vs_report13']:.1e} | QC rec {q['recall']:.3f} "
          f"prec {q['precision']:.3f} stop {q['stop_G']:.4f} | AC rec {res['arms']['AC']['recall']:.3f} | R2 trem QC "
          f"{res['R2']['arms']['QC']['tremor_G']:.3f} AC {res['R2']['arms']['AC']['tremor_G']:.3f}", flush=True)


def run_scale(corpus, scale, seed, cfg, cc, device):
    tag = f"{corpus}_f{TAG[scale]}_s{seed}"; out_p = OUT / "scales" / f"{tag}.json"
    if out_p.exists():
        print(f"[skip] {tag}"); return
    t0 = time.time(); steps, N = cfg["sample_steps"], cc["n_best"]; K = KSCALE[scale]
    cell = Cell(corpus, scale, seed, cfg)
    idx, axR = common_target(corpus, seed, cfg)
    c_ev = cell.c_te[idx]; cond = torch.from_numpy(cell.nS(cell.s_te[idx])).to(device)
    n_div = min(cfg["diversity_conditions"], len(idx))
    ck = (Path(cc["out_dir"]) / "ckpt" / tag) if scale == 0.25 else (D9 / "rq11" / "ckpt" / tag)
    fQ = load_fm(ck / "B2.pt", cfg, cell, 2, device)
    rs = np.random.default_rng([seed, SKEY[scale], 14]); torch.manual_seed(14_100 + seed * 10 + SKEY[scale])
    pools, fb = make_pools(cell, cell.s_te[idx], K, 0)
    m = evaluate(*gen(fQ, cond, q_fn(pools, rs, device), steps, N, n_div, cell), c_ev, cell, axR, cc["pr_k"])
    res = {"corpus": corpus, "scale": scale, "seed": seed, "K": K, "fallback": fb, "QC": {k: m[k] for k in KEEP},
           "seconds": time.time() - t0}
    out_p.write_text(json.dumps(res), encoding="utf-8")
    print(f"[scale] {tag} K={K} fb {fb:.3f} | QC rec {m['recall']:.3f} div {m['diversity']:.4f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/review_campaign.yaml")
    p.add_argument("--stage", choices=["full", "scales"], required=True)
    p.add_argument("--corpora", nargs="+"); p.add_argument("--seeds", type=int, nargs="+")
    a = p.parse_args()
    cc = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(cc["base_config"]).read_text(encoding="utf-8"))
    for d in ("full", "scales", "samples"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for corpus in a.corpora or cc["corpora"]:
        for seed in a.seeds if a.seeds is not None else cc["seeds"]:
            if a.stage == "full":
                run_full(corpus, seed, cfg, cc, device)
            else:
                for sc in (0.5, 0.25, 0.125):
                    run_scale(corpus, sc, seed, cfg, cc, device)


if __name__ == "__main__":
    main()
