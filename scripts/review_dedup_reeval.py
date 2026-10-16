"""Evaluation-only re-run of the review campaign with DE-DUPLICATED evaluation targets (report 08).

Why: windows are 60 frames with stride 30, and pairs are taken at every start of every window, so 42%
of start positions appear twice. 8-10% of the 1,500 evaluation targets were exact duplicates, which
shrinks the k-NN radii of near-deterministic generators and biases recall toward diverse ones.

What: reload the 280 campaign checkpoints (no retraining), rebuild each cell's data exactly (asserted
against the checkpoint normalisation), drop duplicate (state, chunk) pairs from the held-out set, draw
1,500 usable targets per axis definition, and compute EVERY RQ3 quantity in ONE pass so that the
tables are arithmetically consistent:
  main arms at T=1 (A, A_step, C, B1, B2=QC-M, A_z, B2_z) + QC-S at T=1,
  Output-Filtered (rejection on C, max 64 tries), q-mode ablation for B2,
  temperature curves F / QC-M / QC-S at T in TEMPS, with the T=1 points REUSING the main samples.
Training pools are unchanged (duplicates stay in training; identical for all arms).
Outputs: outputs/campaign_dedup/{results,tradeoff,samples}/<cell>.*  (resumable)

    .venv/Scripts/python.exe scripts/review_dedup_reeval.py
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
from run_review_campaign import QDIM, Axes, act_raw, knn_idx, main_metrics, make_fm, sample  # noqa: E402

TEMPS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
ARMS = ["A", "A_step", "C", "B1", "B2", "A_z", "B2_z"]
K_STATE = 32


def first_unique(s, c):
    x = np.concatenate([s, c.reshape(len(c), -1)], 1)
    _, first = np.unique(x, axis=0, return_index=True)
    return np.sort(first)


def run_cell(corpus, scale, seed, cfg, cc, device, out):
    tag = f"{corpus}_f{scale:.2f}_s{seed}"
    if (out / "results" / f"{tag}.json").exists():
        print(f"[skip] {tag}"); return
    t0 = time.time()
    k, steps, N = cfg["chunk_size"], cfg["sample_steps"], cc["n_best"]
    W = np.load(f"data_processed/windows/{corpus}_stateaction_w60_s30.npy").astype(np.float32)
    we = np.load(f"data_processed/stats/{corpus}_stateaction_w60_s30_stats.npz", allow_pickle=True)["window_episode"]
    D = W.shape[2] // 2
    rng = np.random.default_rng([seed, int(round(scale * 100))])          # campaign's first draw = episode subset
    tr_w, te_w, _, _ = episode_split(we, cfg["val_frac"], seed)
    tr_eps = np.unique(we[tr_w])
    keep = rng.choice(tr_eps, size=max(1, int(round(len(tr_eps) * scale))), replace=False)
    tr_w = tr_w[np.isin(we[tr_w], keep)]
    s_tr, c_tr = build_pairs_from_windows(W[tr_w], k, state_dim=D)
    s_te, c_te = build_pairs_from_windows(W[te_w], k, state_dim=D)
    s_mean, s_std = norm_stats(s_tr); a_mean, a_std = norm_stats(c_tr)
    nS = lambda x: ((x - s_mean) / s_std).astype(np.float32)
    nA = lambda x: ((x - a_mean) / a_std).astype(np.float32)
    dA = lambda x: x * a_std + a_mean
    axes = {"raw": Axes(c_tr, lambda c: c), "z": Axes(c_tr, nA)}
    amin, amax = c_tr.reshape(-1, D).min(0), c_tr.reshape(-1, D).max(0)

    ck_dir = Path(cc["out_dir"]) / "ckpt" / tag
    fms = {}
    for arm in ARMS:
        ck = torch.load(ck_dir / f"{arm}.pt", map_location=device, weights_only=False)
        assert np.allclose(ck["a_mean"], a_mean) and np.allclose(ck["s_std"], s_std), f"{tag} {arm}: data rebuild mismatch"
        fm = make_fm(cfg, D, QDIM[arm]).to(device); fm.load_state_dict(ck["state"]); fm.eval(); fms[arm] = fm

    # ---- de-duplicated evaluation targets (per axis definition)
    rs = np.random.default_rng([seed, int(round(scale * 100)), 23])
    torch.manual_seed(1_000 + seed * 10 + int(round(scale * 4)))
    uniq = first_unique(s_te, c_te)
    is_u = np.zeros(len(c_te), bool); is_u[uniq] = True
    tgt = {}
    for n, ax in axes.items():
        us = ax.usable(c_te)
        idx = rs.permutation(np.where(us & is_u)[0])[: cfg["eval_conditions"]]
        tgt[n] = dict(idx=idx, c_ev=c_te[idx], prof=smoothness_profile(c_te[idx]),
                      cond=torch.from_numpy(nS(s_te[idx])).to(device),
                      heldout_usable=int(us.sum()), heldout_usable_unique=int((us & is_u).sum()))
    n_div = min(cfg["diversity_conditions"], cfg["eval_conditions"])

    def qdraw(ax, qd):
        def fn(shape):
            return torch.from_numpy(ax.Q[rs.integers(0, len(ax.Q), shape)][..., :qd].astype(np.float32)).to(device)
        return fn

    def gen(arm, n, qfn, T=1.0):
        t = tgt[n]; cond = t["cond"]
        g = dA(sample(fms[arm], cond, None if qfn is None else qfn((len(cond),)), steps, T).cpu().numpy())
        qs = None if qfn is None else qfn((N, n_div))
        st = np.stack([dA(sample(fms[arm], cond[:n_div], None if qs is None else qs[i], steps, T).cpu().numpy())
                       for i in range(N)], 0)
        return g, st

    def metrics(g, st, n):
        t = tgt[n]
        return main_metrics(g, t["c_ev"], st, amin, amax, t["prof"], cc["pr_k"])

    ax = axes["raw"]; idx = tgt["raw"]["idx"]
    # state-conditional q pools (raw axis, B2)
    nbr = knn_idx(nS(s_te[idx]), nS(s_tr), K_STATE)
    pools, fallback = [], 0
    for i in range(len(idx)):
        u = nbr[i][ax.usable_tr[nbr[i]]]
        if len(u) == 0:
            fallback += 1; pools.append(ax.Q)
        else:
            pools.append(ax.q_tr[u])

    def q_state(shape):
        n_states = shape[-1]; lead = shape[:-1]
        out_q = np.empty((*shape, 2), np.float32)
        for i in range(n_states):
            pick = pools[i][rs.integers(0, len(pools[i]), lead or (1,))]
            if lead: out_q[:, i] = pick
            else: out_q[i] = pick[0]
        return torch.from_numpy(out_q).to(device)

    def q_per_state(shape):
        base = ax.Q[rs.integers(0, len(ax.Q), shape[-1])]
        return torch.from_numpy(np.broadcast_to(base, (*shape, 2)).astype(np.float32).copy()).to(device)

    med = np.median(ax.Q, axis=0).astype(np.float32)

    def q_global(shape):
        return torch.from_numpy(np.broadcast_to(med, (*shape, 2)).astype(np.float32).copy()).to(device)

    res = {"corpus": corpus, "scale": scale, "seed": seed, "D": D, "dedup": True,
           "n_eval": {n: int(len(t["idx"])) for n, t in tgt.items()},
           "heldout_usable": {n: t["heldout_usable"] for n, t in tgt.items()},
           "heldout_usable_unique": {n: t["heldout_usable_unique"] for n, t in tgt.items()},
           "arms": {}, "state_cond_fallback_frac": fallback / len(idx)}
    samp = out / "samples" / tag; samp.mkdir(parents=True, exist_ok=True)

    # ---- main arms at T = 1 (+ QC-S)
    main_s = {}
    for arm in ARMS:
        n = "z" if arm.endswith("_z") else "raw"
        qfn = qdraw(axes[n], QDIM[arm]) if QDIM[arm] else None
        g, st = gen(arm, n, qfn)
        res["arms"][arm] = {"main": metrics(g, st, n)}
        main_s[arm] = (g, st)
    g, st = gen("B2", "raw", q_state)
    res["arms"]["QCs"] = {"main": metrics(g, st, "raw")}
    main_s["QCs"] = (g, st)
    for arm in ("A", "B2", "QCs", "C"):
        np.savez_compressed(samp / f"{arm}.npz", gen=main_s[arm][0].astype(np.float32),
                            div=main_s[arm][1].astype(np.float32), state_idx=idx)

    # ---- Output-Filtered (rejection on C, raw axis)
    cond = tgt["raw"]["cond"]; c_ev = tgt["raw"]["c_ev"]; tries = cc["reject_max_tries"]
    acc = np.full((len(idx), k, D), np.nan, np.float32); have = np.zeros(len(idx), bool); n_try = n_acc = 0
    for _ in range(tries):
        todo = np.where(~have)[0]
        if len(todo) == 0: break
        x = dA(sample(fms["C"], cond[todo], None, steps).cpu().numpy())
        ok = act_raw(x) >= ax.tau; n_try += len(todo); n_acc += int(ok.sum())
        acc[todo[ok]] = x[ok]; have[todo[ok]] = True
    div_acc = [[] for _ in range(n_div)]
    for _ in range(tries):
        todo = np.array([i for i in range(n_div) if len(div_acc[i]) < N])
        if len(todo) == 0: break
        x = dA(sample(fms["C"], cond[todo], None, steps).cpu().numpy())
        ok = act_raw(x) >= ax.tau
        for j, i in enumerate(todo):
            if ok[j]: div_acc[i].append(x[j])
    full = [i for i in range(n_div) if len(div_acc[i]) >= N]
    sel = np.where(have)[0]
    stO = np.stack([np.stack(div_acc[i][:N], 0) for i in full], 1)
    pr, rc = precision_recall(c_ev[sel].reshape(len(sel), -1), acc[sel].reshape(len(sel), -1), k=cc["pr_k"])
    mO = {"recall": rc, "precision": pr, "diversity": float(stO.std(0).mean()),
          "best_of_N": float(((stO - c_ev[full][None]) ** 2).mean(axis=(2, 3)).min(0).mean()),
          "inside": float(((acc[sel] >= amin) & (acc[sel] <= amax)).mean()),
          "accept_rate_pr": n_acc / max(n_try, 1), "frac_states_no_accept": float(1 - have.mean()),
          "frac_div_states_short": float(1 - len(full) / n_div)}
    res["arms"]["OutFilt"] = {"main": mO}

    # ---- q-mode ablation (B2); default and state_conditional reuse the main samples
    res["q_modes"] = {"B2": {"default": res["arms"]["B2"]["main"], "state_conditional": res["arms"]["QCs"]["main"]}}
    for mode, fn in (("per_state_fixed", q_per_state), ("global_median", q_global)):
        g, st = gen("B2", "raw", fn)
        res["q_modes"]["B2"][mode] = metrics(g, st, "raw")

    # ---- temperature curves; T = 1 reuses the main samples
    curves = {}
    for cv, arm, qfn in (("F", "A", None), ("QCm", "B2", qdraw(ax, 2)), ("QCs", "B2", q_state)):
        pts = {}
        for T in TEMPS:
            if T == 1.0:
                m = res["arms"]["QCs" if cv == "QCs" else arm]["main"]
            else:
                g, st = gen(arm, "raw", qfn, T); m = metrics(g, st, "raw")
            pts[str(T)] = {kk: m[kk] for kk in ("recall", "precision", "diversity", "best_of_N", "inside")}
        curves[cv] = pts
    res["temperature_sweep_A"] = curves["F"]
    res["seconds"] = time.time() - t0
    (out / "results").mkdir(parents=True, exist_ok=True); (out / "tradeoff").mkdir(parents=True, exist_ok=True)
    (out / "tradeoff" / f"{tag}.json").write_text(json.dumps(
        {"corpus": corpus, "scale": scale, "seed": seed, "temps": TEMPS, "n_eval": int(len(idx)),
         "state_cond_fallback_frac": res["state_cond_fallback_frac"], "curves": curves, "seconds": res["seconds"]}), encoding="utf-8")
    (out / "results" / f"{tag}.json").write_text(json.dumps(res), encoding="utf-8")
    a, b, q = res["arms"]["A"]["main"], res["arms"]["B2"]["main"], res["arms"]["QCs"]["main"]
    print(f"[done] {tag} {res['seconds'] / 60:.1f} min | uniq usable {tgt['raw']['heldout_usable_unique']}/{tgt['raw']['heldout_usable']} "
          f"| rec F {a['recall']:.3f} QCm {b['recall']:.3f} QCs {q['recall']:.3f} | prec F {a['precision']:.3f} QCs {q['precision']:.3f}",
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/review_campaign.yaml")
    p.add_argument("--out", default="outputs/campaign_dedup")
    p.add_argument("--corpora", nargs="+"); p.add_argument("--seeds", type=int, nargs="+"); p.add_argument("--scales", type=float, nargs="+")
    a = p.parse_args()
    cc = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(cc["base_config"]).read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(yaml.safe_dump({"campaign": cc, "base": cfg, "cli": vars(a), "temps": TEMPS}), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} | out={out}", flush=True)
    for corpus in a.corpora or cc["corpora"]:
        for seed in a.seeds if a.seeds is not None else cc["seeds"]:
            for scale in a.scales or cc["scales"]:
                run_cell(corpus, scale, seed, cfg, cc, device, out)


if __name__ == "__main__":
    main()
