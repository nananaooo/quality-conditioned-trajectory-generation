"""Re-run of chunk 34v2 (filter strength / filter criterion, activity vs consistency target) on the
campaign cells with DE-DUPLICATED evaluation targets (report 10).

Arms: A25 (activity, keep top 75%), A50 (= campaign Filtered, reused), A75 (keep top 25%),
      A_con (action-consistency, keep top 50%), B2 (= campaign Quality-Conditioned, reused; q ~ activity-usable pool).
A25 / A75 / A_con are trained here with the campaign's data rebuild, split, episode subset, seed and training loop
(run_review_campaign.train), and saved under outputs/campaign_dedup/ckpt_fv/<cell>/.

Targets (held-out, duplicates removed, 1,500 each):
  act_usable : the SAME states as the de-duplicated main re-evaluation (samples/<cell>/A.npz state_idx); A50 and B2
               reuse the saved T=1 samples there, so their act-target values equal Table 5 exactly.
  con_usable : unique held-out chunks with consistency >= median consistency of the unique held-out chunks.
Consistency (Eq. 4.3): -mean L2 distance between the z-scored action chunk and those of the 10 nearest reference
chunks in normalised state space; reference = 6,000 random training chunks. FIX vs chunk 34: a query never uses an
exactly identical (state, chunk) pair as its own neighbour (the chunk itself, or its copy from the overlapping window),
which inflated the score of training chunks that happened to be in the reference pool.
    .venv/Scripts/python.exe scripts/review_dedup_filter_variants.py
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

from src.train.train_lstm_vae import episode_split  # noqa: E402
from run_conditional_flow import norm_stats  # noqa: E402
from run_chunk6_external import build_pairs_from_windows  # noqa: E402
from run_chunk31_smoothness_fidelity import smoothness_profile  # noqa: E402
from run_review_campaign import Axes, main_metrics, make_fm, sample, train  # noqa: E402

KNN, N_REF = 10, 6000


def pair_ids(s, c):
    x = np.concatenate([s, c.reshape(len(c), -1)], 1)
    _, first, inv = np.unique(x, axis=0, return_index=True, return_inverse=True)
    return first, inv.reshape(-1)


def consistency(qS, qA, q_id, rS, rA, r_id, batch=2000):
    """qS/rS normalised states, qA/rA z-scored flattened actions; ids: identical pair -> excluded as neighbour."""
    out = np.empty(len(qS), np.float32)
    for i in range(0, len(qS), batch):
        d = ((qS[i:i + batch, None, :] - rS[None]) ** 2).sum(-1)
        d[q_id[i:i + batch, None] == r_id[None, :]] = np.inf
        nbr = np.argpartition(d, KNN, axis=1)[:, :KNN]
        out[i:i + batch] = -np.sqrt(((qA[i:i + batch, None, :] - rA[nbr]) ** 2).sum(-1)).mean(1)
    return out


def run_cell(corpus, scale, seed, cfg, cc, device, out):
    tag = f"{corpus}_f{scale:.2f}_s{seed}"
    if (out / "filter_variants" / f"{tag}.json").exists():
        print(f"[skip] {tag}"); return
    t0 = time.time()
    k, steps, N, lam = cfg["chunk_size"], cfg["sample_steps"], cc["n_best"], cfg.get("lambda_smooth", 0.0)
    W = np.load(f"data_processed/windows/{corpus}_stateaction_w60_s30.npy").astype(np.float32)
    we = np.load(f"data_processed/stats/{corpus}_stateaction_w60_s30_stats.npz", allow_pickle=True)["window_episode"]
    D = W.shape[2] // 2
    rng = np.random.default_rng([seed, int(round(scale * 100))])
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
    ax = Axes(c_tr, lambda c: c)
    amin, amax = c_tr.reshape(-1, D).min(0), c_tr.reshape(-1, D).max(0)
    bs = min(cfg["batch_size"], max(8, len(s_tr) // 4))
    rs = np.random.default_rng([seed, int(round(scale * 100)), 34])
    torch.manual_seed(3_400 + seed * 10 + int(round(scale * 4)))

    # ---- consistency on train (filter) and held-out (target)
    _, tr_id = pair_ids(s_tr, c_tr)
    ref = rs.permutation(len(s_tr))[: min(N_REF, len(s_tr))]
    rS, rA, r_id = nS(s_tr[ref]), nA(c_tr[ref]).reshape(len(ref), -1), tr_id[ref]
    con_tr = consistency(nS(s_tr), nA(c_tr).reshape(len(c_tr), -1), tr_id, rS, rA, r_id)
    te_first, _ = pair_ids(s_te, c_te)
    con_te_u = consistency(nS(s_te[te_first]), nA(c_te[te_first]).reshape(len(te_first), -1),
                           np.full(len(te_first), -1), rS, rA, r_id)

    a_tr = ax.act_tr
    masks = {"A25": a_tr >= np.percentile(a_tr, 25), "A75": a_tr >= np.percentile(a_tr, 75),
             "A_con": con_tr >= np.median(con_tr)}

    # ---- models: reuse A (=A50) and B2, train the three new filters
    ck_dir = Path(cc["out_dir"]) / "ckpt" / tag
    fv_dir = out / "ckpt_fv" / tag; fv_dir.mkdir(parents=True, exist_ok=True)
    fms = {}
    for arm, src, qd in (("A50", "A", 0), ("B2", "B2", 2)):
        ck = torch.load(ck_dir / f"{src}.pt", map_location=device, weights_only=False)
        assert np.allclose(ck["a_mean"], a_mean) and np.allclose(ck["s_std"], s_std), f"{tag}: rebuild mismatch"
        fm = make_fm(cfg, D, qd).to(device); fm.load_state_dict(ck["state"]); fm.eval(); fms[arm] = fm
    all_s, all_c = torch.from_numpy(nS(s_tr)), torch.from_numpy(nA(c_tr))
    val_idx = rs.permutation(np.where(ax.usable(c_te))[0])[: cc["val_chunks"]]
    g = torch.Generator().manual_seed(10_000 + seed)
    val = (torch.from_numpy(nS(s_te[val_idx])).to(device), torch.from_numpy(nA(c_te[val_idx])).to(device),
           torch.zeros(len(val_idx), 2).to(device),
           torch.randn(len(val_idx), k, D, generator=g).to(device), torch.rand(len(val_idx), generator=g).to(device))
    train_info = {}
    for arm, msk in masks.items():
        p = fv_dir / f"{arm}.pt"
        if p.exists():
            ck = torch.load(p, map_location=device, weights_only=False)
            fm = make_fm(cfg, D, 0).to(device); fm.load_state_dict(ck["state"]); fm.eval()
        else:
            tt = time.time()
            m_ = torch.from_numpy(msk)
            fm, h = train(cfg, D, 0, (all_s[m_], all_c[m_], torch.zeros(int(msk.sum()), 2)), bs, cfg["epochs"], seed, lam, device, val)
            torch.save({"state": fm.state_dict(), "hist": h, "a_mean": a_mean, "s_std": s_std}, p)
            train_info[arm] = {"kept": float(msk.mean()), "seconds": time.time() - tt, "final_val": h["val"][-1]}
        fms[arm] = fm

    # ---- targets
    saved = {a: np.load(out / "samples" / tag / f"{a}.npz") for a in ("A", "B2")}
    idx_act = saved["A"]["state_idx"]
    assert np.array_equal(idx_act, saved["B2"]["state_idx"])
    con_thr = float(np.median(con_te_u))
    idx_con = rs.permutation(te_first[con_te_u >= con_thr])[: cfg["eval_conditions"]]
    targets = {"act_usable": idx_act, "con_usable": idx_con}
    n_div = min(cfg["diversity_conditions"], cfg["eval_conditions"])

    res = {"corpus": corpus, "scale": scale, "seed": seed, "kept": {a: float(m.mean()) for a, m in masks.items()},
           "train": train_info, "con_threshold_heldout_unique_median": con_thr,
           "con_target_act_usable_frac": float(ax.usable(c_te[idx_con]).mean()),
           "act_target_con_usable_frac": None, "metrics": {}}
    for tname, idx in targets.items():
        c_ev = c_te[idx]; prof = smoothness_profile(c_ev)
        cond = torch.from_numpy(nS(s_te[idx])).to(device)
        res["metrics"][tname] = {}
        for arm in ("A25", "A50", "A75", "A_con", "B2"):
            if tname == "act_usable" and arm in ("A50", "B2"):
                z = saved["A" if arm == "A50" else "B2"]; gg, st = z["gen"], z["div"]
            else:
                qfn = (lambda shape: torch.from_numpy(ax.Q[rs.integers(0, len(ax.Q), shape)].astype(np.float32)).to(device)) \
                    if arm == "B2" else None
                gg = dA(sample(fms[arm], cond, None if qfn is None else qfn((len(idx),)), steps).cpu().numpy())
                qs = None if qfn is None else qfn((N, n_div))
                st = np.stack([dA(sample(fms[arm], cond[:n_div], None if qs is None else qs[i], steps).cpu().numpy())
                               for i in range(N)], 0)
            m = main_metrics(gg, c_ev, st, amin, amax, prof, cc["pr_k"])
            res["metrics"][tname][arm] = {kk: m[kk] for kk in ("recall", "precision", "diversity", "best_of_N", "inside")}
    res["seconds"] = time.time() - t0
    (out / "filter_variants").mkdir(parents=True, exist_ok=True)
    (out / "filter_variants" / f"{tag}.json").write_text(json.dumps(res), encoding="utf-8")
    M = res["metrics"]
    print(f"[done] {tag} {res['seconds'] / 60:.1f} min | kept " + " ".join(f"{a}={v*100:.0f}%" for a, v in res["kept"].items())
          + " | con-target recall " + " ".join(f"{a}={M['con_usable'][a]['recall']:.3f}" for a in M["con_usable"]), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/review_campaign.yaml")
    p.add_argument("--out", default="outputs/campaign_dedup")
    p.add_argument("--corpora", nargs="+"); p.add_argument("--seeds", type=int, nargs="+"); p.add_argument("--scales", type=float, nargs="+")
    a = p.parse_args()
    cc = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(cc["base_config"]).read_text(encoding="utf-8"))
    out = Path(a.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}", flush=True)
    for corpus in a.corpora or cc["corpora"]:
        for seed in a.seeds if a.seeds is not None else cc["seeds"]:
            for scale in a.scales or cc["scales"]:
                run_cell(corpus, scale, seed, cfg, cc, device, out)


if __name__ == "__main__":
    main()
