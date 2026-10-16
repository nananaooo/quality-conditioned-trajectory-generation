"""Review retraining campaign: A1 (update-matched Filtered), A2 (output-filtered), A3 (temperature-matched
Filtered + precision-recall curves), A4 (q-sampling ablation), A6 (RQ2 on all corpora) and the per-joint
z-scored-axis robustness pair, all evaluated on ONE set of models that are saved to disk.

One "cell" = (corpus, data scale, seed). For each cell the script trains the arms listed in the config
(checkpoint + per-epoch train/val loss saved), runs every evaluation, saves the generated samples used for
the main metrics, and writes one JSON of per-cell results. Existing cells/checkpoints are skipped, so the
run can be resumed. Paired statistics across seeds are computed separately (review_campaign_aggregate.py).

    .venv/Scripts/python.exe scripts/run_review_campaign.py --config configs/review_campaign.yaml
    (smoke test:  ... --epochs 1 --corpora koch_pickplace5 --seeds 0 --scales 0.25 --out outputs/campaign_smoke)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

from _bootstrap import add_project_root

add_project_root()

from src.eval.distribution_metrics import _pairwise_dist, per_joint_wasserstein, precision_recall  # noqa: E402
from src.models.flow_matching import ConditionalFlowConfig, ConditionalFlowField, conditional_flow_loss  # noqa: E402
from src.train.train_lstm_vae import episode_split  # noqa: E402
from run_conditional_flow import norm_stats  # noqa: E402
from run_chunk6_external import build_pairs_from_windows  # noqa: E402
from run_chunk31_smoothness_fidelity import smoothness_profile  # noqa: E402

QDIM = {"A": 0, "A_step": 0, "C": 0, "B1": 1, "B2": 2, "A_z": 0, "B2_z": 2}


# ----------------------------------------------------------------------------- quality axes
def act_raw(c):
    return np.abs(np.diff(c, axis=1)).sum(2).mean(1)


def jerk_raw(c):
    return np.abs(np.diff(c, axis=1, n=2)).sum(2).mean(1)


class Axes:
    """Quality definition on a training pool: raw activity/jerkiness -> z-scored q, tau, usable pool."""

    def __init__(self, c_tr, transform):
        self.transform = transform                      # identity or per-joint z-score of actions
        a, j = act_raw(transform(c_tr)), jerk_raw(transform(c_tr))
        self.am, self.asd = float(a.mean()), float(a.std() + 1e-8)
        self.jm, self.jsd = float(j.mean()), float(j.std() + 1e-8)
        self.tau = float(np.median(a))
        self.act_tr, self.jerk_tr = a, j
        self.q_tr = self.q_of(a, j)
        self.usable_tr = a >= self.tau
        self.Q = self.q_tr[self.usable_tr]

    def q_of(self, a, j):
        return np.stack([(a - self.am) / self.asd, -(j - self.jm) / self.jsd], 1).astype(np.float32)

    def usable(self, c):
        return act_raw(self.transform(c)) >= self.tau


# ----------------------------------------------------------------------------- model
def make_fm(cfg, D, qdim):
    return ConditionalFlowField(ConditionalFlowConfig(
        action_dim=D, state_dim=D, chunk_size=cfg["chunk_size"], hidden_dim=cfg["hidden_dim"],
        num_blocks=cfg["num_blocks"], kernel_size=cfg["kernel_size"], embed_dim=cfg["embed_dim"], quality_dim=qdim))


def val_loss(fm, val, lam, qdim, device):
    """Objective (eq 3.5) on a fixed held-out batch with fixed noise/time draws."""
    fm.eval()
    s, c, q, x0, u = val
    with torch.no_grad():
        x_u = (1 - u[:, None, None]) * x0 + u[:, None, None] * c
        x1_hat = fm(x_u, u, s, q[:, :qdim] if qdim else None)
        loss = torch.nn.functional.mse_loss(x1_hat, c)
        if lam > 0:
            loss = loss + lam * (torch.nn.functional.mse_loss(x1_hat[:, 1:] - x1_hat[:, :-1], c[:, 1:] - c[:, :-1])
                                 + torch.nn.functional.mse_loss(x1_hat[:, 2:] - 2 * x1_hat[:, 1:-1] + x1_hat[:, :-2],
                                                                c[:, 2:] - 2 * c[:, 1:-1] + c[:, :-2]))
    return float(loss)


def train(cfg, D, qdim, tensors, bs, epochs, seed, lam, device, val):
    """Same optimiser/clip/shuffle as the paper's training loop, plus per-epoch train/val loss."""
    torch.manual_seed(seed)
    fm = make_fm(cfg, D, qdim).to(device)
    opt = torch.optim.Adam(fm.parameters(), lr=cfg["lr"])
    loader = DataLoader(TensorDataset(*tensors), batch_size=bs, shuffle=True)
    hist = {"train": [], "val": [], "steps": 0}
    for _ in range(epochs):
        fm.train(); tot, n = 0.0, 0
        for batch in loader:
            batch = [t.to(device) for t in batch]
            bq = batch[2][:, :qdim] if qdim else None
            loss = conditional_flow_loss(fm, batch[1], batch[0], lambda_smooth=lam, quality=bq)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 5.0); opt.step()
            tot += float(loss) * len(batch[0]); n += len(batch[0]); hist["steps"] += 1
        hist["train"].append(tot / n)
        hist["val"].append(val_loss(fm, val, lam, qdim, device))
    fm.eval()
    return fm, hist


@torch.no_grad()
def sample(fm, cond, q=None, steps=50, temp=1.0):
    """Euler integration of the x1-prediction field (identical to conditional_sample) with x0 ~ N(0, T^2 I)."""
    b = cond.size(0)
    x = torch.randn(b, fm.cfg.chunk_size, fm.cfg.action_dim, device=cond.device) * temp
    dt = 1.0 / steps
    for i in range(steps):
        tv = i * dt
        x1 = fm(x, torch.full((b,), tv, device=cond.device), cond, q)
        x = x + (x1 - x) / max(1.0 - tv, 1e-3) * dt
    return x


# ----------------------------------------------------------------------------- metrics
def main_metrics(g, c_ev, stack, amin, amax, real_prof, pr_k):
    """g: [m,k,D] one sample per eval state; stack: [N,n_div,k,D] N samples at the first n_div states."""
    m = len(c_ev)
    pr, rc = precision_recall(c_ev.reshape(m, -1), g[:m].reshape(m, -1), k=pr_k)
    gp = smoothness_profile(g)
    n_div = stack.shape[1]
    return {"recall": rc, "precision": pr,
            "diversity": float(stack.std(0).mean()),
            "best_of_N": float(((stack - c_ev[:n_div][None]) ** 2).mean(axis=(2, 3)).min(0).mean()),
            "wass": per_joint_wasserstein(g[:m], c_ev), "inside": float(((g >= amin) & (g <= amax)).mean()),
            "motion": gp["d1"] / max(real_prof["d1"], 1e-12), "accel": gp["d2"] / max(real_prof["d2"], 1e-12),
            "jerk3": gp["d3"] / max(real_prof["d3"], 1e-12), "rough": gp["rough"] / max(real_prof["rough"], 1e-12)}


def rel_span(y):
    y = np.asarray(y, float)
    return float((y.max() - y.min()) / (abs(y.mean()) + 1e-12))


def nn_min_dist(x, ref, block=512):
    out = np.empty(len(x), np.float32)
    for i in range(0, len(x), block):
        out[i:i + block] = _pairwise_dist(x[i:i + block], ref).min(1)
    return out


def knn_idx(x, ref, k, block=512):
    out = np.empty((len(x), k), np.int64)
    for i in range(0, len(x), block):
        d = _pairwise_dist(x[i:i + block], ref)
        out[i:i + block] = np.argpartition(d, k, axis=1)[:, :k]
    return out


# ----------------------------------------------------------------------------- one cell
def run_cell(corpus, scale, seed, cfg, cc, device, out, epochs_override):
    tag = f"{corpus}_f{scale:.2f}_s{seed}"
    res_path = out / "results" / f"{tag}.json"
    if res_path.exists():
        print(f"[skip] {tag} (results exist)"); return
    t_cell = time.time()
    k, lam, steps = cfg["chunk_size"], cfg.get("lambda_smooth", 0.0), cfg["sample_steps"]
    epochs = epochs_override or cfg["epochs"]
    W = np.load(f"data_processed/windows/{corpus}_stateaction_w60_s30.npy").astype(np.float32)
    we = np.load(f"data_processed/stats/{corpus}_stateaction_w60_s30_stats.npz", allow_pickle=True)["window_episode"]
    D = W.shape[2] // 2
    np.random.seed(seed); torch.manual_seed(seed)
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
    bs = min(cfg["batch_size"], max(8, len(s_tr) // 4))

    axes = {"raw": Axes(c_tr, lambda c: c), "z": Axes(c_tr, nA)}
    amin, amax = c_tr.reshape(-1, D).min(0), c_tr.reshape(-1, D).max(0)
    result = {"corpus": corpus, "scale": scale, "seed": seed, "D": D, "n_train_pairs": int(len(s_tr)),
              "batch_size": bs, "tau": {n: ax.tau for n, ax in axes.items()},
              "usable_frac_train": {n: float(ax.usable_tr.mean()) for n, ax in axes.items()}, "arms": {}}

    # eval targets and validation batches per axis definition
    tgt = {}
    for n, ax in axes.items():
        te_us = np.where(ax.usable(c_te))[0]
        idx = rng.permutation(te_us)[: cfg["eval_conditions"]]
        vidx = rng.permutation(te_us)[: cc["val_chunks"]]
        vq = ax.q_of(act_raw(ax.transform(c_te[vidx])), jerk_raw(ax.transform(c_te[vidx])))
        g = torch.Generator().manual_seed(10_000 + seed)
        val = (torch.from_numpy(nS(s_te[vidx])).to(device), torch.from_numpy(nA(c_te[vidx])).to(device),
               torch.from_numpy(vq).to(device),
               torch.randn(len(vidx), k, D, generator=g).to(device), torch.rand(len(vidx), generator=g).to(device))
        tgt[n] = {"idx": idx, "val": val, "real_prof": smoothness_profile(c_te[idx]), "n_usable_heldout": int(len(te_us))}
        result[f"n_eval_{n}"] = int(len(idx))

    # ---------------- training (or loading) every arm
    ck_dir = out / "ckpt" / tag; ck_dir.mkdir(parents=True, exist_ok=True)
    all_t = (torch.from_numpy(nS(s_tr)), torch.from_numpy(nA(c_tr)))
    nb_all = math.ceil(len(s_tr) / bs)
    fms, hists = {}, {}
    for arm in cc["arms"]:
        ax = axes["z" if arm.endswith("_z") else "raw"]
        qd = QDIM[arm]
        if arm in ("A", "A_step", "A_z"):
            u = ax.usable_tr
            tensors = (all_t[0][u], all_t[1][u], torch.from_numpy(ax.q_tr[u]))
            ep = epochs
            if arm == "A_step":
                ep = int(round(epochs * nb_all / math.ceil(u.sum() / bs)))
        else:
            tensors = (*all_t, torch.from_numpy(ax.q_tr))
            ep = epochs
        p = ck_dir / f"{arm}.pt"
        if p.exists():
            ck = torch.load(p, map_location=device, weights_only=False)
            fm = make_fm(cfg, D, qd).to(device); fm.load_state_dict(ck["state"]); fm.eval(); h = ck["hist"]
        else:
            t0 = time.time()
            fm, h = train(cfg, D, qd, tensors, bs, ep, seed, lam, device, tgt["z" if arm.endswith("_z") else "raw"]["val"])
            h["epochs"], h["train_seconds"] = ep, time.time() - t0
            torch.save({"state": fm.state_dict(), "hist": h, "qdim": qd, "D": D, "s_mean": s_mean, "s_std": s_std,
                        "a_mean": a_mean, "a_std": a_std, "axes": arm.endswith("_z") and "z" or "raw"}, p)
        fms[arm], hists[arm] = fm, h
        tail = h["val"][-10:]
        result["arms"].setdefault(arm, {})["train"] = {
            "epochs": h["epochs"], "steps": h["steps"], "seconds": h.get("train_seconds"),
            "final_train_loss": h["train"][-1], "final_val_loss": h["val"][-1],
            "val_change_last10_rel": float((tail[-1] - tail[0]) / (abs(tail[0]) + 1e-12)) if len(tail) > 1 else None,
            "train_curve": h["train"], "val_curve": h["val"]}
        print(f"  {tag} {arm}: epochs={h['epochs']} steps={h['steps']} val={h['val'][-1]:.4f} "
              f"({h.get('train_seconds') or 0:.0f}s)", flush=True)

    samp_dir = out / "samples" / tag; samp_dir.mkdir(parents=True, exist_ok=True)
    N = cc["n_best"]

    def q_draw(ax, arm, shape):
        if QDIM[arm] == 0:
            return None
        qq = ax.Q[rng.integers(0, len(ax.Q), shape)][..., : QDIM[arm]]
        return torch.from_numpy(qq.astype(np.float32)).to(device)

    def gen_main(arm, ax, idx, q_fn=None, temp=1.0):
        cond = torch.from_numpy(nS(s_te[idx])).to(device)
        n_div = min(cfg["diversity_conditions"], len(idx))
        qm = q_fn(len(idx)) if q_fn else q_draw(ax, arm, (len(idx),))
        g = dA(sample(fms[arm], cond, qm, steps, temp).cpu().numpy())
        st = []
        qs = q_fn((N, n_div)) if q_fn else q_draw(ax, arm, (N, n_div))
        for i in range(N):
            st.append(dA(sample(fms[arm], cond[:n_div], None if qs is None else qs[i], steps, temp).cpu().numpy()))
        return g, np.stack(st, 0)

    # ---------------- E1 main metrics (all arms) + saved samples
    stacks = {}
    for arm in cc["arms"]:
        n = "z" if arm.endswith("_z") else "raw"; idx = tgt[n]["idx"]
        g, st = gen_main(arm, axes[n], idx)
        result["arms"][arm]["main"] = main_metrics(g, c_te[idx], st, amin, amax, tgt[n]["real_prof"], cc["pr_k"])
        np.savez_compressed(samp_dir / f"{arm}.npz", gen=g.astype(np.float32), div=st.astype(np.float32),
                            state_idx=idx, div_state_idx=idx[: st.shape[1]])
        stacks[arm] = (g, st)

    ax = axes["raw"]; idx = tgt["raw"]["idx"]; c_ev = c_te[idx]
    cond_all = torch.from_numpy(nS(s_te[idx])).to(device)
    n_div = min(cfg["diversity_conditions"], len(idx))

    # ---------------- E2 (A2) output-filtered from the Full-Unconditioned model
    if "C" in fms:
        tries = cc["reject_max_tries"]
        acc_g = np.full((len(idx), k, D), np.nan, np.float32); have = np.zeros(len(idx), bool); n_try = n_acc = 0
        for _ in range(tries):
            todo = np.where(~have)[0]
            if len(todo) == 0: break
            x = dA(sample(fms["C"], cond_all[todo], None, steps).cpu().numpy())
            ok = act_raw(x) >= ax.tau; n_try += len(todo); n_acc += int(ok.sum())
            acc_g[todo[ok]] = x[ok]; have[todo[ok]] = True
        div_acc = [[] for _ in range(n_div)]; n_try_d = n_acc_d = 0
        for _ in range(tries):
            todo = np.array([i for i in range(n_div) if len(div_acc[i]) < N])
            if len(todo) == 0: break
            x = dA(sample(fms["C"], cond_all[todo], None, steps).cpu().numpy())
            ok = act_raw(x) >= ax.tau; n_try_d += len(todo); n_acc_d += int(ok.sum())
            for j, i in enumerate(todo):
                if ok[j]: div_acc[i].append(x[j])
        full = [i for i in range(n_div) if len(div_acc[i]) >= N]
        st = np.stack([np.stack(div_acc[i][:N], 0) for i in full], 1) if full else None
        sel = np.where(have)[0]
        m = main_metrics(acc_g[sel], c_ev[sel], st if st is not None else np.zeros((N, 0, k, D)), amin, amax,
                         tgt["raw"]["real_prof"], cc["pr_k"]) if len(sel) > 3 else {}
        if st is not None:   # diversity / best-of-N only over states that filled N accepted samples
            m["diversity"] = float(st.std(0).mean())
            m["best_of_N"] = float(((st - c_ev[full][None]) ** 2).mean(axis=(2, 3)).min(0).mean())
        m.update({"accept_rate_pr": n_acc / max(n_try, 1), "frac_states_no_accept": float(1 - have.mean()),
                  "accept_rate_div": n_acc_d / max(n_try_d, 1), "frac_div_states_short": float(1 - len(full) / n_div)})
        result["arms"]["OutFilt"] = {"main": m}
        np.savez_compressed(samp_dir / "OutFilt.npz", gen=acc_g, accepted=have, state_idx=idx,
                            div=st if st is not None else np.zeros(0), div_state_idx=idx[full] if full else np.zeros(0))

    # ---------------- E3 (A3) temperature sweep on Filtered
    if "A" in fms:
        ts = {}
        for T in cc["temperatures"]:
            g, st = gen_main("A", ax, idx, temp=T)
            ts[str(T)] = main_metrics(g, c_ev, st, amin, amax, tgt["raw"]["real_prof"], cc["pr_k"])
        result["temperature_sweep_A"] = ts

    # ---------------- E3 (A3) precision-recall curves via output post-filter (arms A, C, B1, B2)
    cidx = rng.permutation(np.where(ax.usable(c_te))[0])[: cc["curve_states"]]
    ccond = torch.from_numpy(nS(s_te[cidx])).to(device)
    ref = c_tr[rng.permutation(np.where(ax.usable_tr)[0])[: cc["curve_ref_n"]]].reshape(-1, k * D)
    real_pool = c_te[cidx].reshape(len(cidx), -1)
    pr_n = min(cc["curve_pr_n"], len(cidx))
    real_pr = real_pool[rng.permutation(len(cidx))[:pr_n]]
    curves = {}
    for arm in [a for a in ("A", "C", "B1", "B2") if a in fms]:
        M = cc["curve_M"]
        qs = q_draw(ax, arm, (M, len(cidx)))
        dr = np.stack([dA(sample(fms[arm], ccond, None if qs is None else qs[i], steps).cpu().numpy()) for i in range(M)], 0)
        dist = nn_min_dist(dr.reshape(M * len(cidx), -1), ref).reshape(M, len(cidx))
        curves[arm] = {}
        for rho in cc["curve_rhos"]:
            kk = max(2, int(round(rho * M)))
            order = np.argsort(dist, axis=0)[:kk]
            sel = np.take_along_axis(dr, order[:, :, None, None], axis=0)
            pooled = sel.reshape(kk * len(cidx), -1)
            pr, rc = precision_recall(real_pr, pooled[rng.permutation(len(pooled))[:pr_n]], k=cc["pr_k"])
            curves[arm][str(rho)] = {"precision": pr, "recall": rc, "diversity": float(sel.std(0).mean())}
    result["pr_curves"] = curves

    # ---------------- E4 (A4) q-sampling modes for B2 and B1
    tr_state_n = nS(s_tr)
    nbr = knn_idx(nS(s_te[idx]), tr_state_n, cc["qmode_state_knn"])
    fallback = 0

    def state_cond_q(arm):
        nonlocal fallback
        pools = []
        for i in range(len(idx)):
            u = nbr[i][ax.usable_tr[nbr[i]]]
            if len(u) == 0: fallback += 1; pools.append(None)
            else: pools.append(ax.q_tr[u][:, : QDIM[arm]])

        def fn(shape):
            shape = (shape,) if isinstance(shape, int) else shape
            n_states = shape[-1]; out_q = np.empty((*shape, QDIM[arm]), np.float32)
            for i in range(n_states):
                pool = pools[i] if pools[i] is not None else ax.Q[:, : QDIM[arm]]
                pick = pool[rng.integers(0, len(pool), shape[:-1] or (1,))]
                if len(shape) == 1: out_q[i] = pick[0]
                else: out_q[:, i] = pick
            return torch.from_numpy(out_q).to(device)
        return fn

    qm_res = {}
    for arm in [a for a in ("B2", "B1") if a in fms]:
        qd = QDIM[arm]
        med = np.median(ax.Q[:, :qd], axis=0).astype(np.float32)

        def per_state(shape, _qd=qd):
            shape = (shape,) if isinstance(shape, int) else shape
            base = ax.Q[rng.integers(0, len(ax.Q), shape[-1])][:, :_qd]
            return torch.from_numpy(np.broadcast_to(base, (*shape, _qd)).astype(np.float32).copy()).to(device)

        def glob(shape, _m=med):
            shape = (shape,) if isinstance(shape, int) else shape
            return torch.from_numpy(np.broadcast_to(_m, (*shape, len(_m))).astype(np.float32).copy()).to(device)

        fallback = 0
        sc = state_cond_q(arm)
        qm_res[arm] = {"default": result["arms"][arm]["main"]}
        for mode, fn in (("per_state_fixed", per_state), ("global_median", glob), ("state_conditional", sc)):
            g, st = gen_main(arm, ax, idx, q_fn=fn)
            qm_res[arm][mode] = main_metrics(g, c_ev, st, amin, amax, tgt["raw"]["real_prof"], cc["pr_k"])
        qm_res[arm]["state_conditional_fallback_frac"] = fallback / len(idx)
    result["q_modes"] = qm_res

    # ---------------- E5 (A6) RQ2 on this corpus (full data only)
    if scale == 1.0 and "B2" in fms:
        ridx = rng.permutation(len(s_te))[: cc["rq2_states"]]
        rcond = torch.from_numpy(nS(s_te[ridx])).to(device)
        qa, qs_ = ax.q_tr[:, 0], ax.q_tr[:, 1]
        rq2 = {}
        for axis_i, name in ((0, "activity"), (1, "smoothness")):
            tz = qa if axis_i == 0 else qs_; oz = qs_ if axis_i == 0 else qa
            levels = np.quantile(tz, cc["rq2_levels"]).astype(np.float32)
            cond_other, base_t, base_o = [], [], []
            for p in cc["rq2_levels"]:
                lo, hi = np.quantile(tz, max(0, p - cc["rq2_band"])), np.quantile(tz, min(1, p + cc["rq2_band"]))
                b = (tz >= lo) & (tz <= hi)
                cond_other.append(float(oz[b].mean()))
                base_t.append(float((ax.act_tr if axis_i == 0 else -ax.jerk_tr)[b].mean()))
                base_o.append(float((ax.jerk_tr if axis_i == 0 else ax.act_tr)[b].mean()))
            out_axis = {"levels": levels.tolist(),
                        "data_baseline": {"target_means": base_t, "other_means": base_o,
                                          "leak": rel_span(base_o) / rel_span(base_t)}}
            for sweep, others in (("basic", [0.0] * len(levels)), ("conditional", cond_other)):
                rt, ro, ins = [], [], []
                for z, o in zip(levels, others):
                    qv = np.zeros((len(ridx), 2), np.float32); qv[:, axis_i] = z; qv[:, 1 - axis_i] = o
                    g = dA(sample(fms["B2"], rcond, torch.from_numpy(qv).to(device), steps).cpu().numpy())
                    a_, j_ = act_raw(g), jerk_raw(g)
                    rt.append(float(a_.mean() if axis_i == 0 else -j_.mean()))
                    ro.append(float(j_.mean() if axis_i == 0 else a_.mean()))
                    ins.append(float(((g >= amin) & (g <= amax)).mean()))
                out_axis[sweep] = {"other_values": list(map(float, others)), "realized_target": rt, "realized_other": ro,
                                   "control_spearman": float(spearmanr(levels, rt).statistic),
                                   "leak": rel_span(ro) / rel_span(rt), "inrange_min": min(ins), "inrange_mean": float(np.mean(ins))}
            rq2[name] = out_axis
        result["rq2"] = rq2

    result["cell_seconds"] = time.time() - t_cell
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(result), encoding="utf-8")
    print(f"[done] {tag} in {result['cell_seconds'] / 60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/review_campaign.yaml")
    p.add_argument("--corpora", nargs="+"); p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--scales", type=float, nargs="+"); p.add_argument("--epochs", type=int)
    p.add_argument("--out")
    a = p.parse_args()
    cc = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(cc["base_config"]).read_text(encoding="utf-8"))
    out = Path(a.out or cc["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / "campaign_config_used.yaml").write_text(yaml.safe_dump({"campaign": cc, "base": cfg, "cli": vars(a)}), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} | out={out}", flush=True)
    for corpus in a.corpora or cc["corpora"]:
        for seed in a.seeds if a.seeds is not None else cc["seeds"]:
            for scale in a.scales or cc["scales"]:
                run_cell(corpus, scale, seed, cfg, cc, device, out, a.epochs)


if __name__ == "__main__":
    main()
