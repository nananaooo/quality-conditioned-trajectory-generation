"""Review 12(b): RQ1 validity against the workshop (BECS 2026) rule applied AS PUBLISHED.

Workshop Algorithm 1 (Quality-aware paper/BECS workshop paper 3차 피드백 반영.md):
  V[t] = sum_j |s_t,j - s_t-1,j|  on the continuous session stream
  Vs   = rolling_mean(V, 5)
  tau  = 0.5 * percentile(Vs, 70)            -- PER SESSION
  label = motion if Vs > tau else idle
  segments with < 30 frames take the label of the preceding segment, then re-segment
  quality: usable = motion & dur >= 2.0 s & mean variation >= 1.0
           low-info = idle & mean variation < 0.6, else uncertain

The paper's RQ1 (chunk 16) instead smoothed inside 60-frame windows, used ONE corpus-wide tau,
and skipped the merge. Here: rebuild the exact chunk set (windows w60 s30 from the continuous
episodes, verified equal to the .npy), label every frame with the workshop rule, and recompute
Spearman(activity, Vs), motion/idle AUC, mean-activity ratio (tie-aware ranks, Eq. 4.2).
Also: fidelity vs workshop Table 2/4 (segment counts, label counts) and agreement with the
chunk-16 labels. Sensitivity: rolling alignment (trailing pandas default vs centered),
mean-variation on raw V vs smoothed V, start-frame label vs majority label over the chunk.
Analysis only, no training, deterministic.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from _bootstrap import add_project_root

add_project_root()

from src.data.lerobot_reader import find_dataset_roots, group_episode_frames, read_data_frame, sort_frames, stack_feature_matrix  # noqa: E402
from src.data.windowing import slice_episode  # noqa: E402
from run_chunk16_quality_validity import frame_state_variation, rolling_mean  # noqa: E402

K, WIN, STRIDE, FPS, MIN_SEG = 8, 60, 30, 30, 30
# workshop Table 4: (#segments, #usable, #low-info, #uncertain) for S1..S20
WS_TABLE4 = [(7, 3, 4, 0), (11, 4, 6, 1), (9, 3, 5, 1), (7, 3, 3, 1), (9, 4, 5, 0), (7, 3, 4, 0),
             (9, 4, 5, 0), (9, 4, 4, 1), (12, 4, 5, 3), (7, 2, 3, 2), (7, 3, 4, 0), (9, 4, 4, 1),
             (11, 3, 4, 4), (8, 3, 4, 1), (8, 3, 4, 1), (9, 3, 4, 2), (8, 3, 4, 1), (7, 3, 3, 1),
             (7, 3, 4, 0), (10, 3, 5, 2)]


def auc_ties(scores, labels):
    r = rankdata(scores); n1 = int(labels.sum()); n0 = len(labels) - n1
    return float((r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def runs(lab):
    """[(start, end_exclusive, label)] of consecutive equal labels."""
    edges = np.flatnonzero(np.diff(lab.astype(int))) + 1
    starts = np.r_[0, edges]; ends = np.r_[edges, len(lab)]
    return [(int(a), int(b), int(lab[a])) for a, b in zip(starts, ends)]


def workshop_labels(V, center, mv_on_smooth):
    """Frame labels for one session. V[i] = |s[i+1]-s[i]| summed (len T-1)."""
    if center:
        Vs = rolling_mean(V[None], 5)[0]
    else:
        Vs = pd.Series(V).rolling(5, min_periods=1).mean().to_numpy()
    tau = 0.5 * np.percentile(Vs, 70)
    raw = Vs > tau
    lab = raw.copy()
    segs = runs(raw)
    cur = [s[2] for s in segs]
    for i, (a, b, _) in enumerate(segs):          # sequential pass, preceding label as updated
        if b - a < MIN_SEG and i > 0:
            cur[i] = cur[i - 1]
        lab[a:b] = cur[i]
    final = runs(lab)
    mv_src = Vs if mv_on_smooth else V
    q = []
    for a, b, m in final:
        dur, mv = (b - a) / FPS, float(mv_src[a:b].mean())
        if m == 1 and dur >= 2.0 and mv >= 1.0:
            q.append("usable")
        elif m == 0 and mv < 0.6:
            q.append("low")
        else:
            q.append("unc")
    qlab = np.empty(len(lab), dtype=object)
    for (a, b, _), qq in zip(final, q):
        qlab[a:b] = qq
    return Vs, tau, raw, lab, len(runs(raw)), final, q, qlab


def main():
    root = Path("data_raw/physicai")
    roots = find_dataset_roots(root)
    sess_num = [int(re.search(r"_(\d+)$", r.name).group(1)) for r in roots]

    npy = np.load("data_processed/windows/physicai_stateaction_w60_s30.npy").astype(np.float32)
    streams = []
    for r in roots:
        for _, ep in group_episode_frames(sort_frames(read_data_frame(r))):
            streams.append(stack_feature_matrix(ep, ["observation.state", "action"]))
    assert len(streams) == 20, len(streams)

    # rebuild windows in the builder's order and keep (episode, start frame) per window
    wins, meta = [], []
    for e, x in enumerate(streams):
        for w_i, w in enumerate(slice_episode(x, WIN, STRIDE)):
            wins.append(w); meta.append((e, w_i * STRIDE))
    wins = np.asarray(wins, np.float32)
    assert wins.shape == npy.shape and np.array_equal(wins, npy), "rebuilt windows != npy"
    print(f"rebuilt windows match npy: {wins.shape}, sessions={len(streams)}")

    # chunk order identical to chunk16: loop t outer, window inner
    N = len(wins)
    act = np.concatenate([np.abs(np.diff(wins[:, t:t + K, 6:], axis=1)).sum(2).mean(1) for t in range(WIN - K)])
    ep_idx = np.concatenate([np.array([m[0] for m in meta]) for _ in range(WIN - K)])
    fr_idx = np.concatenate([np.array([m[1] for m in meta]) + t for t in range(WIN - K)])

    # chunk-16 labels (current paper) for agreement
    Vw = frame_state_variation(wins[:, :, :6]); Vws = rolling_mean(Vw, 5)
    tau16 = 0.5 * np.percentile(Vws, 70)
    lab16 = np.concatenate([(Vws[:, t] >= tau16) for t in range(WIN - K)]).astype(int)

    variants = [("trailing, mv=raw V (primary)", False, False), ("centered, mv=raw V", True, False),
                ("trailing, mv=smoothed V", False, True), ("centered, mv=smoothed V", True, True)]
    out = []
    for name, center, mv_s in variants:
        per = [workshop_labels(frame_state_variation(x[None, :, :6])[0], center, mv_s) for x in streams]
        # fidelity vs workshop tables
        n_raw = [p[4] for p in per]; n_fin = [len(p[5]) for p in per]
        seglen = [np.mean([b - a for a, b, _ in p[5]]) for p in per]
        qc = [(p[6].count("usable"), p[6].count("low"), p[6].count("unc")) for p in per]
        tot = np.sum(qc, axis=0)
        # per-session match to Table 4 (session numbers from folder names)
        t4 = [WS_TABLE4[n - 1] for n in sess_num]
        seg_exact = sum(int(n_fin[i] == t4[i][0]) for i in range(20))
        frame_idle = 1 - np.mean(np.concatenate([p[3] for p in per]))

        Vs_c = np.array([per[e][0][f] for e, f in zip(ep_idx, fr_idx)])
        lab_c = np.array([per[e][3][f] for e, f in zip(ep_idx, fr_idx)]).astype(int)
        # majority label over the chunk's frames [f, f+K-1) (V has T-1 entries)
        lab_maj = np.array([per[e][3][f:f + K - 1].mean() >= 0.5 for e, f in zip(ep_idx, fr_idx)]).astype(int)
        q_c = np.array([per[e][7][f] for e, f in zip(ep_idx, fr_idx)])

        sp = float(spearmanr(act, Vs_c).statistic)
        a_start = auc_ties(act, lab_c); a_maj = auc_ties(act, lab_maj)
        ratio = act[lab_c == 1].mean() / act[lab_c == 0].mean()
        m_ui = np.isin(q_c, ["usable", "low"])
        a_q = auc_ties(act[m_ui], (q_c[m_ui] == "usable").astype(int))
        agree = float((lab_c == lab16).mean())
        out.append(dict(name=name, sp=sp, a_start=a_start, a_maj=a_maj, ratio=ratio, a_q=a_q, agree=agree,
                        m1=act[lab_c == 1].mean(), m0=act[lab_c == 0].mean(), frame_idle=frame_idle,
                        chunk_idle=1 - lab_c.mean(), n_raw=np.mean(n_raw), n_fin=np.mean(n_fin),
                        seglen=np.mean(seglen), tot=tot, seg_exact=seg_exact, n_fin_list=n_fin,
                        taus=[p[1] for p in per], q_share=np.mean(np.isin(q_c, ["usable", "low"]))))

    print(f"\ncurrent paper (chunk 16): corpus tau={tau16:.4f}, window-internal smoothing, no merge")
    print("reported: Spearman 0.824, AUC 0.905, mean act motion 2.042 / idle 0.355 (5.8x)\n")
    for o in out:
        print(f"=== workshop rule: {o['name']} ===")
        print(f"  session tau range {min(o['taus']):.3f}-{max(o['taus']):.3f} (corpus tau16 {tau16:.3f})")
        print(f"  fidelity: segments/session before merge {o['n_raw']:.2f} (ws Table2 56.35), "
              f"after {o['n_fin']:.2f} (ws Table2 35.05 / Table3 8.55), mean len {o['seglen']:.1f} fr (ws 54.37)")
        print(f"            quality counts usable/low/unc = {o['tot'].tolist()} of {o['tot'].sum()} "
              f"(ws 65/84/22 of 171); sessions with exact Table-4 segment count: {o['seg_exact']}/20")
        print(f"  frame idle {o['frame_idle']:.3f} | chunk(start-frame) idle {o['chunk_idle']:.3f} | "
              f"agreement with chunk-16 labels {o['agree']:.3f}")
        print(f"  Spearman(act, Vs)            = {o['sp']:.3f}")
        print(f"  AUC act motion/idle (start)  = {o['a_start']:.3f}   (majority-of-chunk label: {o['a_maj']:.3f})")
        print(f"  mean act motion {o['m1']:.3f} / idle {o['m0']:.3f} = {o['ratio']:.1f}x")
        print(f"  AUC act usable vs low-info segments (uncertain excluded, {o['q_share']*100:.1f}% of chunks) = {o['a_q']:.3f}\n")
    print("per-session final segment counts (primary) vs ws Table 4, by session number:")
    order = np.argsort(sess_num)
    print("  ours:", [out[0]['n_fin_list'][i] for i in order])
    print("  ws  :", [WS_TABLE4[sess_num[i] - 1][0] for i in order])


if __name__ == "__main__":
    main()
