"""Paired statistics for the trade-off curves (outputs/campaign/tradeoff/*.json).

1. T = 1 direct comparison: QCs - F, QCm - F (recall, precision, diversity, best_of_N).
2. Precision-matched recall: per seed p* = precision of F at T = 1. On each curve, walk T in increasing
   order and linearly interpolate recall at the FIRST segment whose precision crosses p* (T = 1 exactly
   for F). Seeds whose curve never reaches p* are 'unreachable' and excluded from the test (counted).
3. Diversity-matched: per seed d* = diversity of QCs (resp. QCm) at T = 1. On the F curve interpolate
   recall, precision, best_of_N at the first crossing of d*; compare with QCs (resp. QCm) at T = 1.
   Unreachable if F never reaches d* (max F diversity reported).
Holm correction over the 8-cell family for each (contrast, analysis, metric). Verdict counts per the task:
cells where the QC variant is significantly better on >= 1 metric and not significantly worse on any.
Also writes the figure CSVs (per seed and mean). Output: outputs/campaign/tradeoff/tradeoff_aggregate.md

    .venv/Scripts/python.exe scripts/review_tradeoff_aggregate.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import os
D = Path(os.environ.get("TRADEOFF_DIR", "outputs/campaign/tradeoff"))  # TRADEOFF_DIR=outputs/campaign_dedup/tradeoff for the de-duplicated re-evaluation
CORP = ["being_so101", "rayenghali", "koch_pickplace5", "aloha_tape"]
SHORT = {"being_so101": "being", "rayenghali": "rayen", "koch_pickplace5": "koch", "aloha_tape": "aloha"}
SCALES = [1.0, 0.25]
MET = ["recall", "precision", "diversity", "best_of_N"]
LOWER = {"best_of_N"}


def interp_at(curve, temps, key, target, out_keys):
    """First crossing of curve[key] == target walking T upward; linear interpolation of out_keys."""
    xs = [curve[str(t)][key] for t in temps]
    for i in range(len(temps) - 1):
        a, b = xs[i], xs[i + 1]
        if (a - target) * (b - target) <= 0 and a != b:
            w = (target - a) / (b - a)
            return {k: curve[str(temps[i])][k] + w * (curve[str(temps[i + 1])][k] - curve[str(temps[i])][k]) for k in out_keys}
        if a == target:
            return {k: curve[str(temps[i])][k] for k in out_keys}
    if xs[-1] == target:
        return {k: curve[str(temps[-1])][k] for k in out_keys}
    return None


def paired(d, metric):
    d = np.asarray(d, float); n = len(d)
    if n < 2:
        return dict(mean=float(d.mean()) if n else np.nan, hw=np.nan, p=np.nan, k=int(((d < 0) if metric in LOWER else (d > 0)).sum()), n=n)
    hw = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    p = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else (0.0 if d.mean() != 0 else 1.0)
    better = (d < 0) if metric in LOWER else (d > 0)
    return dict(mean=float(d.mean()), hw=float(hw), p=float(p), k=int(better.sum()), n=n)


def holm(ps):
    ps = np.nan_to_num(np.asarray(ps, float), nan=1.0); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def main():
    R = defaultdict(dict)
    for f in sorted(D.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8")); R[(r["corpus"], r["scale"])][r["seed"]] = r
    temps = next(iter(next(iter(R.values())).values()))["temps"]
    cells = [(c, s) for c in CORP for s in SCALES if (c, s) in R]
    rows = []  # (analysis, contrast, metric, cell, stat, unreachable_count, extra)

    for c, s in cells:
        seeds = sorted(R[(c, s)])
        cur = {sd: R[(c, s)][sd]["curves"] for sd in seeds}
        for v in ("QCs", "QCm"):
            # 1. T = 1 direct
            for m in MET:
                d = [cur[sd][v]["1.0"][m] - cur[sd]["F"]["1.0"][m] for sd in seeds]
                rows.append(("1_T1", f"{v}-F", m, (c, s), paired(d, m), 0, ""))
            # 2. precision-matched recall
            d, unr = [], 0
            for sd in seeds:
                pstar = cur[sd]["F"]["1.0"]["precision"]
                q = interp_at(cur[sd][v], temps, "precision", pstar, ["recall"])
                if q is None:
                    unr += 1; continue
                d.append(q["recall"] - cur[sd]["F"]["1.0"]["recall"])
            rows.append(("2_prec_matched", f"{v}-F", "recall", (c, s), paired(d, "recall"), unr, ""))
            # 3. diversity-matched
            ds = {m: [] for m in ("recall", "precision", "best_of_N")}; unr = 0; fmax = []
            for sd in seeds:
                dstar = cur[sd][v]["1.0"]["diversity"]
                fq = interp_at(cur[sd]["F"], temps, "diversity", dstar, list(ds))
                fmax.append(max(cur[sd]["F"][str(t)]["diversity"] for t in temps))
                if fq is None:
                    unr += 1; continue
                for m in ds:
                    ds[m].append(cur[sd][v]["1.0"][m] - fq[m])
            for m in ds:
                rows.append(("3_div_matched", f"{v}-F", m, (c, s), paired(ds[m], m), unr,
                             f"F max diversity {np.mean(fmax):.4f} vs d* {np.mean([cur[sd][v]['1.0']['diversity'] for sd in seeds]):.4f}"))

    # Holm per (analysis, contrast, metric) family over the 8 cells
    fam = defaultdict(list)
    for i, r in enumerate(rows):
        fam[(r[0], r[1], r[2])].append(i)
    padj = np.full(len(rows), np.nan)
    for key, ids in fam.items():
        adj = holm([rows[i][4]["p"] for i in ids])
        for j, i in enumerate(ids):
            padj[i] = adj[j]

    L = []; w = L.append
    w(f"# 절충 곡선 분석 집계 (셀 {sum(len(v) for v in R.values())}/40)\n")
    w("표기: 평균 ± t 95% CI 반폭 (p, p_Holm[8칸 가족], 앞선 seed 수/검정 seed 수). \\* = p < 0.05, † = p_Holm < 0.05. "
      "best_of_N은 낮을수록 좋다. '도달 불가' = 곡선이 목표값을 가로지르지 않은 seed 수(검정에서 제외).\n")
    titles = {"1_T1": "1. T = 1 직접 비교", "2_prec_matched": "2. 정밀도를 맞춘 recall (p* = F의 T = 1 precision)",
              "3_div_matched": "3. 다양성을 맞춘 비교 (d* = QC 변형의 T = 1 diversity, F 곡선에서 보간)"}

    def cellfmt(st, padj_v, unr, prec):
        if st["n"] == 0:
            return f"도달 불가 {unr}/5"
        star = ("\\*" if st["p"] < .05 else "") + ("†" if padj_v < .05 else "")
        s = (f"{st['mean']:+.{prec}f} ± {st['hw']:.{prec}f}{star} (p={st['p']:.3f}, p_H={padj_v:.3f}, {st['k']}/{st['n']})"
             if st["n"] >= 2 else f"{st['mean']:+.{prec}f} (n={st['n']})")
        return s + (f", 도달 불가 {unr}" if unr else "")

    for an in ("1_T1", "2_prec_matched", "3_div_matched"):
        w(f"## {titles[an]}\n")
        mets = MET if an == "1_T1" else (["recall"] if an == "2_prec_matched" else ["recall", "precision", "best_of_N"])
        for v in ("QCs", "QCm"):
            w(f"**{v} − F**\n")
            w("| 코퍼스 | 규모 | " + " | ".join(mets) + (" | 비고 |" if an == "3_div_matched" else " |"))
            w("|---|---|" + "---|" * len(mets) + ("---|" if an == "3_div_matched" else ""))
            for c, s in cells:
                vals, note = [], ""
                for m in mets:
                    i = next(i for i, r in enumerate(rows) if r[0] == an and r[1] == f"{v}-F" and r[2] == m and r[3] == (c, s))
                    prec = 4 if c == "aloha_tape" and m in ("diversity", "best_of_N") else 3
                    vals.append(cellfmt(rows[i][4], padj[i], rows[i][5], prec)); note = rows[i][6]
                w(f"| {SHORT[c]} | {s} | " + " | ".join(vals) + (f" | {note} |" if an == "3_div_matched" else " |"))
            w("")

    # verdict counts
    w("## 4. 판정 기준 칸 수\n")
    w("기준: 해당 분석에서 QC 변형이 F보다 적어도 한 지표에서 유의하게 좋고(p < 0.05), 나머지 지표에서 유의하게 나쁘지 않은 칸. "
      "Holm 기준(p_Holm < 0.05)도 함께 센다. 도달 불가 seed가 있는 칸은 남은 seed로 판정하며, 검정 seed가 2개 미만이면 판정하지 않는다.\n")
    w("| 분석 | 변형 | 충족 칸 (p) | 충족 칸 (Holm) | 판정 불가 칸 |")
    w("|---|---|---|---|---|")
    for an in ("2_prec_matched", "3_div_matched", "1_T1"):
        mets = ["recall"] if an == "2_prec_matched" else (["recall", "precision", "best_of_N"] if an == "3_div_matched" else ["recall", "precision", "best_of_N"])
        for v in ("QCs", "QCm"):
            ok_p = ok_h = na = 0; which = []
            for c, s in cells:
                ids = [i for i, r in enumerate(rows) if r[0] == an and r[1] == f"{v}-F" and r[2] in mets and r[3] == (c, s)]
                if any(rows[i][4]["n"] < 2 for i in ids):
                    na += 1; continue
                def good(i, alpha_p):
                    st = rows[i][4]; better = st["mean"] < 0 if rows[i][2] in LOWER else st["mean"] > 0
                    return alpha_p < .05 and better
                def bad(i, alpha_p):
                    st = rows[i][4]; worse = st["mean"] > 0 if rows[i][2] in LOWER else st["mean"] < 0
                    return alpha_p < .05 and worse
                if any(good(i, rows[i][4]["p"]) for i in ids) and not any(bad(i, rows[i][4]["p"]) for i in ids):
                    ok_p += 1; which.append(f"{SHORT[c]} {s}")
                if any(good(i, padj[i]) for i in ids) and not any(bad(i, padj[i]) for i in ids):
                    ok_h += 1
            label = {"1_T1": "1 (T = 1, recall·precision·best_of_N)", "2_prec_matched": "2 (정밀도 맞춤 recall)",
                     "3_div_matched": "3 (다양성 맞춤)"}[an]
            w(f"| {label} | {v} | {ok_p}/8 ({', '.join(which) or '-'}) | {ok_h}/8 | {na} |")
    w("")

    # CSVs
    with open(D / "tradeoff_curves_per_seed.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh); cw.writerow(["corpus", "scale", "seed", "curve", "T"] + MET)
        for (c, s), seeds in R.items():
            for sd, r in seeds.items():
                for cv, pts in r["curves"].items():
                    for t in temps:
                        cw.writerow([c, s, sd, cv, t] + [pts[str(t)][m] for m in MET])
    with open(D / "tradeoff_curves_mean.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh); cw.writerow(["corpus", "scale", "curve", "T", "n_seeds"] + MET)
        for (c, s), seeds in R.items():
            for cv in ("F", "QCm", "QCs"):
                for t in temps:
                    cw.writerow([c, s, cv, t, len(seeds)] + [np.mean([seeds[sd]["curves"][cv][str(t)][m] for sd in seeds]) for m in MET])
    fb = {f"{SHORT[c]} {s}": np.mean([R[(c, s)][sd]["state_cond_fallback_frac"] for sd in R[(c, s)]]) for c, s in cells}
    w("상태 조건부 추출에서 𝒬로 대체된 평가 상태 비율(셀 평균): " + ", ".join(f"{k} {v:.3f}" for k, v in fb.items()) + "\n")
    w(f"그림용 CSV: `{D / 'tradeoff_curves_per_seed.csv'}`, `{D / 'tradeoff_curves_mean.csv'}`\n")

    # mean curve table for the report
    w("## 5. 평균 곡선 (seed 평균)\n")
    w("| 코퍼스 | 규모 | 곡선 | " + " | ".join(f"T={t}" for t in temps) + " |")
    w("|---|---|---|" + "---|" * len(temps))
    for c, s in cells:
        for cv in ("F", "QCm", "QCs"):
            cellv = []
            for t in temps:
                mm = {m: np.mean([R[(c, s)][sd]["curves"][cv][str(t)][m] for sd in R[(c, s)]]) for m in MET}
                dv = f"{mm['diversity']:.4f}" if c == "aloha_tape" else f"{mm['diversity']:.2f}"
                cellv.append(f"P {mm['precision']:.3f} / R {mm['recall']:.3f} / D {dv}")
            w(f"| {SHORT[c]} | {s} | {cv} | " + " | ".join(cellv) + " |")
    (D / "tradeoff_aggregate.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {D / 'tradeoff_aggregate.md'}")


if __name__ == "__main__":
    main()
