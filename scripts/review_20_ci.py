"""Request 20, item 1: QC − comparison mean differences with t 95% CIs (final K64 version; no recomputation of metrics).

Input: the same per-seed values as report 15b (loaded through review_14_aggregate.load: report-09 results, report-10
filter variants, report-11 common targets, report-14 K64 QC). rq15b/per_seed_15b.csv holds F, QC and the RQ2 rows but
NOT the F variants (temperature-/update-matched, top 75 / 25 %, consistency), so the loader is used for everything.
Per corpus: mean of the 5 paired seed differences, t CI (df = n − 1). Pooled (recall, precision only): mean of the
paired run differences, t CI (df = n − 1). Runs without a value (temperature-matched aloha) are dropped and counted.
Writes outputs/campaign_dedup/rq20/ci_{table2,rq3}.csv and ci.md.

    .venv/Scripts/python.exe scripts/review_20_ci.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

import review_11_aggregate as agg
import review_14_aggregate as a14
from review_11_aggregate import CORP, SEEDS, SHORT, MET, FR

OUT = Path("outputs/campaign_dedup/rq20"); OUT.mkdir(parents=True, exist_ok=True)
V = agg.V
SCALES = [1.0, 0.5, 0.25, 0.125]


def ci(d):
    d = np.asarray(d, float); n = len(d)
    if n < 2:
        return (float(d.mean()) if n else np.nan), np.nan, np.nan, n
    hw = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n)
    return float(d.mean()), float(d.mean() - hw), float(d.mean() + hw), n


def scale(c, m):
    return (1e3 if m == "diversity" else 1e4) if (c == "aloha_tape" and m in ("diversity", "best_of_N")) else 1.0


def fmt(c, m, t):
    mu, lo, hi, n = t
    if not np.isfinite(mu):
        return "—"
    s = scale(c, m); mu, lo, hi = mu * s, lo * s, hi * s
    if m in ("recall", "precision"):
        d = 3
    else:   # table-2 style: 3 significant digits of the mean magnitude
        d = max(0, 2 - int(math.floor(math.log10(abs(mu))))) if mu != 0 else 3
    return f"{mu:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]" + (f" (n={n})" if n != 5 else "")


def diffs(tab, c, x, y, m):
    return [V[tab][c][sd][x][m] - V[tab][c][sd][y][m] for sd in SEEDS
            if x in V[tab][c][sd] and y in V[tab][c][sd] and m in V[tab][c][sd][y]]


def main():
    a14.load()
    L = []; w = L.append; rows = []
    w("# 요청 20-1 — QC − 비교 구성 평균 차이와 t 95% CI (최종 K64 판본)\n")
    w("칸 = 평균 [하한, 상한]. 코퍼스별 df = n − 1(보통 4), 통합 = 20 run(온도 맞춤은 15 run) df = n − 1. "
      "aloha diversity ×10⁻³, best-of-N ×10⁻⁴(원 기록 단위; 보고 17). 온도 맞춤의 diversity는 정의상 0(맞춤). diversity·best-of-N은 단위가 코퍼스마다 달라 통합하지 않는다.\n")
    comps = [("F", "A"), ("F 온도 맞춤", "A_Tmatch64"), ("F 학습량 맞춤", "A_step"), ("F activity 상위 75%", "A25"),
             ("F activity 상위 25%", "A75"), ("F consistency 50%", "A_con")]
    w("## (a) 전체 데이터 — 표 2\n")
    for m in MET:
        w(f"**{m}**\n"); hdr = "| QC − | " + " | ".join(SHORT[c] for c in CORP) + (" | 통합 |" if m in ("recall", "precision") else " |")
        w(hdr); w("|---|" + "---|" * (len(CORP) + (1 if m in ("recall", "precision") else 0)))
        for lab, arm in comps:
            cells = []; pooled = []
            for c in CORP:
                d = diffs(("R", 1.0), c, "QC64", arm, m); t = ci(d); pooled += d
                cells.append("(맞춤)" if (arm == "A_Tmatch64" and m == "diversity" and t[3]) else fmt(c, m, t))
                rows.append(["a_full", lab, c, m, *t])
            if m in ("recall", "precision"):
                t = ci(pooled); cells.append(fmt("x", m, t)); rows.append(["a_full", lab, "pooled", m, *t])
            w(f"| {lab} | " + " | ".join(cells) + " |")
        w("")
    w("## (b) 데이터 비율 — 표 5, 부록 표 A6 (QC는 K 비례 64/32/16/8)\n")
    for m in MET:
        w(f"**{m}**\n"); w("| 비율 | " + " | ".join(SHORT[c] for c in CORP) + (" | 통합 |" if m in ("recall", "precision") else " |"))
        w("|---|" + "---|" * (len(CORP) + (1 if m in ("recall", "precision") else 0)))
        for sc in SCALES:
            cells = []; pooled = []
            for c in CORP:
                d = diffs(("RQ3", sc), c, "QC64", "A", m); t = ci(d); cells.append(fmt(c, m, t)); pooled += d
                rows.append(["b_rq3", FR[sc], c, m, *t])
            if m in ("recall", "precision"):
                t = ci(pooled); cells.append(fmt("x", m, t)); rows.append(["b_rq3", FR[sc], "pooled", m, *t])
            w(f"| {FR[sc]} | " + " | ".join(cells) + " |")
        w("")
    # ---- checks against the manuscript numbers
    rec = {c: np.mean(diffs(("R", 1.0), c, "QC64", "A", "recall")) for c in CORP}
    lo_c = min(rec, key=rec.get); hi_c = max(rec, key=rec.get)
    m1 = np.mean([np.mean(diffs(("RQ3", 1.0), c, "QC64", "A", "recall")) for c in CORP])
    m8 = np.mean([np.mean(diffs(("RQ3", 0.125), c, "QC64", "A", "recall")) for c in CORP])
    per, pooled, _ = a14.trend("QC64", "recall")
    w("## 원고 숫자 대조\n")
    w(f"- 5.1 'F 대비 recall 차이 0.037–0.085': 재계산 코퍼스별 " + ", ".join(f"{SHORT[c]} {v:.4f}" for c, v in rec.items())
      + f" → 범위 **{rec[lo_c]:.3f}–{rec[hi_c]:.3f}** ({SHORT[lo_c]}–{SHORT[hi_c]}). "
      + ("일치" if f"{rec[lo_c]:.3f}" == "0.037" and f"{rec[hi_c]:.3f}" == "0.085" else "**불일치: 하한이 다르다**"))
    w(f"- 5.3 '평균 차이 0.060 → 0.100': 재계산 {m1:.4f} → {m8:.4f} → {'일치' if (f'{m1:.3f}', f'{m8:.3f}') == ('0.060', '0.100') else '불일치'}")
    w(f"- 5.3 '절반마다 0.014': 통합 기울기 {pooled[0]:.4f} → {'일치' if f'{pooled[0]:.3f}' == '0.014' else '불일치'}\n")
    (OUT / "ci.md").write_text("\n".join(L), encoding="utf-8")
    with open(OUT / "ci_all.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh); cw.writerow(["part", "comparison_or_fraction", "corpus", "metric", "mean", "ci_low", "ci_high", "n_runs"]); cw.writerows(rows)
    print("\n".join(L[-4:]))


if __name__ == "__main__":
    main()
