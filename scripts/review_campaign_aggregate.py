"""Aggregate the review campaign (outputs/campaign/results/*.json) into paired statistics and CSVs.

Paired statistics per (corpus, scale) over seeds: mean difference, t-distribution 95% CI half-width (df = n-1),
two-sided one-sample t-test p, and k/n = number of seeds where the first-named configuration is better.
Writes outputs/campaign/aggregate.md (+ CSV files for curves and per-seed values). Works on partial results.

    .venv/Scripts/python.exe scripts/review_campaign_aggregate.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path("outputs/campaign")
CORP = ["being_so101", "rayenghali", "koch_pickplace5", "aloha_tape"]
SHORT = {"being_so101": "being", "rayenghali": "rayen", "koch_pickplace5": "koch", "aloha_tape": "aloha"}
SCALES = [1.0, 0.25]
LOWER = {"best_of_N", "wass"}
RATIO = {"motion", "accel", "jerk3", "rough"}
MAIN4 = ["recall", "precision", "diversity", "best_of_N"]


def load():
    R = defaultdict(dict)
    for f in sorted((OUT / "results").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        R[(r["corpus"], r["scale"])][r["seed"]] = r
    return R


def paired(x, y, metric):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y; n = len(d)
    if n < 2:
        return dict(mean=float(d.mean()) if n else np.nan, hw=np.nan, p=np.nan, k=0, n=n)
    hw = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    p = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else (0.0 if d.mean() != 0 else 1.0)
    if metric in RATIO:
        better = np.abs(x - 1) < np.abs(y - 1)
    elif metric in LOWER:
        better = d < 0
    else:
        better = d > 0
    return dict(mean=float(d.mean()), hw=float(hw), p=float(p), k=int(better.sum()), n=n)


def fmt(s, prec=3):
    if np.isnan(s["mean"]):
        return "—"
    star = "\\*" if s["p"] < 0.05 else ""
    return f"{s['mean']:+.{prec}f} ± {s['hw']:.{prec}f}{star} (p={s['p']:.3f}, {s['k']}/{s['n']})"


def holm(ps):
    ps = np.nan_to_num(np.asarray(ps, float), nan=1.0); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def get(cell, path):
    v = cell
    for p in path:
        v = v[p]
    return v


def series(R, c, s, path):
    seeds = sorted(R[(c, s)])
    return [get(R[(c, s)][sd], path) for sd in seeds]


def main():
    R = load()
    L = []
    w = L.append
    ncells = sum(len(v) for v in R.values())
    w(f"# 캠페인 집계 (셀 {ncells}/40)\n")
    w("대응 차이 = 평균 ± t 95% CI 반폭 (p, 앞선 seed 수/전체). \\* = p < 0.05. best_of_N·wass는 낮을수록 좋다.\n")

    # ---------------- main table (reproduction of Table 6) + robustness arms
    w("## 1. 각 구성의 지표 (seed 평균)\n")
    arms = ["A", "A_step", "C", "B1", "B2", "OutFilt", "A_z", "B2_z"]
    w("| 코퍼스 | 규모 | 구성 | n | recall | precision | diversity | best_of_N | wass | inside | accel | jerk3 | rough |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in CORP:
        for s in SCALES:
            if (c, s) not in R:
                continue
            for a in arms:
                try:
                    vals = {m: np.mean(series(R, c, s, ["arms", a, "main", m])) for m in
                            ["recall", "precision", "diversity", "best_of_N", "wass", "inside", "accel", "jerk3", "rough"]}
                except KeyError:
                    continue
                n = len(R[(c, s)])
                w(f"| {SHORT[c]} | {s} | {a} | {n} | " + " | ".join(
                    f"{vals[m]:.4f}" if m in ("diversity", "best_of_N", "wass") and c == "aloha_tape" else f"{vals[m]:.3f}"
                    for m in vals) + " |")
    w("")

    # ---------------- paired contrasts
    def contrast_block(title, pairs, metrics=MAIN4):
        w(f"## {title}\n")
        w("| 코퍼스 | 규모 | 대조 | " + " | ".join(metrics) + " |")
        w("|---|---|---|" + "---|" * len(metrics))
        pv = []
        for x, y in pairs:
            for c in CORP:
                for s in SCALES:
                    if (c, s) not in R:
                        continue
                    try:
                        cells = []
                        for m in metrics:
                            st = paired(series(R, c, s, ["arms", x, "main", m]), series(R, c, s, ["arms", y, "main", m]), m)
                            cells.append(fmt(st, 4 if c == "aloha_tape" and m in ("diversity", "best_of_N") else 3))
                            pv.append((x, y, c, s, m, st["p"]))
                        w(f"| {SHORT[c]} | {s} | {x} − {y} | " + " | ".join(cells) + " |")
                    except KeyError:
                        pass
        w("")
        return pv

    pv_main = contrast_block("2. 주 비교 재현: Quality-Conditioned − Filtered", [("B2", "A")])
    pv_a1 = contrast_block("3. A1 업데이트 맞춤 Filtered", [("B2", "A_step"), ("A_step", "A")])
    pv_a2 = contrast_block("4. A2 출력 기각 (Output-Filtered)", [("B2", "OutFilt"), ("OutFilt", "A")])
    pv_dec = contrast_block("5. 분해 (그림 3)", [("C", "A"), ("B2", "C")], ["recall", "diversity"])
    contrast_block("6. 1축 절제", [("B2", "B1")], ["recall", "precision", "diversity"])
    pv_z = contrast_block("7. 견고성: 관절별 z-score 축 (Quality-Conditioned_z − Filtered_z, z 정의의 평가 타깃)", [("B2_z", "A_z")])

    # A1 convergence
    w("## 8. A1 수렴: 마지막 10 epoch 검증 손실의 상대 변화 (seed 평균)\n")
    w("| 코퍼스 | 규모 | " + " | ".join(["A", "A_step", "C", "B1", "B2", "A_z", "B2_z"]) + " |")
    w("|---|---|" + "---|" * 7)
    for c in CORP:
        for s in SCALES:
            if (c, s) not in R:
                continue
            row = []
            for a in ["A", "A_step", "C", "B1", "B2", "A_z", "B2_z"]:
                try:
                    v = series(R, c, s, ["arms", a, "train", "val_change_last10_rel"])
                    fv = series(R, c, s, ["arms", a, "train", "final_val_loss"])
                    ep = series(R, c, s, ["arms", a, "train", "epochs"])
                    row.append(f"{np.mean(v) * 100:+.2f}% (val {np.mean(fv):.4f}, {int(np.mean(ep))} ep)")
                except KeyError:
                    row.append("—")
            w(f"| {SHORT[c]} | {s} | " + " | ".join(row) + " |")
    w("")

    # A2 acceptance
    w("## 9. A2 수용률\n")
    w("| 코퍼스 | 규모 | P/R 수용률 | 수용 실패 상태 비율 | diversity 수용률 | 8개 미달 상태 비율 |")
    w("|---|---|---|---|---|---|")
    for c in CORP:
        for s in SCALES:
            if (c, s) not in R:
                continue
            try:
                g = lambda k: np.mean(series(R, c, s, ["arms", "OutFilt", "main", k]))
                w(f"| {SHORT[c]} | {s} | {g('accept_rate_pr'):.3f} | {g('frac_states_no_accept'):.4f} | "
                  f"{g('accept_rate_div'):.3f} | {g('frac_div_states_short'):.4f} |")
            except KeyError:
                pass
    w("")

    # A3 temperature sweep
    w("## 10. A3 온도 스윕 (Filtered, seed 평균) 과 diversity 맞춤 비교\n")
    temps = None
    for c in CORP:
        for s in SCALES:
            if (c, s) not in R:
                continue
            seeds = sorted(R[(c, s)])
            temps = list(R[(c, s)][seeds[0]]["temperature_sweep_A"].keys())
            w(f"**{SHORT[c]} {s}** — Quality-Conditioned diversity {np.mean(series(R, c, s, ['arms', 'B2', 'main', 'diversity'])):.4f}")
            w("| T | recall | precision | diversity | best_of_N |")
            w("|---|---|---|---|---|")
            divs = {}
            for T in temps:
                m = {k: np.mean(series(R, c, s, ["temperature_sweep_A", T, k])) for k in MAIN4}
                divs[T] = m["diversity"]
                w(f"| {T} | {m['recall']:.3f} | {m['precision']:.3f} | {m['diversity']:.4f} | {m['best_of_N']:.4f} |")
            target = np.mean(series(R, c, s, ["arms", "B2", "main", "diversity"]))
            Tstar = min(divs, key=lambda T: abs(divs[T] - target))
            cells = []
            for m in ["recall", "precision", "best_of_N", "diversity"]:
                st = paired(series(R, c, s, ["arms", "B2", "main", m]), series(R, c, s, ["temperature_sweep_A", Tstar, m]), m)
                cells.append(f"{m} {fmt(st, 4)}")
            w(f"\ndiversity가 가장 가까운 T = {Tstar}. Quality-Conditioned − Filtered@T*: " + "; ".join(cells) + "\n")

    # A3 curves CSV
    with open(OUT / "pr_curves_per_seed.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh)
        cw.writerow(["corpus", "scale", "seed", "config", "rho_or_T", "precision", "recall", "diversity"])
        for (c, s), seeds in R.items():
            for sd, r in seeds.items():
                for arm, pts in r["pr_curves"].items():
                    for rho, v in pts.items():
                        cw.writerow([c, s, sd, arm, f"rho={rho}", v["precision"], v["recall"], v["diversity"]])
                for T, v in r["temperature_sweep_A"].items():
                    cw.writerow([c, s, sd, "A_temperature", f"T={T}", v["precision"], v["recall"], v["diversity"]])
    w(f"precision–recall 곡선 데이터(seed별): `{OUT / 'pr_curves_per_seed.csv'}`\n")
    w("### 곡선 평균 (사후 필터 ρ)\n")
    w("| 코퍼스 | 규모 | 구성 | ρ | precision | recall | diversity |")
    w("|---|---|---|---|---|---|---|")
    for c in CORP:
        for s in SCALES:
            if (c, s) not in R:
                continue
            first = R[(c, s)][sorted(R[(c, s)])[0]]
            for arm in first["pr_curves"]:
                for rho in first["pr_curves"][arm]:
                    m = {k: np.mean(series(R, c, s, ["pr_curves", arm, rho, k])) for k in ("precision", "recall", "diversity")}
                    w(f"| {SHORT[c]} | {s} | {arm} | {rho} | {m['precision']:.3f} | {m['recall']:.3f} | {m['diversity']:.4f} |")
    w("")

    # A4 q modes
    w("## 11. A4 q 샘플링 방식 (기본과의 대응 차이)\n")
    for arm in ["B2", "B1"]:
        w(f"**{arm}**\n")
        w("| 코퍼스 | 규모 | 방식 | recall | precision | diversity | best_of_N |")
        w("|---|---|---|---|---|---|---|")
        for c in CORP:
            for s in SCALES:
                if (c, s) not in R:
                    continue
                try:
                    fb = np.mean(series(R, c, s, ["q_modes", arm, "state_conditional_fallback_frac"]))
                    for mode in ["per_state_fixed", "global_median", "state_conditional"]:
                        cells = [fmt(paired(series(R, c, s, ["q_modes", arm, mode, m]),
                                            series(R, c, s, ["q_modes", arm, "default", m]), m),
                                     4 if c == "aloha_tape" and m in ("diversity", "best_of_N") else 3) for m in MAIN4]
                        w(f"| {SHORT[c]} | {s} | {mode} − default | " + " | ".join(cells) + " |")
                    w(f"| {SHORT[c]} | {s} | (상태 조건부에서 𝒬로 대체된 상태 비율 {fb:.3f}) | | | | |")
                except KeyError:
                    pass
        w("")

    # A6 RQ2
    w("## 12. A6 RQ2 (전체 데이터, Quality-Conditioned)\n")
    w("| 코퍼스 | 축 | 스윕 | 제어 상관 (평균, 최소) | 누수 | in-range 최소 | 데이터 고유 누수 기준값 |")
    w("|---|---|---|---|---|---|---|")
    for c in CORP:
        if (c, 1.0) not in R:
            continue
        for ax in ["activity", "smoothness"]:
            base = series(R, c, 1.0, ["rq2", ax, "data_baseline", "leak"])
            for sw in ["basic", "conditional"]:
                rho = series(R, c, 1.0, ["rq2", ax, sw, "control_spearman"])
                lk = np.array(series(R, c, 1.0, ["rq2", ax, sw, "leak"]))
                ins = series(R, c, 1.0, ["rq2", ax, sw, "inrange_min"])
                hw = stats.t.ppf(0.975, len(lk) - 1) * lk.std(ddof=1) / np.sqrt(len(lk)) if len(lk) > 1 else np.nan
                w(f"| {SHORT[c]} | {ax} | {sw} | {np.mean(rho):.3f}, {np.min(rho):.3f} | {lk.mean():.3f} ± {hw:.3f} | "
                  f"{np.min(ins):.3f} | {np.mean(base):.3f} |")
    w("")

    # B4 Holm on campaign p-values
    w("## 13. B4 Holm 보정 (캠페인 p값)\n")
    for name, pv, mets in (("주 비교 B2 − A (recall, diversity)", pv_main, ("recall", "diversity")),
                           ("분해 C − A, B2 − C (recall, diversity)", pv_dec, ("recall", "diversity"))):
        sel = [p for p in pv if p[4] in mets]
        if not sel:
            continue
        adj = holm([p[-1] for p in sel])
        w(f"- {name}: 보정 전 유의 {sum(p[-1] < .05 for p in sel)}/{len(sel)} → Holm {int((adj < .05).sum())}/{len(sel)}")
        for grp in sorted(set((p[0], p[1], p[4]) for p in sel)):
            idx = [i for i, p in enumerate(sel) if (p[0], p[1], p[4]) == grp]
            w(f"  - {grp[0]} − {grp[1]} {grp[2]}: {sum(sel[i][-1] < .05 for i in idx)}/{len(idx)} → {int(sum(adj[i] < .05 for i in idx))}/{len(idx)}")
    w("")

    # B1 training time
    w("## 14. B1 학습 시간 (초, seed 평균 [최소–최대])\n")
    w("| 코퍼스 | 규모 | " + " | ".join(["A", "A_step", "C", "B1", "B2", "A_z", "B2_z"]) + " |")
    w("|---|---|" + "---|" * 7)
    for c in CORP:
        for s in SCALES:
            if (c, s) not in R:
                continue
            row = []
            for a in ["A", "A_step", "C", "B1", "B2", "A_z", "B2_z"]:
                try:
                    v = [x for x in series(R, c, s, ["arms", a, "train", "seconds"]) if x is not None]
                    row.append(f"{np.mean(v):.0f} [{min(v):.0f}–{max(v):.0f}]" if v else "—")
                except KeyError:
                    row.append("—")
            w(f"| {SHORT[c]} | {s} | " + " | ".join(row) + " |")
    w("")

    (OUT / "aggregate.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT / 'aggregate.md'} ({ncells} cells)")


if __name__ == "__main__":
    main()
