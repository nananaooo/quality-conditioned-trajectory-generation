"""Report 14 aggregation: K32 (earlier reports) vs K64 (rq14) side by side, the pre-fixed adoption rule (request 14,
section 3), Table 2 in two versions, RQ3 trend + figures, trade-off figure with the K64 QC curve.
Reads earlier outputs only; writes ONLY under outputs/campaign_dedup/rq14/.

    .venv/Scripts/python.exe scripts/review_14_aggregate.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import review_11_aggregate as agg
import review_12_aggregate as a12
from review_11_aggregate import CORP, SEEDS, SHORT, MET, FR, cells, fmt, runlevel, run_pairs, fp, oriented, tmatch, J

RQ = Path("outputs/campaign_dedup/rq14"); D9 = Path("outputs/campaign_dedup")
V = agg.V
SCALES = [1.0, 0.5, 0.25, 0.125]
TEMPS = None


def load():
    global TEMPS
    a12.load()
    for c in CORP:
        for sd in SEEDS:
            r = J(RQ / "full" / f"{c}_f1.00_s{sd}.json")
            R = V[("R", 1.0)][c][sd]
            R["QC64"] = r["arms"]["QC"]; R["AC64"] = r["arms"]["AC"]; R["QC64_chk"] = r["QC_checked"]
            R["_QC64"] = {"fallback": r["arms"]["QC"]["fallback"], "replay_dev": r["replay_dev_vs_report13"], "seconds": r["seconds"]}
            tr = J(D9 / "tradeoff" / f"{c}_f1.00_s{sd}.json"); TEMPS = tr["temps"]
            tm = tmatch(tr["curves"]["F"], tr["temps"], r["arms"]["QC"]["diversity"])
            if tm: R["A_Tmatch64"] = tm
            R["_curve64"] = r["temperature_curve_QC"]; R["_curves09"] = tr["curves"]
            V[("R2", 1.0)][c][sd]["QC64"] = r["R2"]["arms"]["QC"]; V[("R2", 1.0)][c][sd]["AC64"] = r["R2"]["arms"]["AC"]
            V[("R2", 1.0)][c][sd]["_fb64"] = {"fallback": r["R2"]["fallback"]}
            V[("CON", 1.0)][c][sd]["QC64_act"] = r["con"]["QC_actpool"]; V[("CON", 1.0)][c][sd]["QC64_con"] = r["con"]["QC_conpool"]
            V[("RQ3", 1.0)][c][sd].update({"A": R["A"], "QCs": R["QCs"], "QC64": R["QC64"]})
            for sc in (0.5, 0.25, 0.125):
                s = J(RQ / "scales" / f"{c}_f{agg.TAG[sc]}_s{sd}.json")
                Rs = V[("R", sc)][c][sd]
                V[("RQ3", sc)][c][sd].update({"A": Rs["A"], "QCs": Rs["QCs"], "QC64": s["QC"], "_fb": {"K32": Rs["_meta"]["state_cond_fallback_frac"], "K64": s["fallback"], "K": s["K"]}})


# ----------------------------------------------------------------------------- side-by-side tables
def pair_table(w, tgt, sc, versions, mets, title, note_unreach=False):
    """versions = [(label, x, y), ...]; rows = corpus x version, then one run-level row per version."""
    w(f"**{title}**\n")
    w("| 코퍼스 | 판본 | " + " | ".join(mets) + " |"); w("|---|---|" + "---|" * len(mets))
    cs = {(lab, m): cells(tgt, sc, x, y, m) for lab, x, y in versions for m in mets}
    for i, c in enumerate(CORP):
        for lab, x, y in versions:
            row = []
            for m in mets:
                s = fmt(cs[(lab, m)][i], c, m)
                if note_unreach and 5 - cs[(lab, m)][i]["n"]:
                    s += f" [도달 불가 {5 - cs[(lab, m)][i]['n']}]"
                row.append(s)
            w(f"| {SHORT[c]} | {lab} | " + " | ".join(row) + " |")
    for lab, x, y in versions:
        row = []
        for m in mets:
            rl = runlevel(run_pairs(tgt, sc, x, y, m), m)
            row.append(f"{rl['k']}/{rl['n']}" + (f" (동률 {rl['ties']})" if rl["ties"] else "") + f", 부호 p={fp(rl['sign_p'])}, W p={fp(rl['wil_p'])}")
        w(f"| **전체** | {lab} | " + " | ".join(row) + " |")
    w("")


def superior(x, y, primary, secondary, tgt="R", sc=1.0):
    det = {}
    ok = True
    for m in primary:
        rl = runlevel(run_pairs(tgt, sc, x, y, m), m); det[m] = rl
        ok &= rl["sign_p"] < .05 and rl["k"] > rl["n"] / 2
    for m in secondary:
        rl = runlevel(run_pairs(tgt, sc, x, y, m), m); det[m] = rl
        ok &= not (rl["sign_p"] < .05 and rl["k"] < rl["n"] / 2)
    return ok, det


def trend(arm, m):
    pts = {c: [] for c in CORP}
    for c in CORP:
        for sc in SCALES:
            for sd in SEEDS:
                A = V[("RQ3", sc)][c][sd]
                pts[c].append((sd, np.log2(sc), float(oriented(A[arm][m], A["A"][m], m))))
    per = {}
    for c in CORP:
        x = np.array([p[1] for p in pts[c]]); y = np.array([p[2] for p in pts[c]])
        r = stats.linregress(x, y); per[c] = (-r.slope, stats.t.ppf(.975, len(x) - 2) * r.stderr, r.pvalue)
    X, Y = [], []
    for ci, c in enumerate(CORP):
        for _, x, y in pts[c]:
            X.append([x] + [1.0 if cj == ci else 0.0 for cj in range(len(CORP))]); Y.append(y)
    X, Y = np.array(X), np.array(Y)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta; df = len(Y) - X.shape[1]; se = math.sqrt(res @ res / df * np.linalg.inv(X.T @ X)[0, 0])
    pooled = (-beta[0], stats.t.ppf(.975, df) * se, 2 * stats.t.sf(abs(beta[0] / se), df))
    sl = np.array([-np.polyfit([p[1] for p in pts[c] if p[0] == sd], [p[2] for p in pts[c] if p[0] == sd], 1)[0] for c in CORP for sd in SEEDS])
    clus = (sl.mean(), stats.t.ppf(.975, 19) * sl.std(ddof=1) / np.sqrt(20), stats.ttest_1samp(sl, 0).pvalue, int((sl > 0).sum()))
    return per, pooled, clus


# ----------------------------------------------------------------------------- Table 2
def sig3(mean, sd):
    if mean == 0 or not np.isfinite(mean):
        return f"{mean:.3g} ± {sd:.3g}"
    d = max(0, 2 - int(math.floor(math.log10(abs(mean)))))
    return f"{mean:.{d}f} ± {sd:.{d}f}"


def table2(w, qc, tm):
    rows = [("F (표준)", "A"), ("F 온도 맞춤", tm), ("F 학습량 맞춤", "A_step"), ("F activity 75%", "A25"),
            ("F activity 25%", "A75"), ("F consistency 50%", "A_con"), ("QC", qc)]
    w("| 코퍼스 | 구성 | recall | precision | diversity | best-of-N |"); w("|---|---|---|---|---|---|")
    for c in CORP:
        al = c == "aloha_tape"
        for lab, arm in rows:
            cellv = []
            for m in MET:
                v = np.array([V[("R", 1.0)][c][sd][arm][m] for sd in SEEDS if arm in V[("R", 1.0)][c][sd]], float)
                if len(v) == 0:
                    cellv.append("도달 불가"); continue
                if al and m == "diversity": v = v * 1e3
                if al and m == "best_of_N": v = v * 1e4
                sd_ = v.std(ddof=1) if len(v) > 1 else 0.0
                s = f"{v.mean():.3f} ± {sd_:.3f}" if m in ("recall", "precision") else sig3(v.mean(), sd_)
                if arm == tm and m == "diversity": s += " (맞춤)"
                if len(v) < 5: s += f" (n={len(v)})"
                cellv.append(s)
            w(f"| {SHORT[c]} | {lab} | " + " | ".join(cellv) + " |")
    w("\naloha: diversity ×10⁻³ rad, best-of-N ×10⁻⁴ rad². 평균 ± 표준편차(5 seed). 온도 맞춤은 도달한 seed만(aloha 0).\n")
    w("**QC가 각 F 변형보다 나은 실행 수 (20회 중; 온도 맞춤은 도달 seed만), 양측 부호 검정 p**\n")
    w("| F 변형 | recall | precision | diversity | best-of-N |"); w("|---|---|---|---|---|")
    for lab, arm in rows[:-1]:
        row = []
        for m in MET:
            if arm == tm and m == "diversity":
                row.append("(맞춤)"); continue
            rl = runlevel(run_pairs("R", 1.0, qc, arm, m), m)
            row.append(f"{rl['k']}/{rl['n']}" + (f" (동률 {rl['ties']})" if rl["ties"] else "") + f", p={fp(rl['sign_p'])}")
        w(f"| {lab} | " + " | ".join(row) + " |")
    w("")


# ----------------------------------------------------------------------------- figures
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "legend.frameon": False})


def fig_rq3():
    style(); out = RQ / "figs"; out.mkdir(exist_ok=True)
    ARMS = [("A", "Filtered", "#2a78d6", "o"), ("QC64", "QC (K proportional, 64 at full)", "#1baf7a", "^")]
    for m, ylab in (("recall", "Recall"), ("diversity", "Diversity")):
        rows = []; fig, axs = plt.subplots(1, 4, figsize=(11.0, 2.9))
        for j, c in enumerate(CORP):
            ax = axs[j]
            for arm, lab, col, mk in ARMS:
                mean, hw = [], []
                for sc in SCALES:
                    v = np.array([V[("RQ3", sc)][c][sd][arm][m] for sd in SEEDS])
                    mean.append(v.mean()); hw.append(stats.t.ppf(.975, 4) * v.std(ddof=1) / np.sqrt(5))
                    rows.append([c, sc, lab, 5, v.mean(), hw[-1]] + list(v))
                mean, hw = np.array(mean), np.array(hw)
                ax.plot(SCALES, mean, "-", color=col, lw=2, zorder=2)
                ax.fill_between(SCALES, mean - hw, mean + hw, color=col, alpha=0.15, lw=0, zorder=1)
                ax.errorbar(SCALES, mean, yerr=hw, fmt="none", ecolor=col, elinewidth=0.8, capsize=2, zorder=2)
                ax.scatter(SCALES, mean, s=26, marker=mk, color=col, edgecolors="white", linewidths=0.8, zorder=3, label=lab if j == 0 else None)
            ax.set_xscale("log", base=2); ax.set_xticks(SCALES); ax.set_xticklabels(["1", "1/2", "1/4", "1/8"]); ax.set_xlim(1.25, 0.1)
            if m == "diversity" and c == "aloha_tape": ax.set_yscale("log")
            ax.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            ax.set_title(c if c == "being_so101" else SHORT[c], color=INK, fontsize=9.5, loc="left"); ax.set_xlabel("Training-data fraction")
            if j == 0: ax.set_ylabel(ylab)
            if m == "diversity" and c == "aloha_tape": ax.set_ylabel("Diversity (log, rad)")
        h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out / f"fig_rq3_{m}_K64.png", dpi=300, bbox_inches="tight"); plt.close(fig)
        with open(out / f"fig_rq3_{m}_K64.csv", "w", newline="", encoding="utf-8") as fh:
            cw = csv.writer(fh); cw.writerow(["corpus", "fraction", "arm", "n_seeds", "mean", "ci95_halfwidth"] + [f"seed{s}" for s in SEEDS]); cw.writerows(rows)


def fig_tradeoff():
    """Figure 2 layout (full data) with the QC curve regenerated at K64; F and QC-M curves are the report-09 ones."""
    style(); out = RQ / "figs"; out.mkdir(exist_ok=True)
    CUR = [("F", "Filtered", "#2a78d6", "o"), ("QCm", "QC-M", "#eb6834", "s"), ("QC64", "QC (K64)", "#1baf7a", "^")]
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.4))
    for j, c in enumerate(CORP):
        for row, (xk, yk) in enumerate([("precision", "recall"), ("diversity", "precision")]):
            ax = axes[row, j]
            for cv, lab, col, mk in CUR:
                M = {}
                for T in TEMPS:
                    vals = [(V[("R", 1.0)][c][sd]["_curve64"] if cv == "QC64" else V[("R", 1.0)][c][sd]["_curves09"][cv])[str(T)] for sd in SEEDS]
                    M[T] = {k: (np.mean([v[k] for v in vals]), np.std([v[k] for v in vals], ddof=1) / np.sqrt(5)) for k in ("precision", "recall", "diversity")}
                    if row == 0:
                        rows.append([c, cv, T] + [M[T][k][0] for k in ("precision", "recall", "diversity")] + [M[T][k][1] for k in ("precision", "recall", "diversity")])
                xs = [M[T][xk][0] for T in TEMPS]; ys = [M[T][yk][0] for T in TEMPS]
                ax.plot(xs, ys, "-", color=col, lw=2, zorder=2)
                ax.errorbar(xs, ys, xerr=[M[T][xk][1] for T in TEMPS], yerr=[M[T][yk][1] for T in TEMPS], fmt="none", ecolor=col, elinewidth=0.8, alpha=0.7, zorder=2)
                ax.scatter(xs, ys, s=22, marker=mk, color=col, edgecolors="white", linewidths=0.8, zorder=3, label=lab if (row == 0 and j == 0) else None)
                ax.scatter([M[1.0][xk][0]], [M[1.0][yk][0]], s=85, marker=mk, color=col, edgecolors=INK, linewidths=1.1, zorder=4)
            if c == "aloha_tape" and xk == "diversity": ax.set_xscale("log")
            ax.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            ax.set_xlabel(xk.capitalize() + (" (log, rad)" if c == "aloha_tape" and xk == "diversity" else ""))
            if j == 0: ax.set_ylabel(yk.capitalize())
            if row == 0: ax.set_title(c if c == "being_so101" else SHORT[c], color=INK, fontsize=9.5, loc="left")
    h, l = axes[0, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h.append(Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, markersize=8, markeredgewidth=1.1)); l.append("T = 1")
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / "fig2_tradeoff_full_K64.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig2_tradeoff_full_K64.pdf", bbox_inches="tight"); plt.close(fig)
    with open(out / "fig2_tradeoff_full_K64.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh); cw.writerow(["corpus", "curve", "T", "precision", "recall", "diversity", "se_precision", "se_recall", "se_diversity"]); cw.writerows(rows)


# ----------------------------------------------------------------------------- report
def main():
    load()
    L = []; w = L.append
    # adoption rule
    comps = [("A", "F (표준)", ["recall", "diversity"]), ("A_Tmatch64", "F 온도 맞춤", ["recall"]), ("A_step", "F 학습량 맞춤", ["recall", "diversity"]),
             ("A25", "F activity 75%", ["recall", "diversity"]), ("A75", "F activity 25%", ["recall", "diversity"]), ("A_con", "F consistency 50%", ["recall", "diversity"])]
    r1, det1 = True, []
    for y, lab, prim in comps:
        ok, d = superior("QC64", y, prim, ["precision", "best_of_N"])
        r1 &= ok; det1.append((lab, ok, d))
    per, pooled, clus = trend("QC64", "recall")
    r2 = pooled[2] < .05 and pooled[0] > 0
    rl3 = runlevel(run_pairs("R", 1.0, "QC64", "QCs", "stop_G"), "stop_G")
    r3 = not (rl3["sign_p"] < .05 and rl3["k"] < rl3["n"] / 2)
    adopt = r1 and r2 and r3
    w("## 채택 판정 (요청 14 3절, 결과 전 고정 규칙)\n")
    w(f"**판정: {'K64 채택' if adopt else 'K32 유지 (K64는 민감도 분석)'}**\n")
    w(f"1. RQ1 우월 판정 (6개 비교 모두): **{'충족' if r1 else '미충족'}**\n")
    w("| 비교 | 판정 | recall | diversity | precision | best-of-N |"); w("|---|---|---|---|---|---|")
    for lab, ok, d in det1:
        w(f"| QC(K64) − {lab} | {'우월' if ok else '**우월 아님**'} | " + " | ".join(
            (f"{d[m]['k']}/{d[m]['n']}, p={fp(d[m]['sign_p'])}" if m in d else "(맞춤)") for m in ("recall", "diversity", "precision", "best_of_N")) + " |")
    w(f"\n2. RQ3 recall 통합 기울기(절반마다 Δ 변화): {pooled[0]:+.4f} ± {pooled[1]:.4f}, p={fp(pooled[2])} → **{'충족' if r2 else '미충족'}**")
    w(f"3. 정지 수준 QC(K64) − QC(K32): K64가 적은 실행 {rl3['k']}/{rl3['n']} (동률 {rl3['ties']}), 부호 p={fp(rl3['sign_p'])} → **{'충족' if r3 else '미충족'}**\n")
    # reference: same rule on K32
    rr = []
    for y, lab, prim in comps:
        yy = "A_Tmatch" if y == "A_Tmatch64" else y
        ok, d = superior("QCs", yy, prim, ["precision", "best_of_N"]); rr.append(f"{lab} {'우월' if ok else '우월 아님'}")
    p32 = trend("QCs", "recall")[1]
    w(f"참고(판정에 쓰지 않음) — 같은 규칙을 K32에 적용: RQ1 " + ", ".join(rr) + f"; RQ3 기울기 {p32[0]:+.4f} (p={fp(p32[2])}).\n")

    w("## a. RQ1 (전체 데이터, 타깃 R)\n")
    for y, lab in (("A", "F (T = 1)"), ("A_step", "F 학습량 맞춤"), ("A25", "F activity 75%"), ("A75", "F activity 25%"), ("A_con", "F consistency 50%")):
        pair_table(w, "R", 1.0, [("K32", "QCs", y), ("K64", "QC64", y)], MET, f"QC − {lab}")
    pair_table(w, "R", 1.0, [("K32", "QCs", "A_Tmatch"), ("K64", "QC64", "A_Tmatch64")], ["recall", "precision", "best_of_N"],
               "QC − F 온도 맞춤 (각 판본의 QC T = 1 diversity에 맞춤)", note_unreach=True)
    fmx = {c: np.mean([max(p["diversity"] for p in V[("R", 1.0)][c][sd]["_curves09"]["F"].values()) for sd in SEEDS]) for c in CORP}
    w("F 곡선 최대 diversity / QC 목표 (K32, K64): " + ", ".join(
        f"{SHORT[c]} {fmx[c]:.4f} / {np.mean([V[('R', 1.0)][c][sd]['QCs']['diversity'] for sd in SEEDS]):.4f}, {np.mean([V[('R', 1.0)][c][sd]['QC64']['diversity'] for sd in SEEDS]):.4f}" for c in CORP) + "\n")
    w(f"절충 곡선 그림(K64 QC 곡선): `{RQ / 'figs' / 'fig2_tradeoff_full_K64.png'}` (+ .pdf, .csv). F·QC-M 곡선은 보고 09 값.\n")

    w("## b. RQ2 학습 측\n")
    pair_table(w, "R", 1.0, [("K32", "QCs", "C"), ("K64", "QC64", "C")], MET, "QC − FU")
    pair_table(w, "R", 1.0, [("K32", "QCs", "OutFilt"), ("K64", "QC64", "OutFilt")], MET, "QC − OF")
    pair_table(w, "R", 1.0, [("K32", "ACs", "A"), ("K64", "AC64", "A")], MET, "AC − F")
    pair_table(w, "R", 1.0, [("K32", "QCs", "ACs"), ("K64", "QC64", "AC64")], MET, "QC − AC (같은 K)")

    w("## c. RQ2 생성 측 (각 방식 − QC)\n")
    for mode, lab in (("mode:global_median", "전역 고정"), ("B2", "상태 무관 추출 (QC-M)"), ("mode:per_state_fixed", "상태별 고정")):
        pair_table(w, "R", 1.0, [("K32", mode, "QCs"), ("K64", mode, "QC64")], MET, f"{lab} − QC ((k/n) = 이 방식이 나은 seed 수)")

    w("## d. RQ3 (공통 타깃 R; K = 64, 32, 16, 8)\n")
    w("| 코퍼스 | " + " | ".join(f"fallback {FR[s]} K32 / K{V[('RQ3', s)][CORP[0]][0].get('_fb', {}).get('K', 64)}" for s in SCALES[1:]) + " |")
    w("|---|" + "---|" * 3)
    for c in CORP:
        w(f"| {SHORT[c]} | " + " | ".join(f"{np.mean([V[('RQ3', s)][c][sd]['_fb']['K32'] for sd in SEEDS]):.3f} / {np.mean([V[('RQ3', s)][c][sd]['_fb']['K64'] for sd in SEEDS]):.3f}" for s in SCALES[1:]) + " |")
    w(f"\n전체 데이터 fallback K32 / K64: " + ", ".join(f"{SHORT[c]} {np.mean([V[('R', 1.0)][c][sd]['_meta']['fallback'] for sd in SEEDS]):.3f} / {np.mean([V[('R', 1.0)][c][sd]['_QC64']['fallback'] for sd in SEEDS]):.3f}" for c in CORP) + "\n")
    mets3 = MET + ["lowq_G"]
    for sc in SCALES:
        pair_table(w, "RQ3", sc, [("기존 K32", "QCs", "A"), (f"비례 K={ {1.0: 64, 0.5: 32, 0.25: 16, 0.125: 8}[sc] }", "QC64", "A")], mets3, f"데이터 비율 {FR[sc]}: QC − F (\\* p<0.05, † Holm 4코퍼스)")
    w("**추세 (절반마다 Δ 변화 = −기울기; 방향은 11b와 같음)**\n")
    w("| 지표 | 판본 | " + " | ".join(SHORT[c] for c in CORP) + " | 통합(고정효과) | (코퍼스, seed)별 기울기, 양수 수/20 |")
    w("|---|---|" + "---|" * (len(CORP) + 2))
    for m in mets3:
        for lab, arm in (("K32", "QCs"), ("K64", "QC64")):
            per, pooled, clus = trend(arm, m)
            w(f"| {m} | {lab} | " + " | ".join(f"{per[c][0]:+.4f} ± {per[c][1]:.4f}{'*' if per[c][2] < .05 else ''}" for c in CORP)
              + f" | {pooled[0]:+.4f} ± {pooled[1]:.4f} (p={fp(pooled[2])}) | {clus[0]:+.4f} ± {clus[1]:.4f} (p={fp(clus[2])}), {clus[3]}/20 |")
    w(f"\n그림 3(K64): `{RQ / 'figs' / 'fig_rq3_recall_K64.png'}`, `{RQ / 'figs' / 'fig_rq3_diversity_K64.png'}` (+ .csv)\n")

    w("## e. 결함 생성 비율\n")
    w("**정지 수준 / 경계 (타깃 R, G)**\n")
    w("| 코퍼스 | F | FU | AC K32 | AC K64 | QC K32 | QC K64 |"); w("|---|---|---|---|---|---|---|")
    for c in CORP:
        w(f"| {SHORT[c]} | " + " | ".join(
            f"{np.mean([V[('R', 1.0)][c][sd][a]['stop_G'] for sd in SEEDS])*100:.2f}% / {np.mean([V[('R', 1.0)][c][sd][a]['boundary_G'] for sd in SEEDS])*100:.1f}%"
            for a in ("A", "C", "ACs", "AC64", "QCs", "QC64")) + " |")
    w("")
    pair_table(w, "R", 1.0, [("K64 − K32", "QC64", "QCs")], ["stop_G", "boundary_G", "lowq_G"], "QC(K64) − QC(K32): 정지·경계·τ 아래 (음수 = K64가 적음)")
    pair_table(w, "R", 1.0, [("K32", "QCs", "A"), ("K64", "QC64", "A")], ["stop_G", "boundary_G", "lowq_G"], "QC − F: 정지·경계·τ 아래")
    w("**떨림 (보고 12 B1, 타깃 R2)**\n")
    pair_table(w, "R2", 1.0, [("K32", "QCs", "ACs"), ("K64", "QC64", "AC64")], MET + ["tremor_G"], "QC − AC (R2)")
    pair_table(w, "R2", 1.0, [("K32", "QCs", "A"), ("K64", "QC64", "A")], MET + ["tremor_G"], "QC − F (R2)")
    for lab, x, y in (("K32", "QCs", "ACs"), ("K64", "QC64", "AC64")):
        pro, con_ = [], []
        for m in ("recall", "precision", "diversity", "best_of_N", "tremor_G"):
            rl = runlevel(run_pairs("R2", 1.0, x, y, m), m)
            if rl["sign_p"] < .05 and rl["k"] > rl["n"] / 2 and m in ("recall", "precision", "tremor_G"): pro.append(f"{m} {rl['k']}/{rl['n']}")
            if rl["sign_p"] < .05 and rl["k"] < rl["n"] / 2: con_.append(f"{m} {rl['k']}/{rl['n']}")
        w(f"- 보고 12 판정 규칙, {lab}: **{'smoothness 기여 있음' if pro and not con_ else '기여 확인 안 됨'}** (QC 쪽 유의: {', '.join(pro) or '없음'}; 반대: {', '.join(con_) or '없음'})")
    w("\n떨림 비율 (G) 평균 K32 → K64: " + "; ".join(
        f"{SHORT[c]} QC {np.mean([V[('R2', 1.0)][c][sd]['QCs']['tremor_G'] for sd in SEEDS])*100:.1f}→{np.mean([V[('R2', 1.0)][c][sd]['QC64']['tremor_G'] for sd in SEEDS])*100:.1f}%, "
        f"AC {np.mean([V[('R2', 1.0)][c][sd]['ACs']['tremor_G'] for sd in SEEDS])*100:.1f}→{np.mean([V[('R2', 1.0)][c][sd]['AC64']['tremor_G'] for sd in SEEDS])*100:.1f}%" for c in CORP) + "\n")

    w("## f. consistency 타깃 (보고 12 B2 설정)\n")
    for y, lab in (("A_con", "consistency 50% 필터"), ("A25", "activity 75% 필터"), ("A50", "F")):
        pair_table(w, "CON", 1.0, [("activity 풀 K32", "QCs", y), ("activity 풀 K64", "QC64_act", y),
                                  ("consistency 풀 K32", "QCs_con", y), ("consistency 풀 K64", "QC64_con", y)], MET, f"QC − {lab}")
    w("fallback (activity 풀 / consistency 풀, K64): " + ", ".join(
        f"{SHORT[c]} {np.mean([V[('CON', 1.0)][c][sd]['QC64_act']['fallback'] for sd in SEEDS]):.3f} / {np.mean([V[('CON', 1.0)][c][sd]['QC64_con']['fallback'] for sd in SEEDS]):.3f}" for c in CORP) + "\n")

    w("## g. 생성 후 확인 (보고 12 6.2 설정)\n")
    w("| 코퍼스 | 판본 | 재추출이 필요했던 G 샘플 | 통과 샘플당 추출 수 (G) | 제외 상태 (G) | 통과 샘플당 추출 수 (div) | 제외 상태 (div) |")
    w("|---|---|---|---|---|---|---|")
    for c in CORP:
        for lab, a in (("K32", "QCs_chk"), ("K64", "QC64_chk"), ("OF", "OF_rerun")):
            r = [V[("R", 1.0)][c][sd][a] for sd in SEEDS]
            w(f"| {SHORT[c]} | {lab} | {np.mean([x['G_frac_resampled'] for x in r])*100:.1f}% | {np.mean([x['G_draws_per_accepted'] for x in r]):.3f} | "
              f"{np.mean([x['G_states_dropped'] for x in r]):.1f} | {np.mean([x['div_draws_per_accepted'] for x in r]):.3f} | {np.mean([x['div_states_dropped'] for x in r]):.1f} |")
    w("")
    pair_table(w, "R", 1.0, [("K32", "QCs_chk", "QCs"), ("K64", "QC64_chk", "QC64")], MET, "(QC + 확인) − QC")
    pair_table(w, "R", 1.0, [("K32", "QCs_chk", "A"), ("K64", "QC64_chk", "A")], MET, "(QC + 확인) − F")
    pair_table(w, "R", 1.0, [("K32", "QCs_chk", "OutFilt"), ("K64", "QC64_chk", "OutFilt")], MET, "(QC + 확인) − OF")

    w("## h. 본문 표 2 수치 (두 판본)\n")
    w("### K32 (기존)\n"); table2(w, "QCs", "A_Tmatch")
    w("### K64\n"); table2(w, "QC64", "A_Tmatch64")

    w("## 점검\n")
    w(f"- 전체 데이터 QC(K64) T = 1 생성과 보고 13 확인 실행의 최대 지표 편차: {max(V[('R', 1.0)][c][sd]['_QC64']['replay_dev'] for c in CORP for sd in SEEDS):.1e}")
    w(f"- 전체 데이터 셀 합계 {sum(V[('R', 1.0)][c][sd]['_QC64']['seconds'] for c in CORP for sd in SEEDS) / 60:.1f}분, 규모 셀 합계 "
      f"{sum(J(p)['seconds'] for p in (RQ / 'scales').glob('*.json')) / 60:.1f}분(GPU).\n")
    (RQ / "aggregate_14.md").write_text("\n".join(L), encoding="utf-8")

    with open(RQ / "per_seed_long.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh); cw.writerow(["corpus", "scale", "target", "seed", "arm", "metric", "value"])
        for (tgt, sc), byc in V.items():
            for c, bys in byc.items():
                for sd, arms in bys.items():
                    for arm, ms in arms.items():
                        if not (arm.endswith("64") or arm.startswith("QC64")) or not isinstance(ms, dict):
                            continue
                        for m, v in ms.items():
                            if isinstance(v, (int, float)):
                                cw.writerow([c, sc, tgt, sd, arm, m, v])
        for c in CORP:
            for sd in SEEDS:
                for T, ms in V[("R", 1.0)][c][sd]["_curve64"].items():
                    for m, v in ms.items():
                        cw.writerow([c, 1.0, "R_tcurve", sd, f"QC64_T{T}", m, v])
    fig_rq3(); fig_tradeoff()
    print("wrote", RQ / "aggregate_14.md", "| adopt:", adopt)


if __name__ == "__main__":
    main()
