"""Report 11 aggregation (v20 RQs). Reads the report-09 re-evaluation (results/, tradeoff/, filter_variants/) and
the report-11 evaluation (rq11/full, rq11/common) and writes
  rq11/aggregate_11a.md   stage 1: RQ1 (QC-S vs Filtered, temperature-/update-matched, filter variants),
                          RQ2 (training side: FU, OF, AC-S; generation side: q-request modes), low-quality fraction
  rq11/aggregate_11b.md   stage 2: RQ3 scale experiment (1, 1/2, 1/4, 1/8) + trend regression
  rq11/per_seed_long.csv  every per-seed value used (corpus, scale, target, seed, arm, metric, value)
  rq11/figs/fig_rq3_{recall,diversity}.png + .csv
Cell statistics: paired t over seeds (t 95% CI half-width), * p < 0.05, Holm over the 4-corpus family (dagger),
reference 8-cell family (full + 1/4 under the report-09 1/4 definition, double dagger). Run-level: 4 corpora x 5
seeds; exact two-sided sign test (ties dropped) and Wilcoxon signed-rank on the oriented differences
(recall/precision: raw; diversity: log(X/Y); best-of-N and low-quality fraction: log(Y/X) or raw for the fraction).

    .venv/Scripts/python.exe scripts/review_11_aggregate.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from review_tradeoff_aggregate import interp_at  # noqa: E402

D9 = Path("outputs/campaign_dedup"); RQ = D9 / "rq11"
CORP = ["being_so101", "rayenghali", "koch_pickplace5", "aloha_tape"]
SHORT = {"being_so101": "being", "rayenghali": "rayen", "koch_pickplace5": "koch", "aloha_tape": "aloha"}
SEEDS = range(5)
MET = ["recall", "precision", "diversity", "best_of_N"]
LOWER = {"best_of_N", "lowq_G", "lowq_div"}
TAG = {1.0: "1.00", 0.5: "0.50", 0.25: "0.25", 0.125: "0.125"}
FR = {1.0: "1", 0.5: "1/2", 0.25: "1/4", 0.125: "1/8"}
NAME = {"A": "Filtered", "A_step": "업데이트 맞춤 Filtered", "A_Tmatch": "온도 맞춤 Filtered", "C": "FU", "OutFilt": "OF",
        "B1": "AC", "ACs": "AC-S", "B2": "QC-M", "QCs": "QC-S", "A25": "activity 75% 필터", "A75": "activity 25% 필터",
        "A_con": "consistency 50% 필터", "mode:global_median": "전역 고정", "mode:per_state_fixed": "상태별 고정",
        "con:QCs": "QC-S", "con:A_con": "consistency 50% 필터", "con:A25": "activity 75% 필터", "con:A50": "Filtered",
        "con:A75": "activity 25% 필터"}

# V[(target, scale)][corpus][seed][arm] = {metric: value}; target "R" = common/full-data target, "Rsub" = report-09 1/4 target
V = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def tmatch(curveF, temps, d_star):
    q = interp_at(curveF, temps, "diversity", d_star, ["recall", "precision", "best_of_N"])
    return None if q is None else {**q, "diversity": d_star}


def load():
    for c in CORP:
        for sd in SEEDS:
            for sc, tgt in ((1.0, "R"), (0.25, "Rsub")):
                tag = f"{c}_f{TAG[sc]}_s{sd}"
                r = J(D9 / "results" / f"{tag}.json"); A = V[(tgt, sc)][c][sd]
                for arm, m in r["arms"].items():
                    A[arm] = dict(m["main"])
                for mode in ("per_state_fixed", "global_median"):
                    A[f"mode:{mode}"] = dict(r["q_modes"]["B2"][mode])
                tr = J(D9 / "tradeoff" / f"{tag}.json")
                tm = tmatch(tr["curves"]["F"], tr["temps"], tr["curves"]["QCs"]["1.0"]["diversity"])
                if tm: A["A_Tmatch"] = tm
                A["_Fmax_div"] = {"diversity": max(p["diversity"] for p in tr["curves"]["F"].values())}
                fv = J(D9 / "filter_variants" / f"{tag}.json")
                for arm in ("A25", "A75", "A_con"):
                    A[arm] = dict(fv["metrics"]["act_usable"][arm])
                assert abs(fv["metrics"]["act_usable"]["A50"]["recall"] - A["A"]["recall"]) < 1e-12
                for arm in ("A25", "A50", "A75", "A_con", "B2"):
                    A[f"con:{arm}"] = dict(fv["metrics"]["con_usable"][arm])
                if sc == 1.0:
                    x = J(RQ / "full" / f"{tag}.json")
                    for arm, m in x["arms"].items():
                        if arm in A:
                            assert abs(A[arm]["recall"] - m["recall"]) < 1e-12 and abs(A[arm]["diversity"] - m["diversity"]) < 1e-12, (tag, arm)
                        A.setdefault(arm, {}).update(m)
                    A["con:QCs"] = x["con_target"]["QCs"]; A["_ref"] = x["reference"]
                    A["_meta"] = {"fallback": x["state_cond_fallback_frac"], "con_fallback": x["con_target"]["fallback"]}
                else:
                    x = J(RQ / "common" / f"{tag}.json")
                    A["con:QCs"] = x["con_target"]["QCs"]
            for sc in (0.25, 0.5, 0.125):
                tag = f"{c}_f{TAG[sc]}_s{sd}"
                p = RQ / "common" / f"{tag}.json"
                if not p.exists():
                    continue
                x = J(p); A = V[("R", sc)][c][sd]
                for arm, m in x["arms"].items():
                    A[arm] = dict(m)
                A["_meta"] = {k: x[k] for k in ("tau_R", "tau_train", "n_eps", "n_eps_total", "episodes", "n_train_pairs",
                                                "batch_size", "usable_frac_train", "R_usable_under_train_tau", "state_cond_fallback_frac", "train")}
                if "temperature_curve_F" in x:
                    tm = tmatch(x["temperature_curve_F"], [float(t) for t in x["temperature_curve_F"]], x["arms"]["QCs"]["diversity"])
                    if tm: A["A_Tmatch"] = tm
                    A["_Fmax_div"] = {"diversity": max(p_["diversity"] for p_ in x["temperature_curve_F"].values())}
                    fvx = V[("Rsub", 0.25)][c][sd]
                    for arm in ("A25", "A50", "A75", "A_con", "B2", "QCs"):
                        A[f"con:{arm}"] = fvx[f"con:{arm}"]
                    A["_ref"] = V[("R", 1.0)][c][sd]["_ref"]


# ----------------------------------------------------------------------------- statistics
def diffs(tgt, sc, c, x, y, m):
    out = []
    for sd in SEEDS:
        A = V[(tgt, sc)][c][sd]
        if x in A and y in A and m in A[x] and m in A[y] and np.isfinite(A[x][m]) and np.isfinite(A[y][m]):
            out.append((sd, A[x][m], A[y][m]))
    return out


def paired(d, m):
    d = np.asarray(d, float); n = len(d)
    better = int(((d < 0) if m in LOWER else (d > 0)).sum())
    if n < 2:
        return dict(mean=float(d.mean()) if n else np.nan, hw=np.nan, p=np.nan, k=better, n=n)
    hw = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    p = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else (0.0 if d.mean() != 0 else 1.0)
    return dict(mean=float(d.mean()), hw=float(hw), p=float(p), k=better, n=n)


def holm(ps):
    ps = np.nan_to_num(np.asarray(ps, float), nan=1.0); o = np.argsort(ps); M = len(ps); adj = np.empty(M); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (M - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def oriented(xv, yv, m):
    """Positive = X better."""
    if m == "diversity":
        return np.log(xv / yv)
    if m == "best_of_N":
        return np.log(yv / xv)
    if m in LOWER:
        return yv - xv
    return xv - yv


def cells(tgt, sc, x, y, m):
    out = []
    for c in CORP:
        d = diffs(tgt, sc, c, x, y, m)
        out.append(paired([a - b for _, a, b in d], m))
    for st, a in zip(out, holm([st["p"] for st in out])):
        st["ph"] = float(a)
    return out


def runlevel(pairs, m):
    o = np.array([oriented(a, b, m) for a, b in pairs], float)
    nz = o[o != 0]; k = int((nz > 0).sum())
    sp = stats.binomtest(k, len(nz), 0.5).pvalue if len(nz) else np.nan
    wp = stats.wilcoxon(nz).pvalue if len(nz) >= 2 else np.nan
    return dict(k=k, n=len(nz), ties=len(o) - len(nz), total=len(o), sign_p=float(sp), wil_p=float(wp))


def run_pairs(tgt, sc, x, y, m, corpora=CORP):
    return [(a, b) for c in corpora for _, a, b in diffs(tgt, sc, c, x, y, m)]


def P(c, m):
    if m in ("lowq_G", "lowq_div"):
        return 3
    if c == "aloha_tape" and m in ("diversity", "best_of_N"):
        return 4
    return 2 if m == "best_of_N" else 3


def fmt(st, c, m, ph8=None):
    if st["n"] == 0:
        return "—"
    pr = P(c, m)
    if st["n"] < 2:
        return f"{st['mean']:+.{pr}f} (n=1)"
    s = f"{st['mean']:+.{pr}f} ± {st['hw']:.{pr}f}" + ("\\*" if st["p"] < .05 else "") + ("†" if st["ph"] < .05 else "")
    if ph8 is not None and ph8 < .05:
        s += "‡"
    return s + f" ({st['k']}/{st['n']})"


def fp(p):
    return "—" if not np.isfinite(p) else (f"{p:.3f}" if p >= 0.001 else f"{p:.1e}")


def table(w, tgt, sc, x, y, mets=MET, title=None, fam8=False, note_unreach=False):
    """Corpus rows x metric columns; bottom row = run-level. (k/n) = seeds where X is better / seeds tested."""
    if title:
        w(f"**{title}**\n")
    w("| 코퍼스 | " + " | ".join(mets) + " |"); w("|---|" + "---|" * len(mets))
    cs = {m: cells(tgt, sc, x, y, m) for m in mets}
    p8 = {}
    if fam8:
        for m in mets:
            st8 = cells("R", 1.0, x, y, m) + cells("Rsub", 0.25, x, y, m)
            a8 = holm([s["p"] for s in st8]); p8[m] = a8[:4] if sc == 1.0 else a8[4:]
    for i, c in enumerate(CORP):
        row = []
        for m in mets:
            s = fmt(cs[m][i], c, m, p8[m][i] if fam8 else None)
            if note_unreach:
                miss = 5 - cs[m][i]["n"]
                if miss: s += f" [도달 불가 {miss}]"
            row.append(s)
        w(f"| {SHORT[c]} | " + " | ".join(row) + " |")
    row = []
    for m in mets:
        rl = runlevel(run_pairs(tgt, sc, x, y, m), m)
        row.append(f"{rl['k']}/{rl['n']}" + (f" (동률 {rl['ties']} 제외)" if rl["ties"] else "") + f", 부호 p={fp(rl['sign_p'])}, Wilcoxon p={fp(rl['wil_p'])}")
    w("| **전체 (실행 단위)** | " + " | ".join(row) + " |")
    w("")
    return cs


def sig_summary(tgt, sc, x, y, m):
    cs = cells(tgt, sc, x, y, m)
    good = sum(1 for s in cs if s["p"] < .05 and ((s["mean"] < 0) if m in LOWER else (s["mean"] > 0)))
    bad = sum(1 for s in cs if s["p"] < .05 and ((s["mean"] > 0) if m in LOWER else (s["mean"] < 0)))
    goodh = sum(1 for s in cs if s["ph"] < .05 and ((s["mean"] < 0) if m in LOWER else (s["mean"] > 0)))
    badh = sum(1 for s in cs if s["ph"] < .05 and ((s["mean"] > 0) if m in LOWER else (s["mean"] < 0)))
    ms = [s["mean"] for s in cs]
    return cs, good, bad, goodh, badh, min(ms), max(ms)


# ----------------------------------------------------------------------------- 11a
def report_a():
    L = []; w = L.append
    w("# 집계 11a — 1단계 재분석 (RQ1·RQ2, 학습 없음)\n")
    w("셀: 평균 차이(앞 구성 − 뒤 구성) ± t 95% CI 반폭. \\* p < 0.05(양측 대응 t), † Holm(4코퍼스 가족) p < 0.05, "
      "‡ 참고용 기존 8칸 가족(전체 + 보고 09 정의의 1/4) Holm p < 0.05. (k/n) = 앞 구성이 나은 seed 수 / 검정 seed 수. "
      "best_of_N은 낮을수록 좋다. 전체 행: 앞 구성이 나은 실행 수 / 동률 제외 실행 수, 양측 정확 부호 검정 p, Wilcoxon 부호 순위 p "
      "(diversity는 log 비, best_of_N은 log 역비).\n")
    for tgt, sc, head in (("R", 1.0, "전체 데이터"), ("R", 0.25, "1/4 데이터 — 공통 타깃 R(전체 학습 풀 τ, 전체 데이터와 같은 1,500 상태)"),
                          ("Rsub", 0.25, "1/4 데이터 — 보고 09 정의(1/4 부분집합 τ로 다시 정한 타깃, 참고)")):
        w(f"# {head}\n")
        fam8 = tgt == "R" and sc == 1.0 or tgt == "Rsub"
        w("## 3.1 RQ1 — QC-S와 Filtered\n")
        table(w, tgt, sc, "QCs", "A", title="(1) QC-S − Filtered (T = 1)", fam8=fam8)
        table(w, tgt, sc, "QCs", "A_Tmatch", mets=["recall", "precision", "best_of_N"], note_unreach=True,
              title="(2) QC-S − 온도 맞춤 Filtered (Filtered 곡선을 QC-S의 T = 1 diversity에 맞춤; diversity는 정의상 0)", fam8=fam8)
        fmx = [np.mean([V[(tgt, sc)][c][sd]["_Fmax_div"]["diversity"] for sd in SEEDS]) for c in CORP]
        dqs = [np.mean([V[(tgt, sc)][c][sd]["QCs"]["diversity"] for sd in SEEDS]) for c in CORP]
        w("Filtered 곡선의 최대 diversity / QC-S 목표: " + ", ".join(f"{SHORT[c]} {a:.4f}/{b:.4f}" for c, a, b in zip(CORP, fmx, dqs)) + "\n")
        table(w, tgt, sc, "QCs", "A_step", title="(3) QC-S − 업데이트 맞춤 Filtered (196–200 epoch)", fam8=fam8)

        w("## 3.2 RQ1 — 필터 변형과 QC-S\n")
        for f in ("A25", "A75", "A_con"):
            table(w, tgt, sc, "QCs", f, title=f"QC-S − {NAME[f]} (타깃 R)", fam8=fam8)
        if tgt == "R" and sc == 1.0 or tgt == "Rsub":
            label = "전체" if sc == 1.0 else "1/4 (그 부분집합의 consistency 타깃; 공통 R과 무관)"
            for f in ("A_con", "A25", "A50", "A75"):
                table(w, tgt, sc, "con:QCs", f"con:{f}", title=f"consistency 타깃, {label}: QC-S − {NAME['con:' + f]}")
            table(w, tgt, sc, "con:QCs", "con:B2", title=f"참고 — consistency 타깃, {label}: QC-S − QC-M")
        w("참고 — 각 변형 필터의 recall이 표준 Filtered보다 낮은 칸 수(평균 기준 / 유의):\n")
        for f in ("A25", "A75", "A_con"):
            cs = cells(tgt, sc, f, "A", "recall")
            w(f"- {NAME[f]}: 평균이 낮은 칸 {sum(s['mean'] < 0 for s in cs)}/4, 유의하게 낮은 칸 {sum(s['mean'] < 0 and s['p'] < .05 for s in cs)}/4")
        w("")

        w("## 3.3 RQ2 학습 측 — 데이터 보존 대 품질 조건\n")
        table(w, tgt, sc, "QCs", "C", title="QC-S − FU (Full-Unconditioned)", fam8=fam8)
        table(w, tgt, sc, "QCs", "OutFilt", title="QC-S − OF (Output-Filtered; 64회 재추출, 통과 못한 상태 제외)", fam8=fam8)
        if "ACs" in V[(tgt, sc)][CORP[0]][0]:
            table(w, tgt, sc, "QCs", "ACs", title="QC-S − AC-S (AC-S: Activity-Conditioned, q는 QC-S와 같은 32 이웃 usable 집합)")
            table(w, tgt, sc, "ACs", "A", title="AC-S − Filtered")
            table(w, tgt, sc, "ACs", "B1", title="참고 — AC-S − AC(분포 추출)")
        else:
            table(w, tgt, sc, "QCs", "B1", title="QC-S − AC(분포 추출) (이 정의에는 AC-S 없음)")
        if tgt == "R":
            fb = [np.mean([V[(tgt, sc)][c][sd].get("_meta", {}).get("fallback", V[(tgt, sc)][c][sd].get("_meta", {}).get("state_cond_fallback_frac", np.nan)) for sd in SEEDS]) for c in CORP]
            w("상태 이웃 fallback 비율(셀 평균): " + ", ".join(f"{SHORT[c]} {f:.3f}" for c, f in zip(CORP, fb)) + "\n")

        w("## 3.4 RQ2 생성 측 — 요청 방식 (같은 Quality-Conditioned 모델, 각 방식 − QC-S)\n")
        w("4칸 범위: recall·precision은 원 차이, diversity·best_of_N은 QC-S 대비 상대 변화(seed별 비 − 1의 평균; 코퍼스 간 단위가 달라서).\n")
        w("| 방식 | 지표 | 4칸 범위 | 유의 칸 (방식 우위/QC-S 우위) | Holm | 실행 단위: QC-S가 나은 수/n, 부호 p, Wilcoxon p |")
        w("|---|---|---|---|---|---|")
        for mode in ("mode:global_median", "B2", "mode:per_state_fixed"):
            for m in MET:
                cs, g, b, gh, bh, lo, hi = sig_summary(tgt, sc, mode, "QCs", m)
                rl = runlevel(run_pairs(tgt, sc, mode, "QCs", m), m)
                if m in ("diversity", "best_of_N"):
                    rel = [np.mean([a / b_ - 1 for _, a, b_ in diffs(tgt, sc, c, mode, "QCs", m)]) for c in CORP]
                    rng_ = f"{min(rel)*100:+.0f}% … {max(rel)*100:+.0f}%"
                else:
                    rng_ = f"{lo:+.3f} … {hi:+.3f}"
                w(f"| {NAME[mode]} | {m} | {rng_} | {g}/{b} | {gh}/{bh} | "
                  f"{rl['n'] - rl['k']}/{rl['n']}, {fp(rl['sign_p'])}, {fp(rl['wil_p'])} |")
        w("\n상태별 고정(부록): 각 평가 상태에 Q에서 뽑은 q 하나를 고정한다. G(상태당 1개)의 q는 QC-M과 같은 분포에서 뽑히므로 "
          "precision·recall의 기댓값은 QC-M과 같다. 차이는 diversity·best-of-N(상태당 8개가 같은 q)에서만 생긴다.\n")
        if tgt == "R":
            lowq_section(w, sc)
    (RQ / "aggregate_11a.md").write_text("\n".join(L), encoding="utf-8")


def lowq_section(w, sc):
    w(f"## 3.5 저품질 생성 비율 (activity < τ, τ = 타깃 R을 정의한 전체 학습 풀 activity 중앙값) — {FR[sc]} 데이터\n")
    arms = [a for a in ("A", "C", "B1", "ACs", "B2", "QCs") if "lowq_G" in V[("R", sc)][CORP[0]][0].get(a, {})]
    for key, lab in (("lowq_G", "생성 집합 G (1,500)"), ("lowq_div", "diversity 집합 (256 × 8)")):
        w(f"**{lab}: 구성별 평균 ± 95% CI 반폭**\n")
        w("| 코퍼스 | " + " | ".join(NAME[a] for a in arms) + " | held-out 전체 chunk (고유) | 타깃 R |")
        w("|---|" + "---|" * (len(arms) + 2))
        for c in CORP:
            row = []
            for a in arms:
                v = np.array([V[("R", sc)][c][sd][a][key] for sd in SEEDS])
                row.append(f"{v.mean()*100:.1f} ± {stats.t.ppf(.975, 4) * v.std(ddof=1) / np.sqrt(5) * 100:.1f}%")
            ref = [V[("R", sc)][c][sd]["_ref"] for sd in SEEDS]
            row.append(f"{np.mean([r['heldout_lowq_all'] for r in ref])*100:.1f}% ({np.mean([r['heldout_lowq_unique'] for r in ref])*100:.1f}%)")
            row.append(f"{np.mean([r['R_lowq'] for r in ref])*100:.1f}%")
            w(f"| {SHORT[c]} | " + " | ".join(row) + " |")
        w("")
    w("**activity·smoothness z 평균 (G; 전체 학습 풀 기준 z) — 구성별, 마지막 열은 타깃 R**\n")
    w("| 코퍼스 | " + " | ".join(NAME[a] for a in arms) + " | 타깃 R |")
    w("|---|" + "---|" * (len(arms) + 1))
    for c in CORP:
        row = []
        for a in arms:
            row.append(f"{np.mean([V[('R', sc)][c][sd][a]['zact_G'] for sd in SEEDS]):+.2f} / {np.mean([V[('R', sc)][c][sd][a]['zsmooth_G'] for sd in SEEDS]):+.2f}")
        ref = [V[("R", sc)][c][sd]["_ref"] for sd in SEEDS]
        row.append(f"{np.mean([r['R_zact'] for r in ref]):+.2f} / {np.mean([r['R_zsmooth'] for r in ref]):+.2f}")
        w(f"| {SHORT[c]} | " + " | ".join(row) + " |")
    w("\n(셀 = activity z / smoothness z. smoothness z = −jerk z, 클수록 매끄럽다.)\n")
    for y in [a for a in ("A", "C") if a in arms]:
        table(w, "R", sc, "QCs", y, mets=["lowq_G", "lowq_div"],
              title=f"QC-S − {NAME[y]}: 저품질 비율 차이 (음수 = QC-S가 적음; (k/n) = QC-S가 적은 seed 수)")


# ----------------------------------------------------------------------------- 11b
SCALES = [1.0, 0.5, 0.25, 0.125]


def report_b():
    L = []; w = L.append
    w("# 집계 11b — RQ3 규모 실험 (데이터 비율 1, 1/2, 1/4, 1/8; 공통 타깃 R)\n")
    w("모든 규모에서 같은 평가 타깃(held-out 분할 + 전체 학습 풀 τ, seed별 1,500 상태)을 쓴다. 1은 보고 09 값, 1/4은 공통 R로 재평가한 값이다.\n")
    # subsets / steps
    w("## 부분집합과 학습 규모\n")
    w("| 코퍼스 | 비율 | 에피소드 수 (seed 0–4) | 학습 쌍 수 평균 | 배치 | Filtered step | QC step | Filtered 학습 쌍 비율 |")
    w("|---|---|---|---|---|---|---|---|")
    camp = {}
    for c in CORP:
        for sc in SCALES:
            if sc == 1.0:
                rs = [J(Path("outputs/campaign/results") / f"{c}_f1.00_s{sd}.json") for sd in SEEDS]
                eps = [V[("R", 0.25)][c][sd]["_meta"]["n_eps_total"] for sd in SEEDS]
                npair = np.mean([r["n_train_pairs"] for r in rs]); bs = {r["batch_size"] for r in rs}
                sa = [r["arms"]["A"]["train"]["steps"] for r in rs]; sb = [r["arms"]["B2"]["train"]["steps"] for r in rs]
                uf = np.mean([r["usable_frac_train"]["raw"] for r in rs])
            else:
                M = [V[("R", sc)][c][sd]["_meta"] for sd in SEEDS]
                eps = [m["n_eps"] for m in M]; npair = np.mean([m["n_train_pairs"] for m in M]); bs = {m["batch_size"] for m in M}
                sa = [m["train"]["A"]["steps"] for m in M]; sb = [m["train"]["B2"]["steps"] for m in M]
                uf = np.mean([m["usable_frac_train"] for m in M])
            w(f"| {SHORT[c]} | {FR[sc]} | {', '.join(map(str, eps))} | {npair:,.0f} | {'/'.join(map(str, sorted(bs)))} | "
              f"{min(sa):,}–{max(sa):,} | {min(sb):,}–{max(sb):,} | {uf:.3f} |")
    # nesting check
    bad = 0
    for c in CORP:
        for sd in SEEDS:
            e8, e4, e2 = (set(V[("R", s)][c][sd]["_meta"]["episodes"]) for s in (0.125, 0.25, 0.5))
            bad += not (e8 <= e4 <= e2)
    w(f"\n중첩 확인(1/8 ⊂ 1/4 ⊂ 1/2 ⊂ 전체): 위반 {bad}/20.\n")
    w("학습 측 τ(부분집합 activity 중앙값)와 공통 R의 τ 비(τ_train/τ_R) 및 공통 R chunk 중 학습 τ 기준 usable 비율:\n")
    for c in CORP:
        w(f"- {SHORT[c]}: " + ", ".join(
            f"{FR[sc]} {np.mean([V[('R', sc)][c][sd]['_meta']['tau_train'] / V[('R', sc)][c][sd]['_meta']['tau_R'] for sd in SEEDS]):.3f}"
            f" / {np.mean([V[('R', sc)][c][sd]['_meta']['R_usable_under_train_tau'] for sd in SEEDS]):.3f}" for sc in SCALES[1:]))
    fb = {c: [np.mean([V[("R", sc)][c][sd].get("_meta", {}).get("state_cond_fallback_frac", V[("R", sc)][c][sd].get("_meta", {}).get("fallback", np.nan))
                       for sd in SEEDS]) for sc in SCALES] for c in CORP}
    w("\nQC-S 상태 이웃 fallback 비율(1, 1/2, 1/4, 1/8): " + "; ".join(f"{SHORT[c]} " + ", ".join(f"{v:.3f}" for v in fb[c]) for c in CORP) + "\n")

    # means
    w("## 구성별 평균 (5 seed)\n")
    w("| 코퍼스 | 비율 | recall F / QC-S | precision F / QC-S | diversity F / QC-S | best-of-N F / QC-S | 저품질 G F / QC-S |")
    w("|---|---|---|---|---|---|---|")
    for c in CORP:
        for sc in SCALES:
            row = []
            for m in MET + ["lowq_G"]:
                a = np.mean([V[("R", sc)][c][sd]["A"][m] for sd in SEEDS]); q = np.mean([V[("R", sc)][c][sd]["QCs"][m] for sd in SEEDS])
                pr = 4 if (c == "aloha_tape" and m in ("diversity", "best_of_N")) else (2 if m == "best_of_N" else 3)
                row.append(f"{a*100:.1f}% / {q*100:.1f}%" if m == "lowq_G" else f"{a:.{pr}f} / {q:.{pr}f}")
            w(f"| {SHORT[c]} | {FR[sc]} | " + " | ".join(row) + " |")
    w("")

    # per-scale tables with 16-cell Holm
    w("## 규모·코퍼스별 QC-S − Filtered\n")
    w("\\* p < 0.05, † Holm(4코퍼스 × 4규모 = 16칸 가족, 지표별) p < 0.05, (k/5) = QC-S가 나은 seed 수. 전체 행 = 그 규모의 실행 단위 20회.\n")
    mets = MET + ["lowq_G"]
    allc = {m: {sc: cells("R", sc, "QCs", "A", m) for sc in SCALES} for m in mets}
    for m in mets:
        a16 = holm([st["p"] for sc in SCALES for st in allc[m][sc]])
        for j, sc in enumerate(SCALES):
            for i, st in enumerate(allc[m][sc]):
                st["ph"] = float(a16[j * 4 + i])
    for sc in SCALES:
        w(f"**데이터 비율 {FR[sc]}**\n")
        w("| 코퍼스 | " + " | ".join(mets) + " |"); w("|---|" + "---|" * len(mets))
        for i, c in enumerate(CORP):
            w(f"| {SHORT[c]} | " + " | ".join(fmt(allc[m][sc][i], c, m) for m in mets) + " |")
        row = []
        for m in mets:
            rl = runlevel(run_pairs("R", sc, "QCs", "A", m), m)
            row.append(f"{rl['k']}/{rl['n']}" + (f" (동률 {rl['ties']})" if rl["ties"] else "") + f", 부호 p={fp(rl['sign_p'])}, W p={fp(rl['wil_p'])}")
        w("| **전체 (실행 단위)** | " + " | ".join(row) + " |\n")

    # trend
    w("## 추세: seed 수준 Δ를 log2(데이터 비율)에 회귀\n")
    w("Δ = QC-S − Filtered를 실행 단위 검정과 같은 방향으로 둔다(양수 = QC-S 우위; recall·precision 원 차이, diversity log(QC-S/F), "
      "best-of-N log(F/QC-S), 저품질 비율 F − QC-S). x = log2(비율) ∈ {0, −1, −2, −3}. "
      "**'절반마다 변화' = −기울기**(데이터가 절반이 될 때 Δ가 늘어나는 양; 양수 = 데이터가 적을수록 격차 확대).\n")
    w("- 코퍼스별: OLS, 20점(5 seed × 4 규모), t 95% CI(df 18).")
    w("- 통합: 코퍼스 고정효과 + 공통 기울기, OLS(80점, df 75).")
    w("- 견고성: 같은 seed는 held-out 분할과 중첩 부분집합을 공유하므로 점들이 독립이 아니다. (코퍼스, seed)마다 기울기를 따로 구해 "
      "20개 기울기의 평균 ± t CI(df 19)와 부호 수도 적는다.\n")
    w("| 지표 | " + " | ".join(SHORT[c] for c in CORP) + " | 통합(고정효과) | (코퍼스, seed)별 기울기 평균, 양수 수/20 |")
    w("|---|" + "---|" * (len(CORP) + 2))
    trend_rows = []
    for m in mets:
        pts = {c: [] for c in CORP}
        for c in CORP:
            for sc in SCALES:
                for sd in SEEDS:
                    A = V[("R", sc)][c][sd]
                    pts[c].append((sd, np.log2(sc), float(oriented(A["QCs"][m], A["A"][m], m))))
        row = []
        for c in CORP:
            x = np.array([p[1] for p in pts[c]]); y = np.array([p[2] for p in pts[c]])
            r = stats.linregress(x, y); hw = stats.t.ppf(.975, len(x) - 2) * r.stderr
            row.append(f"{-r.slope:+.4f} ± {hw:.4f}" + ("\\*" if r.pvalue < .05 else "") + f" (p={fp(r.pvalue)})")
            trend_rows.append((m, c, -r.slope, hw, r.pvalue))
        X, Y = [], []
        for ci, c in enumerate(CORP):
            for _, x, y in pts[c]:
                X.append([x] + [1.0 if cj == ci else 0.0 for cj in range(len(CORP))]); Y.append(y)
        X, Y = np.array(X), np.array(Y)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta; dfree = len(Y) - X.shape[1]; s2 = res @ res / dfree
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0]); tval = beta[0] / se; pv = 2 * stats.t.sf(abs(tval), dfree)
        hw = stats.t.ppf(.975, dfree) * se
        row.append(f"{-beta[0]:+.4f} ± {hw:.4f}" + ("\\*" if pv < .05 else "") + f" (p={fp(pv)})")
        sl = []
        for c in CORP:
            for sd in SEEDS:
                xs = np.array([p[1] for p in pts[c] if p[0] == sd]); ys = np.array([p[2] for p in pts[c] if p[0] == sd])
                sl.append(-np.polyfit(xs, ys, 1)[0])
        sl = np.array(sl); t = stats.ttest_1samp(sl, 0)
        row.append(f"{sl.mean():+.4f} ± {stats.t.ppf(.975, 19) * sl.std(ddof=1) / np.sqrt(20):.4f} (p={fp(t.pvalue)}), {int((sl > 0).sum())}/20")
        w(f"| {m} | " + " | ".join(row) + " |")
    w("\n(diversity·best-of-N 단위는 log 비이므로 절반마다 변화 0.10 ≈ 비율 10.5% 증가.)\n")

    figures(w)
    (RQ / "aggregate_11b.md").write_text("\n".join(L), encoding="utf-8")


def figures(w):
    out = RQ / "figs"; out.mkdir(exist_ok=True)
    INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
    ARMS = [("A", "Filtered", "#2a78d6", "o"), ("QCs", "QC-S", "#1baf7a", "^")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "legend.frameon": False})
    for m, ylab in (("recall", "Recall"), ("diversity", "Diversity")):
        rows = []
        fig, axs = plt.subplots(1, 4, figsize=(11.0, 2.9))
        for j, c in enumerate(CORP):
            ax = axs[j]
            for arm, lab, col, mk in ARMS:
                mean, hw = [], []
                for sc in SCALES:
                    v = np.array([V[("R", sc)][c][sd][arm][m] for sd in SEEDS])
                    mean.append(v.mean()); hw.append(stats.t.ppf(.975, 4) * v.std(ddof=1) / np.sqrt(5))
                    rows.append([c, sc, lab, len(v), v.mean(), hw[-1]] + list(v))
                mean, hw = np.array(mean), np.array(hw)
                ax.plot(SCALES, mean, "-", color=col, lw=2, zorder=2)
                ax.fill_between(SCALES, mean - hw, mean + hw, color=col, alpha=0.15, lw=0, zorder=1)
                ax.errorbar(SCALES, mean, yerr=hw, fmt="none", ecolor=col, elinewidth=0.8, capsize=2, zorder=2)
                ax.scatter(SCALES, mean, s=26, marker=mk, color=col, edgecolors="white", linewidths=0.8, zorder=3,
                           label=lab if j == 0 else None)
            ax.set_xscale("log", base=2); ax.set_xticks(SCALES); ax.set_xticklabels(["1", "1/2", "1/4", "1/8"])
            ax.set_xlim(1.25, 0.1)
            if m == "diversity" and c == "aloha_tape":
                ax.set_yscale("log")
            ax.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_title(SHORT[c] if c != "being_so101" else "being_so101", color=INK, fontsize=9.5, loc="left")
            ax.set_xlabel("Training-data fraction")
            if j == 0:
                ax.set_ylabel(ylab)
            if m == "diversity" and c == "aloha_tape":
                ax.set_ylabel("Diversity (log, rad)")
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / f"fig_rq3_{m}.png", dpi=300, bbox_inches="tight"); plt.close(fig)
        with open(out / f"fig_rq3_{m}.csv", "w", newline="", encoding="utf-8") as fh:
            cw = csv.writer(fh); cw.writerow(["corpus", "fraction", "arm", "n_seeds", "mean", "ci95_halfwidth"] + [f"seed{s}" for s in SEEDS])
            cw.writerows(rows)
    w(f"그림: `{out / 'fig_rq3_recall.png'}`, `{out / 'fig_rq3_diversity.png'}` (데이터 `.csv` 동봉). "
      "선 = 5 seed 평균, 띠·오차 막대 = t 95% CI. 가로축 log2, 왼쪽이 전체 데이터. aloha diversity는 log 축.\n")


def write_csv():
    with open(RQ / "per_seed_long.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh); cw.writerow(["corpus", "scale", "target", "seed", "arm", "metric", "value"])
        for (tgt, sc), byc in sorted(V.items()):
            for c, bys in byc.items():
                for sd, arms in sorted(bys.items()):
                    for arm, ms in arms.items():
                        if arm.startswith("_") or not isinstance(ms, dict):
                            continue
                        for m, v in ms.items():
                            if isinstance(v, (int, float)):
                                cw.writerow([c, sc, tgt, sd, arm, m, v])


def main():
    load()
    write_csv(); report_a()
    if all((RQ / "common" / f"{c}_f{TAG[s]}_s{sd}.json").exists() for c in CORP for s in (0.5, 0.125) for sd in SEEDS):
        report_b()
    else:
        print("stage 2 incomplete; 11b skipped")
    print("wrote", RQ / "aggregate_11a.md")


if __name__ == "__main__":
    main()
