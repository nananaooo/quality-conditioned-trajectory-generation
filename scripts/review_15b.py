"""Report 15b: precise K64-version numbers for the manuscript (aggregation of saved results only; the only re-scoring is
of SAVED samples — the Filtered RQ3 samples for the stop / boundary split). No generation, no training.
Writes outputs/campaign_dedup/rq15b/{aggregate_15b.md, per_seed_15b.csv}.

    .venv/Scripts/python.exe scripts/review_15b.py
"""
from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

import review_11_aggregate as agg
import review_14_aggregate as a14
from review_11_aggregate import CORP, SEEDS, SHORT, MET, FR, J, runlevel, run_pairs, fp, oriented, holm

OUT = Path("outputs/campaign_dedup/rq15b"); D9 = Path("outputs/campaign_dedup")
V = agg.V
SCALES = [1.0, 0.5, 0.25, 0.125]
TAG = {1.0: "1.00", 0.5: "0.50", 0.25: "0.25", 0.125: "0.125"}
ROWS = []          # per-seed CSV rows


def ms(v):
    v = np.asarray(v, float); return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def sig3(m, s):
    if m == 0 or not np.isfinite(m):
        return f"{m:.3g} ± {s:.3g}"
    d = max(0, 2 - int(math.floor(math.log10(abs(m)))))
    return f"{m:.{d}f} ± {s:.{d}f}"


def cellfmt(c, m, vals):
    v = np.asarray(vals, float)
    if c == "aloha_tape" and m == "diversity": v = v * 1e3
    if c == "aloha_tape" and m == "best_of_N": v = v * 1e4
    mu, sd = ms(v)
    if m in ("recall", "precision"):
        return f"{mu:.3f} ± {sd:.3f}"
    if m in ("lowq_G", "stop_G", "boundary_G"):
        return f"{mu*100:.2f} ± {sd*100:.2f}"
    return sig3(mu, sd)


def paired_p(d):
    d = np.asarray(d, float)
    return float(stats.ttest_1samp(d, 0).pvalue) if d.std() > 0 else (0.0 if d.mean() != 0 else 1.0)


# ----------------------------------------------------------------------------- RQ3 stop / boundary for F (saved samples)
def f_split_scales():
    import sys
    sys.path.insert(0, "scripts")
    from review_11_rq import Cell
    from run_review_campaign import act_raw
    cc = yaml.safe_load(Path("configs/review_campaign.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(cc["base_config"]).read_text(encoding="utf-8"))
    chk = []
    for c in CORP:
        for sd in SEEDS:
            full = Cell(c, 1.0, sd, cfg)
            p25, tau = float(np.quantile(full.ax.act_tr, 0.25)), full.ax.tau
            V[("RQ3", 1.0)][c][sd]["_p25"] = p25
            for sc in (0.5, 0.25, 0.125):
                z = np.load(D9 / "rq11" / "samples" / f"{c}_f{TAG[sc]}_s{sd}.npz")
                x = act_raw(z["A_gen"])
                A = V[("RQ3", sc)][c][sd]["A"]
                stop, below = float((x < p25).mean()), float((x < tau).mean())
                chk.append(abs(below - A["lowq_G"]))
                A["stop_G"], A["boundary_G"] = stop, below - stop
    return max(chk)


# ----------------------------------------------------------------------------- report
def main():
    t0 = time.time()
    a14.load()
    maxdev = f_split_scales()
    L = []; w = L.append
    w("# 집계 15b — K64 판본 정밀 수치와 지표 구현 확인\n")
    w("셀 = 5 seed 평균 ± 표준편차(절대값). aloha diversity ×10⁻³ rad, best-of-N ×10⁻⁴ rad². 비율 열은 %. "
      "실행 단위 = 20회 양측 정확 부호 검정(동률·결과 없음 제외, n 표기) + Wilcoxon.\n")

    # ================================================================ a. RQ3
    w("## a. RQ3 비율별 절대값 (공통 타깃 R, QC는 K 비례 64/32/16/8) — 원고 표 5, 부록 표 A6(a), 초록\n")
    w(f"F의 1/2–1/8 정지·경계는 저장 샘플(`rq11/samples/`)을 다시 채점한 값이다. 정지 임계 = 그 seed **전체 학습 풀** activity의 p25, "
      f"τ = 전체 학습 풀 중앙값(공통 R과 같은 τ). 재채점한 τ 아래 비율과 저장된 값의 최대 차이: {maxdev:.1e}.\n")
    w("| 코퍼스 | 비율 | 구성 | recall | precision | diversity | best-of-N | τ 아래 (%) | 정지 (%) | 경계 (%) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for c in CORP:
        for sc in SCALES:
            for lab, arm in (("F", "A"), ("QC", "QC64")):
                rr = [V[("RQ3", sc)][c][sd][arm] for sd in SEEDS]
                cells = [cellfmt(c, m, [r[m] for r in rr]) for m in MET]
                cells.append(cellfmt(c, "lowq_G", [r["lowq_G"] for r in rr]))
                for m in ("stop_G", "boundary_G"):
                    cells.append(cellfmt(c, m, [r[m] for r in rr]) if all(m in r for r in rr) else "**새 생성 필요**")
                w(f"| {SHORT[c]} | {FR[sc]} | {lab} | " + " | ".join(cells) + " |")
                for sd, r in zip(SEEDS, rr):
                    for m in MET + ["lowq_G", "stop_G", "boundary_G"]:
                        if m in r: ROWS.append(["a_RQ3", c, sc, sd, lab, m, r[m]])
    w("")
    w("**QC − F 평균 recall 차이 (4코퍼스 평균, 코퍼스 평균의 평균)** — 표 5 차이 행, 초록(1/8)\n")
    w("| 비율 | " + " | ".join(SHORT[c] for c in CORP) + " | 4코퍼스 평균 |"); w("|---|---|---|---|---|---|")
    for sc in SCALES:
        d = [np.mean([V[("RQ3", sc)][c][sd]["QC64"]["recall"] - V[("RQ3", sc)][c][sd]["A"]["recall"] for sd in SEEDS]) for c in CORP]
        w(f"| {FR[sc]} | " + " | ".join(f"{x:+.4f}" for x in d) + f" | **{np.mean(d):+.4f}** |")
    w("")
    w("**QC − F 대응 t-검정: Holm 16칸(4코퍼스 × 4비율, 지표별) 대 Holm 4코퍼스(비율별, 보고 14의 †)** — 셀 = 평균 차이 (p, p_Holm16, p_Holm4); ‡ = Holm16 < 0.05, † = Holm4 < 0.05\n")
    mets = MET + ["lowq_G"]
    for m in mets:
        D = {(c, sc): [V[("RQ3", sc)][c][sd]["QC64"][m] - V[("RQ3", sc)][c][sd]["A"][m] for sd in SEEDS] for c in CORP for sc in SCALES}
        keys = list(D); ps = [paired_p(D[k]) for k in keys]; h16 = dict(zip(keys, holm(ps)))
        h4 = {}
        for sc in SCALES:
            ks = [(c, sc) for c in CORP]; h4.update(dict(zip(ks, holm([paired_p(D[k]) for k in ks]))))
        w(f"*{m}*\n"); w("| 코퍼스 | " + " | ".join(FR[s] for s in SCALES) + " |"); w("|---|---|---|---|---|")
        for c in CORP:
            row = []
            for sc in SCALES:
                d = np.mean(D[(c, sc)]); p = paired_p(D[(c, sc)])
                pr = 4 if (c == "aloha_tape" and m in ("diversity", "best_of_N")) else (2 if m == "best_of_N" else (4 if m == "lowq_G" else 3))
                row.append(f"{d:+.{pr}f} (p={fp(p)}, H16={fp(h16[(c, sc)])}, H4={fp(h4[(c, sc)])}){'‡' if h16[(c, sc)] < .05 else ''}{'†' if h4[(c, sc)] < .05 else ''}")
            w(f"| {SHORT[c]} | " + " | ".join(row) + " |")
        n16 = sum(v < .05 for v in h16.values()); n4 = sum(v < .05 for v in h4.values())
        w(f"\n유의 칸: Holm16 {n16}/16, Holm4 {n4}/16\n")
    w("**기울기 (seed별 QC − F를 log2(1/비율)에 회귀; recall·precision 원 차이, diversity log(QC/F), best-of-N log(F/QC), τ 아래 F − QC)**\n")
    rep14 = {"recall": {"being_so101": 0.0012, "rayenghali": 0.0114, "koch_pickplace5": 0.0309, "aloha_tape": 0.0141, "pooled": 0.0144}}
    w("| 지표 | " + " | ".join(SHORT[c] for c in CORP) + " | 통합(고정효과) | (코퍼스, seed)별 |"); w("|---|---|---|---|---|---|---|")
    for m in mets:
        per, pooled, clus = a14.trend("QC64", m)
        w(f"| {m} | " + " | ".join(f"{per[c][0]:+.4f} ± {per[c][1]:.4f} (p={fp(per[c][2])})" for c in CORP)
          + f" | {pooled[0]:+.4f} ± {pooled[1]:.4f} (p={fp(pooled[2])}) | {clus[0]:+.4f} (p={fp(clus[2])}), {clus[3]}/20 |")
    per, pooled, _ = a14.trend("QC64", "recall")
    same = all(round(per[c][0], 4) == rep14["recall"][c] for c in CORP) and round(pooled[0], 4) == rep14["recall"]["pooled"]
    w(f"\n보고 14(K64 행)와 대조 — recall 코퍼스별·통합 기울기: **{'일치' if same else '불일치'}**. 다른 지표도 같은 함수(`review_14_aggregate.trend`)와 같은 데이터라 보고 14 표와 같다.\n")
    w("**1/8 best-of-N (원고: F가 더 좋음)**\n")
    w("| 코퍼스 | F | QC | QC − F | 대응 t p | QC가 나은 seed |"); w("|---|---|---|---|---|---|")
    for c in CORP:
        f = [V[("RQ3", 0.125)][c][sd]["A"]["best_of_N"] for sd in SEEDS]; q = [V[("RQ3", 0.125)][c][sd]["QC64"]["best_of_N"] for sd in SEEDS]
        d = np.array(q) - np.array(f); sc_ = 1e4 if c == "aloha_tape" else 1
        w(f"| {SHORT[c]} | {cellfmt(c, 'best_of_N', f)} | {cellfmt(c, 'best_of_N', q)} | {d.mean()*sc_:+.3g} | {fp(paired_p(d))} | {int((d < 0).sum())}/5 |")
    w("")

    # ================================================================ b. Table 3
    w("## b. 표 3 (전체 데이터, 타깃 R, T = 1)\n")
    rows = [("F", "A"), ("FU", "C"), ("OF", "OutFilt"), ("AC", "AC64"), ("전역 고정", "mode:global_median"), ("상태 무관 추출", "B2"), ("QC", "QC64")]
    single = CORP[:3]
    def cmean(c, arm, m): return np.mean([V[("R", 1.0)][c][sd][arm][m] for sd in SEEDS])
    w("| 행 | recall (4코퍼스 평균) | precision (4코퍼스 평균) | diversity 상대 (단일 팔 3 평균; 코퍼스별) | best-of-N 상대 (단일 팔 3 평균; 코퍼스별) | aloha diversity 상대 | aloha best-of-N 상대 |")
    w("|---|---|---|---|---|---|---|")
    for lab, arm in rows:
        rec = np.mean([cmean(c, arm, "recall") for c in CORP]); pre = np.mean([cmean(c, arm, "precision") for c in CORP])
        rel = {m: {c: 100 * cmean(c, arm, m) / cmean(c, "QC64", m) for c in CORP} for m in ("diversity", "best_of_N")}
        w(f"| {lab} | {rec:.3f} | {pre:.3f} | {np.mean([rel['diversity'][c] for c in single]):.1f} ({', '.join(f'{rel['diversity'][c]:.1f}' for c in single)}) | "
          f"{np.mean([rel['best_of_N'][c] for c in single]):.1f} ({', '.join(f'{rel['best_of_N'][c]:.1f}' for c in single)}) | {rel['diversity']['aloha_tape']:.1f} | {rel['best_of_N']['aloha_tape']:.1f} |")
        for c in CORP:
            for sd in SEEDS:
                for m in MET:
                    ROWS.append(["b_table3", c, 1.0, sd, lab, m, V[("R", 1.0)][c][sd][arm][m]])
    w("\n**각 행 대비 QC의 실행 단위 판정 (QC가 나은 수/n, 부호 p; Wilcoxon p)** — 표 3의 \\*\n")
    w("| 행 | recall | precision | diversity | best-of-N |"); w("|---|---|---|---|---|")
    for lab, arm in rows[:-1]:
        cells = []
        for m in MET:
            rl = runlevel(run_pairs("R", 1.0, "QC64", arm, m), m)
            cells.append(f"{rl['k']}/{rl['n']}, p={fp(rl['sign_p'])}{' \\*' if rl['sign_p'] < .05 else ''} (W {fp(rl['wil_p'])})")
        w(f"| {lab} | " + " | ".join(cells) + " |")
    w("\n주의: OF의 precision·recall은 64회 안에 통과한 상태(G·R 모두 그 상태로 제한), diversity·best-of-N은 N = 8을 채운 상태만으로 계산한다(보고 09 규칙).\n")

    # ================================================================ c. A5(a)
    w("## c. 부록 표 A5(a): 요청 방식 − QC (전체 데이터)\n")
    w("recall·precision = 코퍼스별 평균 차이; diversity·best-of-N = 100 × (방식 평균 / QC 평균 − 1) %. 실행 단위 = QC가 나은 수/n.\n")
    w("| 방식 | 지표 | being | rayen | koch | aloha | 범위 | 실행 단위 (QC 우위/n, 부호 p, W p) |"); w("|---|---|---|---|---|---|---|---|")
    vals = {}
    for lab, arm in (("전역 고정", "mode:global_median"), ("상태 무관 추출", "B2"), ("상태별 고정", "mode:per_state_fixed")):
        for m in MET:
            if m in ("recall", "precision"):
                x = [cmean(c, arm, m) - cmean(c, "QC64", m) for c in CORP]; f = [f"{v:+.3f}" for v in x]; rng = f"{min(x):+.3f} … {max(x):+.3f}"
            else:
                x = [100 * (cmean(c, arm, m) / cmean(c, "QC64", m) - 1) for c in CORP]; f = [f"{v:+.1f}%" for v in x]; rng = f"{min(x):+.0f} … {max(x):+.0f}%"
            vals[(lab, m)] = x
            rl = runlevel(run_pairs("R", 1.0, "QC64", arm, m), m)
            w(f"| {lab} | {m} | " + " | ".join(f) + f" | {rng} | {rl['k']}/{rl['n']}, {fp(rl['sign_p'])}, {fp(rl['wil_p'])} |")
    g, p_ = vals[("전역 고정", "diversity")], vals[("상태별 고정", "diversity")]
    w(f"\n**점검 — diversity 범위:** 전역 고정 {', '.join(f'{v:+.2f}%' for v in g)} / 상태별 고정 {', '.join(f'{v:+.2f}%' for v in p_)}. "
      f"정수로 반올림한 범위는 각각 {min(g):+.0f} … {max(g):+.0f}%, {min(p_):+.0f} … {max(p_):+.0f}%이다. "
      "두 방식 모두 한 상태의 8개 샘플이 같은 q를 받으므로 diversity가 잡음에서만 나와 값이 비슷하다. 코퍼스별 값은 서로 다르다(복사가 아니라 실제로 가까운 값).\n")

    # ================================================================ d. stop / boundary full data
    w("## d. τ 아래 생성 비율 (전체 데이터, 타깃 R) — 부록 표 A4(a), 표 4\n")
    w("| 코퍼스 | 구성 | 정지 (%) | 경계 (%) | 합계 = τ 아래 (%) |"); w("|---|---|---|---|---|")
    sums = {}
    for c in CORP:
        for lab, arm in (("F", "A"), ("FU", "C"), ("AC", "AC64"), ("QC", "QC64")):
            r = [V[("R", 1.0)][c][sd][arm] for sd in SEEDS]
            s, b = np.mean([x["stop_G"] for x in r]) * 100, np.mean([x["boundary_G"] for x in r]) * 100
            sums.setdefault(lab, []).append(s + b)
            w(f"| {SHORT[c]} | {lab} | {s:.2f} | {b:.2f} | {s + b:.2f} |")
            for sd, x in zip(SEEDS, r):
                for m in ("stop_G", "boundary_G"):
                    ROWS.append(["d_lowq", c, 1.0, sd, lab, m, x[m]])
    w("\n합계 범위(4코퍼스, 소수 둘째 자리): " + ", ".join(f"{k} {min(v):.2f}–{max(v):.2f}%" for k, v in sums.items()) + "\n")

    # ================================================================ e. aloha duplicate
    w("## e. 점검: aloha best-of-N, 학습량 맞춤 대 activity 75%\n")
    a = [V[("R", 1.0)]["aloha_tape"][sd]["A_step"]["best_of_N"] * 1e4 for sd in SEEDS]
    b = [V[("R", 1.0)]["aloha_tape"][sd]["A25"]["best_of_N"] * 1e4 for sd in SEEDS]
    w("| 구성 | seed 0 | 1 | 2 | 3 | 4 | 평균 ± 표준편차 (4자리) |"); w("|---|---|---|---|---|---|---|")
    w("| 학습량 맞춤 (A_step) | " + " | ".join(f"{x:.4f}" for x in a) + f" | {np.mean(a):.4f} ± {np.std(a, ddof=1):.4f} |")
    w("| activity 75% (A25) | " + " | ".join(f"{x:.4f}" for x in b) + f" | {np.mean(b):.4f} ± {np.std(b, ddof=1):.4f} |")
    w("\n두 구성의 seed별 값은 서로 다르고(출처: `results/<셀>.json`의 A_step, `filter_variants/<셀>.json`의 A25), 평균·표준편차가 3자리 반올림에서 우연히 같다. 행 대응 오류가 아니므로 판정 재계산이 필요 없다.\n")
    t = time.time() - t0
    w(f"## 소요 시간\n\n집계 {t:.0f}초(CPU, 셀 데이터 재구성 포함). 새 생성 없음.\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "aggregate_15b.md").write_text("\n".join(L), encoding="utf-8")
    with open(OUT / "per_seed_15b.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh); cw.writerow(["section", "corpus", "fraction", "seed", "config", "metric", "value"]); cw.writerows(ROWS)

    # ---- automatic comparison with report 14 (RQ3 per-cell mean differences) for the 'last digit' list
    txt = (D9 / "rq14" / "aggregate_14.md").read_text(encoding="utf-8")
    diffs = []
    for sc, kk in ((1.0, "비례 K=64"), (0.5, "비례 K=32"), (0.25, "비례 K=16"), (0.125, "비례 K=8")):
        sec = txt.split(f"데이터 비율 {FR[sc]}: QC − F")[1].split("**전체**")[0]
        for c in CORP:
            mrow = re.search(rf"\| {SHORT[c]} \| {re.escape(kk)} \| (.+) \|", sec)
            cols = [x.strip().split(" ")[0] for x in mrow.group(1).split("|")]
            for m, s in zip(mets, cols):
                mine = np.mean([V[("RQ3", sc)][c][sd]["QC64"][m] - V[("RQ3", sc)][c][sd]["A"][m] for sd in SEEDS])
                dec = len(s.split(".")[1]) if "." in s else 0
                if f"{mine:+.{dec}f}" != s:
                    diffs.append(f"{SHORT[c]} {FR[sc]} {m}: 보고 14 {s} / 재집계 {mine:+.{dec}f}")
    print("report-14 RQ3 cell mismatches:", diffs or "none", "| seconds", round(t))


if __name__ == "__main__":
    main()
