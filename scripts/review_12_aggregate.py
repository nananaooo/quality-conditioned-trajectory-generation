"""Report 12 aggregation. Reuses the report-11 loaders and statistics (review_11_aggregate): paired t, Holm over the
4-corpus family, run-level exact sign test + Wilcoxon on oriented differences (positive = first arm better; ratio
metrics such as tremor / below-tau fractions: other − first). Writes outputs/campaign_dedup/rq12/aggregate_12.md and
per_seed_long.csv.

    .venv/Scripts/python.exe scripts/review_12_aggregate.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import review_11_aggregate as agg
from review_11_aggregate import CORP, SEEDS, SHORT, MET, table, runlevel, run_pairs, cells, fp

RQ = Path("outputs/campaign_dedup/rq12")
agg.LOWER.update({"tremor_G", "tremor_div", "below_tau_G", "below_tau_div", "stop_G", "boundary_G"})
V = agg.V
C = {}


def load():
    agg.load()
    for c in CORP:
        for sd in SEEDS:
            r = json.loads((RQ / "cells" / f"{c}_f1.00_s{sd}.json").read_text(encoding="utf-8")); C[(c, sd)] = r
            for key, tgt in (("B1_25", "R2"), ("B1_50", "R2_50")):
                for arm, m in r[key]["arms"].items():
                    V[(tgt, 1.0)][c][sd][arm] = dict(m)
            R = V[("R", 1.0)][c][sd]
            con = V[("CON", 1.0)][c][sd]
            con["QCs_con"] = r["B2"]["QCs_conpool"]
            for a in ("QCs", "A_con", "A25", "A50", "B2"):
                con[a] = R[f"con:{a}"]
            for a, m in r["below_tau"]["arms"].items():
                R[a].update(m)
            R["QCs_chk"] = r["check"]["QCs_checked"]; R["OF_rerun"] = r["check"]["OF_rerun"]


def mean_ci(vals):
    v = np.asarray(vals, float)
    from scipy import stats
    return v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def decision(tgt):
    """Pre-fixed rule (request 12, 4.3) on the run-level sign test of QC-S − AC-S."""
    pro, con_ = [], []
    for m in ("recall", "precision", "diversity", "best_of_N", "tremor_G"):
        rl = runlevel(run_pairs(tgt, 1.0, "QCs", "ACs", m), m)
        if rl["sign_p"] < .05 and rl["k"] > rl["n"] / 2 and m in ("recall", "precision", "tremor_G"):
            pro.append(f"{m} ({rl['k']}/{rl['n']}, p={fp(rl['sign_p'])})")
        if rl["sign_p"] < .05 and rl["k"] < rl["n"] / 2:
            con_.append(f"{m} ({rl['k']}/{rl['n']}, p={fp(rl['sign_p'])})")
    verdict = "smoothness 기여 있음" if pro and not con_ else "기여 확인 안 됨"
    return verdict, pro, con_


def report():
    L = []; w = L.append
    w("# 집계 12 — B1(두 축 타깃) · B2(consistency 요청 풀) · 저품질 생성 분석 (전체 데이터, 학습 없음)\n")
    w("셀: 평균 차이(앞 − 뒤) ± t 95% CI 반폭, \\* p < 0.05, † Holm(4코퍼스) p < 0.05, (k/n) = 앞 구성이 나은 seed 수. "
      "전체 행: 앞 구성이 나은 실행 수/동률 제외 n, 양측 정확 부호 검정 p, Wilcoxon p. best_of_N·비율 지표는 낮을수록 좋다.\n")

    # ---------------- checks
    w("## 시작 전 확인 (요청 3절)\n")
    w("### a. R2 (두 축 usable held-out 고유 chunk) 수\n")
    w("| 코퍼스 | 25% 기준: 가용 수 (seed 0–4) | 사용 수 | 50% 기준: 가용 수 | 사용 수 |")
    w("|---|---|---|---|---|")
    for c in CORP:
        r = [C[(c, sd)] for sd in SEEDS]
        w(f"| {SHORT[c]} | {', '.join(str(x['B1_25']['n_R2_available']) for x in r)} | {', '.join(str(x['B1_25']['n_R2']) for x in r)} | "
          f"{', '.join(str(x['B1_50']['n_R2_available']) for x in r)} | {', '.join(str(x['B1_50']['n_R2']) for x in r)} |")
    w("")
    w("### b. 떨림 비율과 구간 임계\n")
    w("| 코퍼스 | 학습 usable 중 떨림 (25% / 50%) | held-out usable(고유) 중 떨림 (25% / 50%) |")
    w("|---|---|---|")
    for c in CORP:
        r = [C[(c, sd)] for sd in SEEDS]
        f = lambda k, kk: np.mean([x[k][kk] for x in r])
        w(f"| {SHORT[c]} | {f('B1_25', 'train_tremor_among_usable'):.3f} / {f('B1_50', 'train_tremor_among_usable'):.3f} | "
          f"{f('B1_25', 'heldout_tremor_among_usable_unique'):.3f} / {f('B1_50', 'heldout_tremor_among_usable_unique'):.3f} |")
    w("\n구간별 임계 예시 (seed 0, 25% 기준). 구간 = 학습 usable activity 10분위(원 단위 activity 경계), 임계 = 그 구간 smoothness z 하위 25%:\n")
    for c in CORP:
        b = C[(c, 0)]["B1_25"]
        w(f"- {SHORT[c]}: 경계 " + ", ".join(f"{e:.3g}" for e in b["edges"]) + " | 임계 " + ", ".join(f"{t:+.2f}" for t in b["thr"]))
    w("")
    w("### c. 학습 요청 풀의 z 평균 (activity z / smoothness z)\n")
    w("| 코퍼스 | activity-usable 풀 | 두 축 usable 풀 (25%) | 두 축 usable 풀 (50%) |")
    w("|---|---|---|---|")
    for c in CORP:
        r = [C[(c, sd)] for sd in SEEDS]
        g = lambda k, kk: np.mean([x[k]["pool_means"][kk] for x in r], 0)
        a, t25, t50 = g("B1_25", "act_pool"), g("B1_25", "two_axis_pool"), g("B1_50", "two_axis_pool")
        w(f"| {SHORT[c]} | {a[0]:+.3f} / {a[1]:+.3f} | {t25[0]:+.3f} / {t25[1]:+.3f} | {t50[0]:+.3f} / {t50[1]:+.3f} |")
    w("")
    w("### d. 요청 풀 fallback 비율 (셀 평균)\n")
    w("| 코퍼스 | 기존 activity 풀 (타깃 R) | B1 두 축 풀 25% (R2) | B1 50% | consistency 타깃: 기존 activity 풀 (11a) | B2 consistency 풀 |")
    w("|---|---|---|---|---|---|")
    for c in CORP:
        r = [C[(c, sd)] for sd in SEEDS]
        fbR = np.mean([V[("R", 1.0)][c][sd]["_meta"]["fallback"] for sd in SEEDS])
        w(f"| {SHORT[c]} | {fbR:.3f} | {np.mean([x['B1_25']['fallback'] for x in r]):.3f} | {np.mean([x['B1_50']['fallback'] for x in r]):.3f} | "
          f"{np.mean([x['B2']['fallback_actpool_11a'] for x in r]):.3f} | {np.mean([x['B2']['fallback'] for x in r]):.3f} |")
    w("")
    w("consistency 50% 필터가 남기는 학습 chunk 중 activity-usable 비율: " + ", ".join(
        f"{SHORT[c]} {np.mean([C[(c, sd)]['B2']['con_pool_act_usable_frac'] for sd in SEEDS]):.3f}" for c in CORP) + "\n")
    w("### e. 소요 시간\n")
    w(f"20셀 합계 {sum(C[k]['seconds'] for k in C) / 60:.1f}분(GPU).\n")

    # ---------------- B1
    w("## B1 — 두 축 타깃 R2 (떨림 = 같은 activity 구간에서 smoothness 하위 25%)\n")
    w("⚠ aloha는 이전 RQ2 분석에서 smoothness 단독 제어가 되지 않았다(제어 상관 0.000). 값은 그대로 싣는다.\n")
    mets = MET + ["tremor_G"]
    table(w, "R2", 1.0, "QCs", "ACs", mets=mets, title="QC-S − AC-S (주 비교, R2)")
    verdict, pro, con_ = decision("R2")
    w(f"**사전 고정 판정 (4.3): {verdict}.** QC-S 쪽 유의(recall·precision·떨림 비율 중): {', '.join(pro) or '없음'}. "
      f"반대 방향 유의(4지표 + 떨림 비율): {', '.join(con_) or '없음'}.\n")
    table(w, "R", 1.0, "QCs", "ACs", mets=MET, title="참고 — 기존 타깃 R에서의 QC-S − AC-S (11a 값)")
    table(w, "R2", 1.0, "QCs", "A", mets=mets, title="참고 — QC-S − Filtered (R2)")
    table(w, "R2", 1.0, "ACs", "A", mets=mets, title="참고 — AC-S − Filtered (R2)")
    w("**생성 chunk 부류 비율 (G, 평균 ± 95% CI 반폭)** — (1) activity < τ, (2) 떨림. 참고: held-out activity-usable 고유 chunk 중 떨림 비율\n")
    w("| 코퍼스 | Filtered (1) / (2) | AC-S (1) / (2) | QC-S (1) / (2) | held-out 떨림 |")
    w("|---|---|---|---|---|")
    for c in CORP:
        row = []
        for a in ("A", "ACs", "QCs"):
            m1, h1 = mean_ci([V[("R2", 1.0)][c][sd][a]["below_tau_G"] for sd in SEEDS])
            m2, h2 = mean_ci([V[("R2", 1.0)][c][sd][a]["tremor_G"] for sd in SEEDS])
            row.append(f"{m1*100:.1f} ± {h1*100:.1f}% / {m2*100:.1f} ± {h2*100:.1f}%")
        row.append(f"{np.mean([C[(c, sd)]['B1_25']['heldout_tremor_among_usable_unique'] for sd in SEEDS])*100:.1f}%")
        w(f"| {SHORT[c]} | " + " | ".join(row) + " |")
    w("\n떨림 비율의 분모는 생성 chunk 전체다(activity < τ 포함). diversity 집합에서의 떨림 비율은 CSV(`tremor_div`).\n")
    w("**activity·smoothness z 평균 (G)**\n")
    w("| 코퍼스 | Filtered | AC-S | QC-S | R2 |")
    w("|---|---|---|---|---|")
    for c in CORP:
        row = [f"{np.mean([V[('R2', 1.0)][c][sd][a]['zact_G'] for sd in SEEDS]):+.2f} / {np.mean([V[('R2', 1.0)][c][sd][a]['zsmooth_G'] for sd in SEEDS]):+.2f}"
               for a in ("A", "ACs", "QCs")]
        rz = np.mean([C[(c, sd)]["B1_25"]["R2_z"] for sd in SEEDS], 0)
        row.append(f"{rz[0]:+.2f} / {rz[1]:+.2f}")
        w(f"| {SHORT[c]} | " + " | ".join(row) + " |")
    w("")

    w("### 보조 — 하위 50% 기준 (요약)\n")
    w("| 비교 | 지표 | 유의 칸 (QC-S 우위/열세) | Holm | 실행 단위 (QC-S가 나은 수/n, 부호 p, Wilcoxon p) |")
    w("|---|---|---|---|---|")
    for x, y in (("QCs", "ACs"), ("QCs", "A")):
        for m in mets:
            cs, g, b, gh, bh, lo, hi = agg.sig_summary("R2_50", 1.0, x, y, m)
            rl = runlevel(run_pairs("R2_50", 1.0, x, y, m), m)
            w(f"| {'QC-S − AC-S' if y == 'ACs' else 'QC-S − Filtered'} | {m} | {g}/{b} | {gh}/{bh} | {rl['k']}/{rl['n']}, {fp(rl['sign_p'])}, {fp(rl['wil_p'])} |")
    v50, pro50, con50 = decision("R2_50")
    w(f"\n같은 규칙을 50% 기준에 적용하면(참고, 판정에 쓰지 않음): {v50}. QC-S 쪽: {', '.join(pro50) or '없음'}; 반대: {', '.join(con50) or '없음'}.\n")
    w("떨림 비율 (G) 평균, 50% 기준: " + "; ".join(
        f"{SHORT[c]} F {np.mean([V[('R2_50', 1.0)][c][sd]['A']['tremor_G'] for sd in SEEDS])*100:.1f}% / AC-S "
        f"{np.mean([V[('R2_50', 1.0)][c][sd]['ACs']['tremor_G'] for sd in SEEDS])*100:.1f}% / QC-S "
        f"{np.mean([V[('R2_50', 1.0)][c][sd]['QCs']['tremor_G'] for sd in SEEDS])*100:.1f}%" for c in CORP) + "\n")

    # ---------------- B2
    w("## B2 — consistency 타깃, 요청 풀을 consistency 기준으로 교체 (재학습 없음)\n")
    for f, nm in (("A_con", "consistency 50% 필터"), ("A25", "activity 75% 필터"), ("A50", "Filtered")):
        table(w, "CON", 1.0, "QCs_con", f, title=f"QC-S(consistency 풀) − {nm}")
        table(w, "CON", 1.0, "QCs", f, title=f"기존(11a) QC-S(activity 풀) − {nm}")
    table(w, "CON", 1.0, "QCs_con", "QCs", title="QC-S(consistency 풀) − QC-S(activity 풀)")

    # ---------------- 6.1
    w("## 6.1 τ 아래 생성물의 위치 (타깃 R, G 1,500)\n")
    w("정지 수준 = activity < 학습 풀 p25, 경계 = p25 ≤ activity < τ(p50). 표본은 보고 09·11a 저장 샘플 그대로.\n")
    w("| 코퍼스 | Filtered 정지 / 경계 | FU | AC-S | QC-S | held-out 전체 chunk |")
    w("|---|---|---|---|---|---|")
    for c in CORP:
        row = []
        for a in ("A", "C", "ACs", "QCs"):
            s, hs = mean_ci([V[("R", 1.0)][c][sd][a]["stop_G"] for sd in SEEDS])
            b, hb = mean_ci([V[("R", 1.0)][c][sd][a]["boundary_G"] for sd in SEEDS])
            row.append(f"{s*100:.2f} ± {hs*100:.2f}% / {b*100:.1f} ± {hb*100:.1f}%")
        h = [C[(c, sd)]["below_tau"]["heldout"] for sd in SEEDS]
        row.append(f"{np.mean([x['stop'] for x in h])*100:.1f}% / {np.mean([x['boundary'] for x in h])*100:.1f}%")
        w(f"| {SHORT[c]} | " + " | ".join(row) + " |")
    w("")
    table(w, "R", 1.0, "QCs", "A", mets=["stop_G", "boundary_G", "lowq_G"], title="QC-S − Filtered: 정지·경계·τ 아래 비율 (음수 = QC-S가 적음)")
    table(w, "R", 1.0, "QCs", "C", mets=["stop_G", "boundary_G"], title="참고 — QC-S − FU")

    # ---------------- 6.2
    w("## 6.2 생성 후 품질 확인: QC-S + OF 규칙 (타깃 R)\n")
    w("규칙: activity < τ이면 잡음과 q(같은 상태 이웃 풀)를 다시 뽑는다. 상태당 최대 64회, 끝까지 통과 못한 상태는 제외. "
      "G와 diversity 집합(256 × 8) 모두. OF는 같은 계측 코드로 Full-Unconditioned에서 다시 돌렸다.\n")
    w("| 코퍼스 | 구성 | 재추출이 필요했던 G 샘플 비율 | 통과 샘플당 추출 수 (G) | 제외 상태 (G, /1,500) | 통과 샘플당 추출 수 (div) | 제외 상태 (div, /256) |")
    w("|---|---|---|---|---|---|---|")
    for c in CORP:
        for a, nm in (("QCs_chk", "QC-S + 확인"), ("OF_rerun", "OF")):
            r = [V[("R", 1.0)][c][sd][a] for sd in SEEDS]
            w(f"| {SHORT[c]} | {nm} | {np.mean([x['G_frac_resampled'] for x in r])*100:.1f}% | {np.mean([x['G_draws_per_accepted'] for x in r]):.3f} | "
              f"{np.mean([x['G_states_dropped'] for x in r]):.1f} (최대 {max(x['G_states_dropped'] for x in r)}) | "
              f"{np.mean([x['div_draws_per_accepted'] for x in r]):.3f} | {np.mean([x['div_states_dropped'] for x in r]):.1f} |")
    w("")
    table(w, "R", 1.0, "QCs_chk", "QCs", title="(QC-S + 확인) − QC-S")
    table(w, "R", 1.0, "QCs_chk", "A", title="(QC-S + 확인) − Filtered")
    table(w, "R", 1.0, "QCs_chk", "OutFilt", title="(QC-S + 확인) − OF (보고 09 값)")
    table(w, "R", 1.0, "OF_rerun", "OutFilt", title="점검 — OF 재실행 − OF(보고 09): 같은 규칙, 다른 잡음")
    (RQ / "aggregate_12.md").write_text("\n".join(L), encoding="utf-8")


def write_csv():
    with open(RQ / "per_seed_long.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh); cw.writerow(["corpus", "target", "seed", "arm", "metric", "value"])
        for tgt in ("R2", "R2_50", "CON", "R"):
            for c in CORP:
                for sd in SEEDS:
                    for arm, ms in V[(tgt, 1.0)][c][sd].items():
                        if arm.startswith("_") or not isinstance(ms, dict) or (tgt == "R" and arm.startswith("con:")):
                            continue
                        for m, v in ms.items():
                            if isinstance(v, (int, float)):
                                cw.writerow([c, tgt, sd, arm, m, v])


if __name__ == "__main__":
    load(); report(); write_csv()
    print("wrote", RQ / "aggregate_12.md")
