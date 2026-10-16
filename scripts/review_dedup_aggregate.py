"""Aggregate the de-duplicated re-evaluation (outputs/campaign_dedup/results) into the v12 Table 5, 6, A1
quantities, with the previous (duplicate-target) campaign values alongside. Table 7 / Figure 2 come from
review_tradeoff_aggregate.py with TRADEOFF_DIR=outputs/campaign_dedup/tradeoff.

Paired t-test over 5 seeds (df 4), t 95% CI, Holm over the 8-cell family per (contrast, metric).
    .venv/Scripts/python.exe scripts/review_dedup_aggregate.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

NEW, OLD = Path("outputs/campaign_dedup/results"), Path("outputs/campaign/results")
CORP = ["being_so101", "rayenghali", "koch_pickplace5", "aloha_tape"]
SHORT = {"being_so101": "being", "rayenghali": "rayen", "koch_pickplace5": "koch", "aloha_tape": "aloha"}
SCALES = [1.0, 0.25]
CELLS = [(c, s) for c in CORP for s in SCALES]
MET = ["recall", "precision", "diversity", "best_of_N"]
LOWER = {"best_of_N"}


def load(d):
    R = defaultdict(dict)
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8")); R[(r["corpus"], r["scale"])][r["seed"]] = r
    return R


def val(r, arm, m):
    if arm.startswith("mode:"):
        return r["q_modes"]["B2"][arm[5:]][m]
    return r["arms"][arm]["main"][m]


def paired(d, m):
    d = np.asarray(d, float); n = len(d)
    hw = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    p = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else (0.0 if d.mean() != 0 else 1.0)
    better = (d < 0) if m in LOWER else (d > 0)
    return dict(mean=float(d.mean()), hw=float(hw), p=float(p), k=int(better.sum()))


def holm(ps):
    ps = np.nan_to_num(np.asarray(ps, float), nan=1.0); o = np.argsort(ps); M = len(ps); adj = np.empty(M); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (M - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def contrast(R, x, y, m):
    out = [paired([val(R[cs][sd], x, m) - val(R[cs][sd], y, m) for sd in sorted(R[cs])], m) for cs in CELLS]
    for st, a in zip(out, holm([st["p"] for st in out])):
        st["ph"] = float(a)
    return out


def prec(c, m):
    return 4 if c == "aloha_tape" and m in ("diversity", "best_of_N") else 3


def main():
    N, O = load(NEW), load(OLD)
    assert all(len(N[cs]) == 5 for cs in CELLS), {cs: len(N[cs]) for cs in CELLS}
    L = []; w = L.append
    w("# 중복 제거 재평가 집계 (표 5·6·A1)\n")
    ex = next(iter(N[CELLS[0]].values()))
    w("평가 타깃 = held-out usable 중 고유 (상태, chunk)에서 1,500개. 셀별 고유 비율: " + ", ".join(
        f"{SHORT[c]} {s}: {np.mean([R['heldout_usable_unique']['raw'] / R['heldout_usable']['raw'] for R in N[(c, s)].values()]):.3f}"
        for c, s in CELLS) + "\n")

    # ---------------- Table 5
    w("## 표 5 — 네 구성 평균 (T = 1) 과 Δ = QC-M − Filtered  [괄호: 기존 중복 포함 값]\n")
    for sc in SCALES:
        w(f"### 규모 {sc}\n")
        w("| 지표 | 코퍼스 | Filtered | Full-Uncond | Activity-Cond | QC-M | Δ (p, k/5, Holm) | 기존 Δ |")
        w("|---|---|---|---|---|---|---|---|")
        for m in MET:
            con_n = contrast(N, "B2", "A", m); con_o = contrast(O, "B2", "A", m)
            for c in CORP:
                i = CELLS.index((c, sc)); R = N[(c, sc)]; P = prec(c, m)
                means = [np.mean([val(R[sd], a, m) for sd in R]) for a in ("A", "C", "B1", "B2")]
                st, so = con_n[i], con_o[i]
                star = ("\\*" if st["p"] < .05 else "") + ("†" if st["ph"] < .05 else "")
                w(f"| {m} | {SHORT[c]} | " + " | ".join(f"{v:.{P}f}" for v in means)
                  + f" | {st['mean']:+.{P}f}{star} (p={st['p']:.3f}, {st['k']}/5) | {so['mean']:+.{P}f}{'*' if so['p'] < .05 else ''} |")
        w("")
    ins = [np.mean([val(N[cs][sd], a, "inside") for sd in N[cs]]) for cs in CELLS for a in ("A", "C", "B1", "B2")]
    w(f"관절 범위 유효성(inside) 범위, 네 구성 × 8칸: {min(ins)*100:.1f}–{max(ins)*100:.1f}%\n")
    ratio = [np.mean([val(N[cs][sd], 'B2', 'diversity') for sd in N[cs]]) / np.mean([val(N[cs][sd], 'A', 'diversity') for sd in N[cs]])
             for cs in CELLS if cs[0] != "aloha_tape"]
    w(f"diversity 배율 QC-M/Filtered (단일 팔 6칸): {min(ratio):.2f}–{max(ratio):.2f}\n")

    # ---------------- Appendix A1 contrasts
    w("## 대조 요약 (부록 표 A1 용): 유의 칸 수 p<0.05 / Holm, 평균 범위\n")
    pairs = [("C", "A", "Full-Uncond − Filtered"), ("OutFilt", "A", "Output-Filtered − Filtered"),
             ("B2", "C", "QC-M − Full-Uncond"), ("B2", "OutFilt", "QC-M − Output-Filtered"),
             ("B2", "B1", "QC-M − Activity-Cond"), ("B2", "A_step", "QC-M − 업데이트 맞춤 Filtered"),
             ("A_step", "A", "업데이트 맞춤 − Filtered"), ("B2_z", "A_z", "z축 QC-M − z축 Filtered"),
             ("QCs", "A", "QC-S − Filtered (T=1)"), ("B2", "A", "QC-M − Filtered")]
    w("| 대조 | 지표 | 유의(+)/유의(−) p | Holm (+/−) | 평균 범위 | 기존 유의 (+/−) |")
    w("|---|---|---|---|---|---|")
    for x, y, name in pairs:
        for m in MET:
            cn = contrast(N, x, y, m)
            try:
                co = contrast(O, x, y, m)
            except KeyError:
                co = None
            def cnt(cc, key):
                good = sum(1 for st in cc if st[key] < .05 and ((st["mean"] < 0) if m in LOWER else (st["mean"] > 0)))
                bad = sum(1 for st in cc if st[key] < .05 and ((st["mean"] > 0) if m in LOWER else (st["mean"] < 0)))
                return f"{good}/{bad}"
            rng_ = [st["mean"] for st in cn]
            P = 4 if m in ("diversity", "best_of_N") else 3
            w(f"| {name} | {m} | {cnt(cn, 'p')} | {cnt(cn, 'ph')} | {min(rng_):+.{P}f} … {max(rng_):+.{P}f} | {cnt(co, 'p') if co else '—'} |")
    w("\n(+) = 앞 구성이 좋음. best_of_N은 낮을수록 좋음.\n")

    # Output-Filtered per cell (v12 point 4)
    w("## Output-Filtered − Filtered 칸별\n")
    w("| 코퍼스 | 규모 | recall | precision | diversity | 통과 실패 상태 |")
    w("|---|---|---|---|---|---|")
    cons = {m: contrast(N, "OutFilt", "A", m) for m in ("recall", "precision", "diversity")}
    for i, (c, s) in enumerate(CELLS):
        cells = []
        for m in ("recall", "precision", "diversity"):
            st = cons[m][i]; P = prec(c, m)
            cells.append(f"{st['mean']:+.{P}f}{'*' if st['p'] < .05 else ''} ({st['k']}/5)")
        fail = np.mean([N[(c, s)][sd]["arms"]["OutFilt"]["main"]["frac_states_no_accept"] for sd in N[(c, s)]])
        w(f"| {SHORT[c]} | {s} | " + " | ".join(cells) + f" | {fail*100:.1f}% |")
    w("")

    # ---------------- Table 6
    w("## 표 6 — q 추출 방식, 분포 추출(QC-M) 대비 (8칸 범위, 유의 칸 수)\n")
    w("| 방식 | recall | precision | diversity (상대) | best_of_N |")
    w("|---|---|---|---|---|")
    for mode, name in (("per_state_fixed", "상태별 고정"), ("global_median", "전역 고정"), ("state_conditional", "상태 조건부 (QC-S)")):
        row = []
        for m in ("recall", "precision"):
            cn = contrast(N, f"mode:{mode}", "mode:default", m)
            ms = [st["mean"] for st in cn]; sig = [st for st in cn if st["p"] < .05]
            dirs = "모두 감소" if sig and all(st["mean"] < 0 for st in sig) else ("모두 증가" if sig and all(st["mean"] > 0 for st in sig) else "")
            row.append(f"{min(ms):+.3f} ~ {max(ms):+.3f} ({len(sig)}{', ' + dirs if dirs else ''})")
        rel = []; nsig = 0
        for cs in CELLS:
            d = [val(N[cs][sd], f"mode:{mode}", "diversity") / val(N[cs][sd], "mode:default", "diversity") - 1 for sd in N[cs]]
            rel.append(np.mean(d))
        cn = contrast(N, f"mode:{mode}", "mode:default", "diversity"); nsig = sum(st["p"] < .05 for st in cn)
        row.append(f"{min(rel)*100:+.0f} ~ {max(rel)*100:+.0f}% ({nsig})")
        cn = contrast(N, f"mode:{mode}", "mode:default", "best_of_N")
        worse = sum(st["p"] < .05 and st["mean"] > 0 for st in cn); better = sum(st["p"] < .05 and st["mean"] < 0 for st in cn)
        row.append(f"악화 {worse} / 개선 {better}")
        w(f"| {name} | " + " | ".join(row) + " |")
    w("")
    # precision recovery
    rec = []
    for cs in CELLS:
        R = N[cs]
        f = np.mean([val(R[sd], "A", "precision") for sd in R]); qm = np.mean([val(R[sd], "B2", "precision") for sd in R])
        qs = np.mean([val(R[sd], "QCs", "precision") for sd in R])
        rec.append((qs - qm) / (f - qm) if f > qm else np.nan)
    w("QC-S의 precision 회복률 (QC-S − QC-M)/(F − QC-M), 칸별: " + ", ".join(f"{SHORT[c]} {s}: {r*100:.0f}%" for (c, s), r in zip(CELLS, rec)) + "\n")
    cn = contrast(N, "B2", "A", "precision")
    w("표 5의 precision 열세 범위(QC-M − F): " + f"{min(st['mean'] for st in cn):+.3f} ~ {max(st['mean'] for st in cn):+.3f}\n")
    fb = [np.mean([R["state_cond_fallback_frac"] for R in N[cs].values()]) for cs in CELLS]
    w(f"상태 조건부 대체 비율: {min(fb):.3f}–{max(fb):.3f}\n")

    out = Path("outputs/campaign_dedup/aggregate_dedup.md"); out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
