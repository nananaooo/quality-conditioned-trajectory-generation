"""Report 13 selection (pre-registered rule): per run rank the 8 candidates on recall (desc), precision (desc),
diversity (desc), best-of-N (asc); mean of the 4 ranks, averaged over the 20 runs; minimum wins, ties -> closer to
K32_w0. Writes outputs/campaign_dedup/rq13/selection.md and selection.json.

    .venv/Scripts/python.exe scripts/review_13_select.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

OUT = Path("outputs/campaign_dedup/rq13")
MET = [("recall", -1), ("precision", -1), ("diversity", -1), ("best_of_N", 1)]


def main():
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "val").glob("*.json"))]
    names = list(runs[0]["cand"])
    R = np.zeros((len(runs), len(names)))
    per_metric = {m: np.zeros((len(runs), len(names))) for m, _ in MET}
    for i, r in enumerate(runs):
        rk = []
        for m, sgn in MET:
            x = rankdata([sgn * r["cand"][n][m] for n in names]); per_metric[m][i] = x; rk.append(x)
        R[i] = np.mean(rk, 0)
    score = R.mean(0)
    dist = {n: abs(np.log2(int(n.split("_")[0][1:])) - 5) + int(n[-1]) for n in names}
    order = sorted(names, key=lambda n: (round(score[names.index(n)], 9), dist[n]))
    sel = order[0]
    L = ["# 요청 개선 시험 — 검증 분할 선택 결과 (사전 등록 규칙)\n",
         f"검증 실행 {len(runs)}회(4코퍼스 × 5 seed, 1/2 모델, 1/2 부분집합 밖 학습 에피소드). 순위 1 = 최선.\n",
         "| 후보 | 평균 순위 (4지표) | recall 순위 | precision 순위 | diversity 순위 | best-of-N 순위 | Δrecall vs 현재 | Δprecision | Δdiversity(상대) | ΔBoN(상대) | fallback |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    cur = "K32_w0"
    for n in order:
        j = names.index(n)
        d = lambda m: np.mean([r["cand"][n][m] - r["cand"][cur][m] for r in runs])
        rel = lambda m: np.mean([r["cand"][n][m] / r["cand"][cur][m] - 1 for r in runs])
        L.append(f"| {n}{' ← 현재' if n == cur else ''}{' ← 선택' if n == sel else ''} | {score[j]:.3f} | "
                 + " | ".join(f"{per_metric[m][:, j].mean():.2f}" for m, _ in MET)
                 + f" | {d('recall'):+.4f} | {d('precision'):+.4f} | {rel('diversity')*100:+.1f}% | {rel('best_of_N')*100:+.1f}% | "
                 f"{np.mean([r['cand'][n]['fallback'] for r in runs]):.3f} |")
    L.append(f"\n선택: **{sel}** (K_full = {sel.split('_')[0][1:]}, {'거리 가중' if sel.endswith('1') else '균등'}).\n")
    L.append("검증에서 Filtered 대비 (참고, 현재 K32_w0): recall " + f"{np.mean([r['cand'][cur]['recall'] - r['F']['recall'] for r in runs]):+.4f}, "
             f"precision {np.mean([r['cand'][cur]['precision'] - r['F']['precision'] for r in runs]):+.4f}\n")
    (OUT / "selection.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "selection.json").write_text(json.dumps({"selected": sel, "score": dict(zip(names, score.tolist()))}), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
