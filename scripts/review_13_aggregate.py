"""Report 13 confirmation on R (pre-registered criteria) using the report-11 statistics.

    .venv/Scripts/python.exe scripts/review_13_aggregate.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import review_11_aggregate as agg
from review_11_aggregate import CORP, SEEDS, MET, table, runlevel, run_pairs, fp

OUT = Path("outputs/campaign_dedup/rq13")


def main():
    agg.load()
    sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))["selected"]
    V = agg.V
    for c in CORP:
        for sd in SEEDS:
            r = json.loads((OUT / "confirm" / f"{c}_f1.00_s{sd}_{sel}.json").read_text(encoding="utf-8"))
            T = V[("CONF", 1.0)][c][sd]
            T["sel"], T["cur"] = r["arms"]["selected"], r["arms"]["current"]
            T["A"], T["QCs09"], T["B2"] = V[("R", 1.0)][c][sd]["A"], V[("R", 1.0)][c][sd]["QCs"], V[("R", 1.0)][c][sd]["B2"]
    L = []; w = L.append
    w(f"# 요청 개선 시험 — R 확인 (선택 = {sel}, 현재 = K32_w0)\n")
    w("셀 형식은 보고 11a와 같다. 선택·현재는 같은 난수 흐름으로 새로 생성했고, Filtered·QC-M은 보고 09 값이다.\n")
    table(w, "CONF", 1.0, "sel", "cur", mets=MET + ["lowq_G"], title="선택 − 현재 QC-S")
    table(w, "CONF", 1.0, "sel", "A", mets=MET + ["lowq_G"], title="선택 − Filtered")
    table(w, "CONF", 1.0, "cur", "A", mets=MET, title="참고 — 현재 QC-S(새 생성) − Filtered")
    table(w, "CONF", 1.0, "sel", "B2", mets=MET, title="참고 — 선택 − QC-M")
    table(w, "CONF", 1.0, "cur", "QCs09", mets=MET, title="점검 — 현재 QC-S(새 생성) − QC-S(보고 09): 같은 설정, 다른 잡음")
    # pre-registered criteria
    c1_pro = [m for m in ("precision", "best_of_N") if (rl := runlevel(run_pairs("CONF", 1.0, "sel", "cur", m), m))["sign_p"] < .05 and rl["k"] > rl["n"] / 2]
    c1_con = [m for m in MET if (rl := runlevel(run_pairs("CONF", 1.0, "sel", "cur", m), m))["sign_p"] < .05 and rl["k"] < rl["n"] / 2]
    c2 = {m: runlevel(run_pairs("CONF", 1.0, "sel", "A", m), m) for m in ("recall", "diversity")}
    ok2 = all(v["sign_p"] < .05 and v["k"] > v["n"] / 2 for v in c2.values())
    ok1 = bool(c1_pro) and not c1_con
    w("## 사전 등록 기준\n")
    w(f"1. 선택 − 현재: precision·best-of-N 중 선택 쪽 유의 = {c1_pro or '없음'}; 반대 방향 유의(4지표) = {c1_con or '없음'} → **{'충족' if ok1 else '미충족'}**")
    w(f"2. 선택 − Filtered: recall {c2['recall']['k']}/{c2['recall']['n']} (p={fp(c2['recall']['sign_p'])}), "
      f"diversity {c2['diversity']['k']}/{c2['diversity']['n']} (p={fp(c2['diversity']['sign_p'])}) → **{'충족' if ok2 else '미충족'}**")
    w(f"\n**판정: {'개선' if ok1 and ok2 else '개선 확인 안 됨'}**\n")
    (OUT / "confirm_aggregate.md").write_text("\n".join(L), encoding="utf-8")
    with open(OUT / "per_seed_long.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh); cw.writerow(["stage", "corpus", "seed", "arm", "metric", "value"])
        for p in sorted((OUT / "val").glob("*.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            for arm, ms in [("F", r["F"])] + list(r["cand"].items()):
                for m, v in ms.items():
                    cw.writerow(["val", r["corpus"], r["seed"], arm, m, v])
        for c in CORP:
            for sd in SEEDS:
                for arm in ("sel", "cur"):
                    for m, v in V[("CONF", 1.0)][c][sd][arm].items():
                        cw.writerow(["confirm", c, sd, arm, m, v])
    print("\n".join(L))


if __name__ == "__main__":
    main()
