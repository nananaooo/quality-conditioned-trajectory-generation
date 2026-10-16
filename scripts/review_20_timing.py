"""Request 20, item 5: generation time of the final QC (K = 64), full data, seed 0, saved checkpoints (no training).

Per query batch (B = 1 latency, B = 256 throughput), timed end to end:
  1. state standardisation (training-split statistics)
  2. NN_K over the training states (brute force, K = 64, as in the evaluation code) + quality-request draw from the usable
     neighbours (Eq. 3.6; fallback = usable pool Q)
  3. Euler integration, 50 steps (x1-prediction field)
  4. de-standardisation to joint units
One-off costs (model load, index build = standardised training states + usable mask + q table) are timed separately.
GPU (CUDA, synchronised) and CPU (same machine, torch default threads). Warm-up 20, then 200 (B = 1) / 50 (B = 256) reps;
median and IQR. Writes outputs/campaign_dedup/rq20/timing.md / timing.csv.

    .venv/Scripts/python.exe scripts/review_20_timing.py
"""
from __future__ import annotations

import csv
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from _bootstrap import add_project_root

add_project_root()

from src.eval.distribution_metrics import _pairwise_dist  # noqa: E402
from run_review_campaign import make_fm  # noqa: E402
from review_11_rq import Cell  # noqa: E402

OUT = Path("outputs/campaign_dedup/rq20")
K, STEPS = 64, 50


def cpu_name():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out or platform.processor()
    except Exception:
        return platform.processor()


def build_index(cell):
    return {"S": cell.nS(cell.s_tr), "ok": cell.ax.usable_tr, "q": cell.ax.q_tr, "Q": cell.ax.Q}


def request(idx, xs, rng):
    d = _pairwise_dist(xs, idx["S"])
    nbr = np.argpartition(d, K, axis=1)[:, :K]
    q = np.empty((len(xs), 2), np.float32)
    for i in range(len(xs)):
        u = nbr[i][idx["ok"][nbr[i]]]
        pool = idx["q"][u] if len(u) else idx["Q"]
        q[i] = pool[rng.integers(0, len(pool))]
    return q


@torch.no_grad()
def integrate(fm, cond, q):
    b = cond.size(0); x = torch.randn(b, fm.cfg.chunk_size, fm.cfg.action_dim, device=cond.device); dt = 1.0 / STEPS
    for i in range(STEPS):
        tv = i * dt
        x1 = fm(x, torch.full((b,), tv, device=cond.device), cond, q)
        x = x + (x1 - x) / max(1.0 - tv, 1e-3) * dt
    return x


def run(cell, fm, idx, states, device, B, reps, warm=20):
    rng = np.random.default_rng(0); sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    T = {"total": [], "norm_knn_q": [], "integrate": [], "denorm": []}
    for r in range(warm + reps):
        s = states[rng.integers(0, len(states), B)]
        sync(); t0 = time.perf_counter()
        xs = cell.nS(s); q = request(idx, xs, rng)
        cond = torch.from_numpy(xs).to(device); qt = torch.from_numpy(q).to(device)
        sync(); t1 = time.perf_counter()
        x = integrate(fm, cond, qt); sync(); t2 = time.perf_counter()
        out = cell.dA(x.cpu().numpy()); t3 = time.perf_counter()
        if r >= warm:
            T["total"].append(t3 - t0); T["norm_knn_q"].append(t1 - t0); T["integrate"].append(t2 - t1); T["denorm"].append(t3 - t2)
    return {k: np.array(v) for k, v in T.items()}


def q3(a):
    return np.percentile(a, [25, 50, 75])


def main():
    cc = yaml.safe_load(Path("configs/review_campaign.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(cc["base_config"]).read_text(encoding="utf-8"))
    L = []; w = L.append; rows = []
    cpu = cpu_name(); threads = torch.get_num_threads()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    w("# 요청 20-5 — 생성 시간 (QC, K = 64, 전체 데이터, seed 0, 저장 체크포인트)\n")
    w(f"환경: GPU {gpu}; CPU {cpu}, torch CPU 스레드 {threads} (os.cpu_count = {os.cpu_count()}); torch {torch.__version__}, Python {platform.python_version()}.\n")
    w("이웃 검색은 평가 코드와 같은 전수 비교(numpy, CPU)다. GPU 행에서도 1–2단계는 CPU에서 돈다.\n")
    for corpus in ("being_so101", "aloha_tape"):
        tag = f"{corpus}_f1.00_s0"
        t0 = time.perf_counter(); cell = Cell(corpus, 1.0, 0, cfg); t_data = time.perf_counter() - t0
        ckp = Path(cc["out_dir"]) / "ckpt" / tag / "B2.pt"
        for dev_name in ("cuda", "cpu"):
            device = torch.device(dev_name)
            t0 = time.perf_counter()
            ck = torch.load(ckp, map_location=device, weights_only=False)
            fm = make_fm(cfg, cell.D, 2).to(device); fm.load_state_dict(ck["state"]); fm.eval()
            if device.type == "cuda": torch.cuda.synchronize()
            t_load = time.perf_counter() - t0
            t0 = time.perf_counter(); idx = build_index(cell); t_idx = time.perf_counter() - t0
            states = cell.s_te
            nparam = sum(p.numel() for p in fm.parameters())
            for B, reps in ((1, 200), (256, 50)):
                T = run(cell, fm, idx, states, device, B, reps)
                tot = q3(T["total"]); share = {k: np.median(T[k]) / np.median(T["total"]) for k in ("norm_knn_q", "integrate", "denorm")}
                thr = B / T["total"]
                rows.append([corpus, dev_name, B, *tot, *q3(thr), share["norm_knn_q"], share["integrate"], share["denorm"], t_load, t_idx, nparam])
                w(f"- **{corpus} / {dev_name.upper()} / B = {B}**: 총 시간 중앙값 {tot[1]*1e3:.2f} ms (IQR {tot[0]*1e3:.2f}–{tot[2]*1e3:.2f}); "
                  f"처리량 중앙값 {np.median(thr):,.0f} chunk/s (IQR {q3(thr)[0]:,.0f}–{q3(thr)[2]:,.0f}); "
                  f"단계 비율(중앙값 기준) 표준화+이웃 검색+q {share['norm_knn_q']*100:.1f}%, 적분 {share['integrate']*100:.1f}%, 되돌리기 {share['denorm']*100:.1f}%")
            w(f"  - 일회성: 모델 로드 {t_load*1e3:.0f} ms, 색인 구축 {t_idx*1e3:.1f} ms (학습 상태 {len(cell.s_tr):,}개), 데이터 재구성 {t_data:.1f} s")
        size_mb = ckp.stat().st_size / 2 ** 20
        sd_mb = sum(v.numel() * v.element_size() for v in ck["state"].values()) / 2 ** 20
        w(f"  - 파라미터 {nparam:,}개; 체크포인트 파일 {size_mb:.2f} MB (가중치만 {sd_mb:.2f} MB, 나머지는 정규화 통계·학습 곡선)\n")
    w("| 코퍼스 | 장치 | B | 지연 중앙값 (ms) | IQR (ms) | 처리량 중앙값 (chunk/s) | 이웃+q 비율 | 적분 비율 |")
    w("|---|---|---|---|---|---|---|---|")
    for r in rows:
        w(f"| {r[0]} | {r[1].upper()} | {r[2]} | {r[4]*1e3:.2f} | {r[3]*1e3:.2f}–{r[5]*1e3:.2f} | {r[7]:,.0f} | {r[9]*100:.1f}% | {r[10]*100:.1f}% |")
    (OUT / "timing.md").write_text("\n".join(L), encoding="utf-8")
    with open(OUT / "timing.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh)
        cw.writerow(["corpus", "device", "batch", "total_q1_s", "total_median_s", "total_q3_s", "thr_q1", "thr_median", "thr_q3",
                     "share_norm_knn_q", "share_integrate", "share_denorm", "model_load_s", "index_build_s", "n_params"])
        cw.writerows(rows)
    print("\n".join(L))


if __name__ == "__main__":
    main()
