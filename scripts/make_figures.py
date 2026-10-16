"""v20 figures from saved per-seed results only (no generation, no retraining).

Figure 2 (fig2_tradeoff_v20): temperature trade-off, full data, target R. Curves: Filtered (activity top 50%) and QC (K = 64).
  4 corpora (columns) x 2 rows (top precision-recall, bottom diversity-precision). T = 0.5 ... 3.0, T = 1 emphasised,
  5-seed mean, error bars = standard error, aloha diversity axis logarithmic.
  Sources: F = outputs/campaign_dedup/tradeoff/<cell>.json (report 09), QC = outputs/campaign_dedup/rq14/full/<cell>.json.
Figure 3 (fig3_scale_v20): data fraction 1, 1/2, 1/4, 1/8 on the common target R. F and QC (K proportional 64/32/16/8).
  4 corpora x 2 rows (recall, diversity), log2 x axis, 5-seed mean with t 95% CI band, aloha diversity axis logarithmic.
  Sources: F = report 09 results (1) and rq11/common (1/2, 1/4, 1/8); QC = rq14/full (1) and rq14/scales.
Drawn at final print size (4.64 in wide); labels 8 pt, ticks 7 pt. No figure title (caption carries it). Existing files are never overwritten.

    python scripts/make_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

D9 = Path("outputs/campaign_dedup"); OUT = Path("figures"); OUT.mkdir(exist_ok=True)
CORP = [("being_so101", "being_so101"), ("rayenghali", "rayenghali"), ("koch_pickplace5", "koch"), ("aloha_tape", "aloha")]
SEEDS = range(5)
TEMPS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
SCALES = [1.0, 0.5, 0.25, 0.125]
TAG = {1.0: "1.00", 0.5: "0.50", 0.25: "0.25", 0.125: "0.125"}
F_STYLE = ("F", "#2a78d6", "o")
QC_STYLE = ("QC", "#1baf7a", "^")
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
W_IN = 4.64            # r2: final print width (JWE River text width 334 pt); figures are drawn at final size
FS = 8                 # axis labels, corpus titles, legend
FS_TICK = 7            # tick labels
LW, MS, MS_T1, ELW = 1.0, 3.5, 5.5, 0.6   # line width, marker size, T = 1 marker size (pt), error-bar width
H2, H3 = 2.8, 2.7      # figure heights (in)
REV = "final"      # figure files: figures/fig2_tradeoff_final.*, figures/fig3_scale_final.* (existing files are never overwritten)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS_TICK,
                     "ytick.labelsize": FS_TICK, "legend.fontsize": FS, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5,
                     "ytick.major.pad": 1.5, "legend.frameon": False, "pdf.fonttype": 42})


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def safe_save(fig, stem):
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        assert not p.exists(), f"{p} exists — refusing to overwrite"
    fig.savefig(OUT / f"{stem}.pdf"); fig.savefig(OUT / f"{stem}.png", dpi=600)


def style_ax(ax):
    ax.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig2():
    data = {}
    for c, _ in CORP:
        F = [J(D9 / "tradeoff" / f"{c}_f1.00_s{sd}.json")["curves"]["F"] for sd in SEEDS]
        Q = [J(D9 / "rq14" / "full" / f"{c}_f1.00_s{sd}.json")["temperature_curve_QC"] for sd in SEEDS]
        for name, cur in (("F", F), ("QC", Q)):
            for T in TEMPS:
                v = {m: np.array([cv[str(T)][m] for cv in cur]) for m in ("precision", "recall", "diversity")}
                data[(c, name, T)] = {m: (x.mean(), x.std(ddof=1) / np.sqrt(len(x))) for m, x in v.items()}
    fig, axes = plt.subplots(2, 4, figsize=(W_IN, H2), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)
    for j, (c, lab) in enumerate(CORP):
        for row, (xk, yk) in enumerate([("precision", "recall"), ("diversity", "precision")]):
            ax = axes[row, j]
            for name, (leg, col, mk) in (("F", F_STYLE), ("QC", QC_STYLE)):
                xs = [data[(c, name, T)][xk][0] for T in TEMPS]; ys = [data[(c, name, T)][yk][0] for T in TEMPS]
                ax.plot(xs, ys, "-", color=col, lw=LW, zorder=2)
                ax.errorbar(xs, ys, xerr=[data[(c, name, T)][xk][1] for T in TEMPS], yerr=[data[(c, name, T)][yk][1] for T in TEMPS],
                            fmt="none", ecolor=col, elinewidth=ELW, alpha=0.7, zorder=2)
                ax.scatter(xs, ys, s=MS ** 2, marker=mk, color=col, edgecolors="white", linewidths=0.3, zorder=3)
                t1 = data[(c, name, 1.0)]
                ax.scatter([t1[xk][0]], [t1[yk][0]], s=MS_T1 ** 2, marker=mk, color=col, edgecolors=INK, linewidths=0.7, zorder=4)
            if c == "aloha_tape" and xk == "diversity":
                ax.set_xscale("log"); lo, hi = ax.get_xlim(); ax.set_xlim(lo, hi * 1.6)
            else:
                ax.xaxis.set_major_locator(MaxNLocator(2 if xk == "precision" else 3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            style_ax(ax)
            ax.set_xlabel("Precision" if xk == "precision" else "Diversity")   # r3: no unit (aloha grippers are not rad, report 17)
            if j == 0:
                ax.set_ylabel({"recall": "Recall", "precision": "Precision"}[yk])
            if row == 0:
                ax.set_title(lab, fontsize=FS, color=INK, loc="left", pad=2)
    handles = [Line2D([], [], color=F_STYLE[1], marker=F_STYLE[2], lw=LW, markersize=MS, label=F_STYLE[0]),
               Line2D([], [], color=QC_STYLE[1], marker=QC_STYLE[2], lw=LW, markersize=MS, label="QC"),
               Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=INK, markersize=MS_T1, markeredgewidth=0.7, label="T = 1")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, handletextpad=0.4, columnspacing=1.2)
    safe_save(fig, f"fig2_tradeoff_{REV}"); plt.close(fig)
    with open(OUT / f"fig2_tradeoff_{REV}.csv", "x", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh); cw.writerow(["corpus", "curve", "T", "precision_mean", "precision_se", "recall_mean", "recall_se", "diversity_mean", "diversity_se"])
        for (c, name, T), d in data.items():
            cw.writerow([c, name, T] + [x for m in ("precision", "recall", "diversity") for x in d[m]])
    return data


def fig3():
    data = {}
    for c, _ in CORP:
        for sc in SCALES:
            if sc == 1.0:
                F = [J(D9 / "results" / f"{c}_f1.00_s{sd}.json")["arms"]["A"]["main"] for sd in SEEDS]
                Q = [J(D9 / "rq14" / "full" / f"{c}_f1.00_s{sd}.json")["arms"]["QC"] for sd in SEEDS]
            else:
                F = [J(D9 / "rq11" / "common" / f"{c}_f{TAG[sc]}_s{sd}.json")["arms"]["A"] for sd in SEEDS]
                Q = [J(D9 / "rq14" / "scales" / f"{c}_f{TAG[sc]}_s{sd}.json")["QC"] for sd in SEEDS]
            for name, rs in (("F", F), ("QC", Q)):
                for m in ("recall", "diversity"):
                    x = np.array([r[m] for r in rs])
                    data[(c, name, sc, m)] = (x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)), x)
    fig, axes = plt.subplots(2, 4, figsize=(W_IN, H3), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)
    for j, (c, lab) in enumerate(CORP):
        for row, m in enumerate(("recall", "diversity")):
            ax = axes[row, j]
            for name, (leg, col, mk) in (("F", F_STYLE), ("QC", QC_STYLE)):
                mean = np.array([data[(c, name, sc, m)][0] for sc in SCALES]); hw = np.array([data[(c, name, sc, m)][1] for sc in SCALES])
                ax.fill_between(SCALES, mean - hw, mean + hw, color=col, alpha=0.18, lw=0, zorder=1)
                ax.plot(SCALES, mean, "-", color=col, lw=LW, zorder=2)
                ax.scatter(SCALES, mean, s=MS ** 2, marker=mk, color=col, edgecolors="white", linewidths=0.3, zorder=3)
            ax.set_xscale("log", base=2); ax.set_xticks(SCALES); ax.set_xticklabels(["1", "½", "¼", "⅛"]); ax.set_xlim(1.3, 0.1)
            ax.minorticks_off()
            if m == "diversity" and c == "aloha_tape":
                ax.set_yscale("log")
            else:
                ax.yaxis.set_major_locator(MaxNLocator(3))
            style_ax(ax)
            if row == 1:
                ax.set_xlabel("Data fraction")
            if j == 0:
                ax.set_ylabel({"recall": "Recall", "diversity": "Diversity"}[m])
            if m == "diversity" and c == "aloha_tape":
                ax.set_ylabel("Diversity")   # r3: unit removed (report 17)
            if row == 0:
                ax.set_title(lab, fontsize=FS, color=INK, loc="left", pad=2)
    handles = [Line2D([], [], color=F_STYLE[1], marker=F_STYLE[2], lw=LW, markersize=MS, label=F_STYLE[0]),
               Line2D([], [], color=QC_STYLE[1], marker=QC_STYLE[2], lw=LW, markersize=MS, label="QC")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, handletextpad=0.4, columnspacing=1.2)
    safe_save(fig, f"fig3_scale_{REV}"); plt.close(fig)
    with open(OUT / f"fig3_scale_{REV}.csv", "x", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh); cw.writerow(["corpus", "curve", "fraction", "metric", "mean", "ci95_halfwidth"] + [f"seed{s}" for s in SEEDS])
        for (c, name, sc, m), (mu, hw, x) in data.items():
            cw.writerow([c, name, sc, m, mu, hw] + list(x))
    return data


if __name__ == "__main__":
    d2 = fig2(); d3 = fig3()
    print("T = 1 points (mean recall / precision / diversity):")
    for c, lab in CORP:
        for name in ("F", "QC"):
            t = d2[(c, name, 1.0)]
            print(f"  {lab:12s} {name:2s} R {t['recall'][0]:.3f} P {t['precision'][0]:.3f} D {t['diversity'][0]:.4g}")
    print("Fig 3 recall means (1, 1/2, 1/4, 1/8):")
    for c, lab in CORP:
        for name in ("F", "QC"):
            print(f"  {lab:12s} {name:2s} " + " ".join(f"{d3[(c, name, sc, 'recall')][0]:.3f}" for sc in SCALES))
