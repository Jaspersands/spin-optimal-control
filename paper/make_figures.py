"""
Vector figures for the note, drawn from ../figures/comparison.json.

    python paper/make_figures.py

Nothing is recomputed: every curve and marker is read from the JSON that
benchmarks/make_figures.py writes, so the figures show the package results as
they are. The layouts are redrawn at the size they appear in the note.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "figures" / "comparison.json"
OUT = HERE / "figures"

STACK = ("zeeman", "exchange", "markovian", "coherent")
COLORS = {"zeeman": "#3a75c4", "exchange": "#dd7240", "markovian": "#7a68c8", "coherent": "#2a9d74"}
MARKERS = {"zeeman": "o", "exchange": "s", "markovian": "D", "coherent": "^"}
LABELS = {"zeeman": "Zeeman noise", "exchange": "exchange noise", "markovian": "Markovian $T_1$, $T_2$",
          "coherent": "coherent"}
INK, INK2, GRID = "#1f1f1f", "#5a5a5a", "#e4e4e4"
NAMES = {"xue2022": "Xue 2022", "noiri2022": "Noiri 2022", "mills2022": "Mills 2022", "tanttu2024": "Tanttu 2024 A"}
METHOD = {"xue2022": "GST", "noiri2022": "IRB", "mills2022": "IRB", "tanttu2024": "IRB"}
GATE = {"xue2022": "CZ", "noiri2022": "CNOT", "mills2022": "CZ", "tanttu2024": "CZ"}
TEXT_W = 7.0                       # inches, two-column text width of revtex reprint


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a8a",
        "axes.linewidth": 0.6, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "legend.frameon": False, "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "lines.linewidth": 1.0,
    })


def sci(x, digits=2):
    """3.36e-03 -> 3.36\\times10^{-3} for mathtext."""
    m, e = f"{x:.{digits}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


# --------------------------------------------------------------------------- #
def summary(data, path):
    rows = data["devices"]
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.45))
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        c = r["published"]
        for k in STACK:
            if c[k] > 1e-6:
                ax.plot(c[k], y + 0.22, MARKERS[k], ms=4.2, color=COLORS[k], mec="white", mew=0.4)
        meas, err = 1 - r["measured_fidelity"], r["measured_uncertainty"]
        ax.plot([c["total"], meas], [y, y], color="#c4c4c4", lw=0.8, zorder=0)
        ax.plot(c["total"], y, "o", ms=6, mfc="white", mec=INK, mew=1.1)
        ax.errorbar(meas, y, xerr=[[min(err, 0.9 * meas)], [err]], fmt="o", ms=6, color=INK, elinewidth=0.8,
                    capsize=2)
        share = 100 * c["total"] / meas
        note = ", inside the error bar" if meas - err <= c["total"] else ""
        ax.text(np.sqrt(c["total"] * meas), y - 0.36, f"model {share:.0f}% of measured{note}", ha="center",
                fontsize=7, color=INK2)
        idle = r.get("coherent_at_idle_dez")
        if idle and idle > 10 * max(c["coherent"], 1e-12):
            ax.plot(idle, y + 0.22, MARKERS["coherent"], ms=4.2, mfc="white", mec=COLORS["coherent"], mew=0.8)
            ax.text(idle, y + 0.44, r"coherent at idle $\Delta E_z$", fontsize=6.5, color=INK2, ha="center",
                    va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{NAMES[r['key']]}\n{GATE[r['key']]}, {r['gate_time_ns']:.0f} ns, {METHOD[r['key']]}"
                        for r in rows], fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 2.5e-2)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.grid(True, axis="x", which="major")
    ax.set_xlabel("average gate infidelity at the published gate time")
    handles = [Line2D([], [], ls="", marker=MARKERS[k], ms=4.2, color=COLORS[k], mec="white", label=LABELS[k])
               for k in STACK]
    handles += [Line2D([], [], ls="", marker="o", ms=5.5, mfc="white", mec=INK, mew=1.1, label="model total"),
                Line2D([], [], ls="", marker="o", ms=5.5, color=INK, label="measured")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), ncol=1, handletextpad=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def budget_panel(ax, r):
    T_all = np.asarray(r["sweep"]["duration_ns"])
    t_min = r.get("t_min_ns") or T_all[0]
    keep = T_all >= t_min * (1 - 1e-9)
    T = T_all[keep]
    base = np.zeros_like(T)
    meas = 1.0 - r["measured_fidelity"]
    for k in STACK:
        y = 100 * np.asarray(r["sweep"]["stack"][k])[keep]
        ax.fill_between(T, base, base + y, color=COLORS[k], lw=0, alpha=0.85)
        base = base + y
    if t_min > T_all[0]:
        ax.axvspan(T_all[0], t_min, color="#f2f2f2", lw=0, zorder=0)
    if r["model_shape"] != "cosine":
        ax.plot(T, 100 * np.asarray(r["sweep"]["total_cosine"])[keep], color=INK2, lw=0.8, ls=(0, (1, 1.2)))
    ax.errorbar([r["gate_time_ns"]], [100 * meas], yerr=[100 * r["measured_uncertainty"]], fmt="o", ms=4,
                color=INK, elinewidth=0.8, capsize=2, zorder=5)
    pub = r["published"]["total"]
    ax.plot([r["gate_time_ns"]], [100 * pub], "o", ms=4, mfc="white", mec=INK, mew=0.9, zorder=6)
    ax.axvline(r["gate_time_ns"], color="#bdbdbd", lw=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(T_all[0], T_all[-1])
    ax.set_ylim(0, 100 * (meas + r["measured_uncertainty"]) * 1.35)
    ax.grid(True, axis="y")
    shape = {"cosine": "cosine", "tukey": "Tukey", "square": "square"}[r["model_shape"]]
    ax.set_title(f"{NAMES[r['key']]}, {shape} window", loc="left")
    ax.set_xlabel("gate time (ns)")


def budget_vs_time(data, path):
    rows = data["devices"]
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.55))
    for ax, r in zip(axes.ravel(), rows):
        budget_panel(ax, r)
        ax.title.set_fontsize(7.5)
        ax.tick_params(labelsize=6.8)
        ax.xaxis.label.set_fontsize(7.2)
    for ax in axes[0]:
        ax.set_xlabel("")
    for ax in axes[:, 0]:
        ax.set_ylabel("infidelity (%)", fontsize=7.2)
    handles = [Patch(color=COLORS[k], alpha=0.85, label=LABELS[k]) for k in STACK]
    handles += [Line2D([], [], color=INK2, lw=0.8, ls=(0, (1, 1.2)), label="total, cosine window"),
                Patch(color="#f2f2f2", label="beyond exchange ceiling"),
                Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=INK, label="model, published time"),
                Line2D([], [], ls="", marker="o", ms=4, color=INK, label="measured")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0,
               handlelength=1.3, handletextpad=0.4, fontsize=6.6, labelspacing=0.3)
    fig.tight_layout(pad=0.25, rect=(0, 0, 1, 0.8), h_pad=0.4, w_pad=0.5)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
GAP_COLORS = {"calibration_residual": "#7a68c8", "gate_zeeman": "#2a9d74", "slow_exchange": COLORS["exchange"],
              "slow_zeeman": COLORS["zeeman"]}
GAP_SHORT = {"slow_zeeman": "Zeeman noise below the Ramsey band", "slow_exchange": "exchange noise below the Ramsey band",
             "gate_zeeman": "qubit frequencies moved by the gate voltage",
             "calibration_residual": "coherent residual measured by GST"}
DEVICE_STYLE = {"xue2022": ("-", "o"), "noiri2022": ((0, (5, 2)), "s"), "mills2022": ((0, (1, 1.5)), "D"),
                "tanttu2024": ((0, (6, 2, 1, 2)), "^")}


def gap_bars(ax, gaps, legend_ax):
    ys = np.arange(len(gaps))[::-1]
    for y, g in zip(ys, gaps):
        left = 0.0
        for k in ("calibration_residual", "gate_zeeman", "slow_exchange", "slow_zeeman"):
            t = g["terms"][k]
            if t["status"] not in ("modelled", "upper bound", "measured") or not t["share"] or t["share"] <= 0:
                continue
            w = 100 * t["share"]
            ax.barh(y, w, left=left, height=0.56, color=GAP_COLORS[k], lw=0,
                    hatch="//////" if t["status"] == "upper bound" else None, edgecolor="white")
            left += w
        ax.barh(y, 100 - left, left=left, height=0.56, color="#ececec", lw=0)
        ax.text(101.5, y, (f"{100 * g['closed_share']:.1f}%" if g['closed_share'] > 0 else "0%") + rf" of ${sci(g['gap'])}$", va="center", fontsize=7.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([NAMES[g["key"]] for g in gaps])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, len(gaps) - 0.45)
    ax.grid(True, axis="x")
    ax.set_xlabel("share of the gap closed by terms with published inputs (%)")
    handles = [Patch(color=GAP_COLORS[k], label=GAP_SHORT[k]) for k in GAP_SHORT]
    handles += [Patch(facecolor="#bdbdbd", hatch="//////", edgecolor="white", label="upper bound (fed back during run)"),
                Patch(color="#ececec", label="not closed by published inputs")]
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center left", fontsize=7, handlelength=1.2)


def _spread(points, min_dex=0.3):
    order = sorted(range(len(points)), key=lambda i: points[i])
    out = list(points)
    for a, b in zip(order[:-1], order[1:]):
        if np.log10(out[b]) - np.log10(out[a]) < min_dex:
            out[b] = out[a] * 10**min_dex
    return out


def gap_sweep(ax, gaps, term, title, xlabel, sum_with=None, marks=None, flip=False, star_color=None):
    for g in gaps:
        t = g["terms"][term]
        if not t.get("sweep"):
            continue
        x = np.asarray(t["sweep"]["x"])
        y = np.asarray(t["sweep"]["y"], dtype=float)
        if sum_with:
            y = y + np.asarray(g["terms"][sum_with]["sweep"]["y"], dtype=float)
        r = y / g["gap"]
        ls, mk = DEVICE_STYLE[g["key"]]
        ax.plot(x, np.abs(r), color=INK2, lw=0.8, ls=ls)
        pos = r > 0
        ax.plot(x[pos], r[pos], mk, ms=3, color=INK, ls="")
        ax.plot(x[~pos], -r[~pos], mk, ms=3, mfc="white", mec=INK2, mew=0.7, ls="")
    for xm, ym in (marks or []):
        ax.plot([xm], [ym], "*", ms=8, color=star_color or COLORS["zeeman"], mec="white", mew=0.4, zorder=5)
    ax.axhline(1.0, color=INK, lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 30)
    ax.grid(True, axis="both", which="major")
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    if flip:
        ax.invert_xaxis()


def gap_figure(data, path):
    gaps = data["gap"]
    fig = plt.figure(figsize=(TEXT_W, 4.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.0], hspace=0.62, wspace=0.42,
                          left=0.085, right=0.93, top=0.97, bottom=0.19)
    gap_bars(fig.add_axes([0.13, 0.69, 0.42, 0.28]), gaps, fig.add_axes([0.70, 0.66, 0.29, 0.32]))
    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    marks = [(g["terms"]["slow_exchange"]["hours"],
              (g["terms"]["slow_zeeman"]["value"] + g["terms"]["slow_exchange"]["value"]) / g["gap"])
             for g in gaps if g["terms"]["slow_zeeman"]["value"] is not None
             and g["terms"]["slow_exchange"]["value"] is not None]
    gap_sweep(axes[0], gaps, "slow_zeeman", "(b) noise slower than the Ramsey average",
              "time without recalibration (h)", sum_with="slow_exchange", marks=marks)
    gap_sweep(axes[1], gaps, "distortion", "(c) pulse distortion, single-pole low-pass",
              "bandwidth of the exchange line (MHz)", flip=True)
    mk = [(g["terms"]["gate_zeeman"]["df_dlnj_peak_mhz"][1], g["terms"]["gate_zeeman"]["share"])
          for g in gaps if g["terms"]["gate_zeeman"]["status"] == "modelled"]
    gap_sweep(axes[2], gaps, "gate_zeeman", "(d) qubit frequency moved by the gate",
              r"$df/d\ln J$ of one qubit (MHz)", marks=mk, star_color=GAP_COLORS["gate_zeeman"])
    axes[0].set_ylabel("added error / gap")
    fig.text(0.012, 0.975, "(a)", fontsize=8, va="top")
    handles = [Line2D([], [], ls="", marker="o", ms=3.5, color=INK, label="adds error"),
               Line2D([], [], ls="", marker="o", ms=3.5, mfc="white", mec=INK2, label="removes error (absolute value)"),
               Line2D([], [], ls="", marker="*", ms=9, color=INK2, mec="white",
                      label="at the published or derived input (Xue, Tanttu)")]
    handles += [Line2D([], [], color=INK2, lw=0.8, ls=DEVICE_STYLE[k][0], marker=DEVICE_STYLE[k][1], ms=3,
                       mfc=INK, mec=INK, label=NAMES[k].replace(" A", "")) for k in DEVICE_STYLE]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=7,
               columnspacing=1.4, handletextpad=0.3, handlelength=2.6)
    fig.savefig(path)
    plt.close(fig)


def main():
    style()
    OUT.mkdir(exist_ok=True)
    data = json.loads(DATA.read_text())
    summary(data, OUT / "summary.pdf")
    budget_vs_time(data, OUT / "budget_vs_time.pdf")
    gap_figure(data, OUT / "gap_mechanisms.pdf")
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.pdf"))))


if __name__ == "__main__":
    main()
