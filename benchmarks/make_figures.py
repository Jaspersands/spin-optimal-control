"""
Figures and numbers for the comparison with published devices.

    python benchmarks/make_figures.py            # writes figures/*.png and figures/comparison.json
    python benchmarks/make_figures.py --help

For every device in ``data/devices.json``: the CZ error budget against gate
time (stacked, for the pulse shape closest to the published one), the total
for a cosine window (and GRAPE where the coherent error matters), the best gate
time per shape, and the measured fidelity; then the recalibration interval for
Xue 2022 under assumed drift; then how much of each device's gap between the
measured and the modelled error the candidate mechanisms close
(``devices.gap_analysis``), with the unpublished inputs swept. Takes about five
minutes.

The coherent term is sampled on a fine grid (it has sharp minima at the
synchronised times of square-edged pulses); the noise terms are smooth and are
computed on a coarse grid and interpolated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spin_optimal_control.devices import best_gate_times, comparison_table, gap_analysis, load_devices  # noqa: E402
from spin_optimal_control.error_budget import (  # noqa: E402
    LABELS,
    coherent_error,
    cz_error_budget,
    cz_error_budget_for_pulse,
    grape_cz_pulse,
    shortest_gate_time,
)

FIG = ROOT / "figures"
STACK = ("zeeman", "exchange", "markovian", "coherent")            # bottom to top
COLORS = {"zeeman": "#3a75c4", "exchange": "#dd7240", "markovian": "#7a68c8", "coherent": "#2a9d74"}
MARKERS = {"zeeman": "o", "exchange": "s", "markovian": "D", "coherent": "^"}
INK, INK2, GRID = "#1f1f1f", "#5a5a5a", "#e6e6e6"
SHORT = {"zeeman": "Zeeman", "exchange": "exchange", "markovian": "Markovian", "coherent": "coherent"}


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9a9a9a",
        "axes.linewidth": 0.8, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 9.5,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "legend.frameon": False, "font.family": "DejaVu Sans",
    })


def grape_needed(budget_cos) -> bool:
    """GRAPE only pays where the cosine window's coherent error is a visible part of the budget."""
    return budget_cos.coherent > 0.05 * budget_cos.total


NOISE = ("zeeman", "exchange", "markovian")


def _loglog_interp(x_new, x, y):
    y = np.maximum(np.asarray(y, dtype=float), 1e-16)
    return np.exp(np.interp(np.log(x_new), np.log(x), np.log(y)))


def device_results(dev, n_coarse: int = 36, n_fine: int = 700, n_grape: int = 7):
    nm = dev.noise_parameters()
    lo, hi = 8.0, 1000.0
    t_min = shortest_gate_time(dev.model_shape, dev.j_cap_mhz, dev.tukey_alpha) if dev.j_cap_mhz else None
    extra = [dev.cz_time_ns] + ([t_min] if t_min else [])
    Tc = np.unique(np.append(np.geomspace(lo, hi, n_coarse), extra))
    Tf = np.unique(np.append(np.geomspace(lo, hi, n_fine), extra))
    sweep = {"duration_ns": Tf.tolist(), "stack": {}}
    for shape, key in ((dev.model_shape, "model"), ("cosine", "cosine")):
        if key == "cosine" and dev.model_shape == "cosine":
            break
        coarse = [cz_error_budget(nm, shape, t, tukey_alpha=dev.tukey_alpha, n_omega=120) for t in Tc]
        noise = {k: _loglog_interp(Tf, Tc, [b.contributions[k] for b in coarse]) for k in NOISE}
        coh = np.array([coherent_error(nm, shape, t, dev.tukey_alpha) for t in Tf])
        total = coh + sum(noise.values())
        if key == "model":
            sweep["stack"] = {**{k: v.tolist() for k, v in noise.items()}, "coherent": coh.tolist()}
            sweep["total_model_shape"] = total.tolist()
        sweep["total_cosine" if key == "cosine" else "total_model"] = total.tolist()
    best = best_gate_times(dev, ("square", "cosine"))
    pub = dev.budget()
    pub_cos = cz_error_budget(nm, "cosine", dev.cz_time_ns)
    out = {
        "key": dev.key, "label": dev.label, "short": dev.short, "gate": dev.gate, "model_shape": dev.model_shape,
        "gate_time_ns": dev.cz_time_ns, "measured_fidelity": dev.measured_fidelity,
        "measured_uncertainty": dev.measured_uncertainty, "method": dev.method,
        "j_cap_mhz": dev.j_cap_mhz, "t_min_ns": t_min, "zeeman_alpha": list(nm.zeeman_alpha),
        "delta_ez_model_mhz": dev.delta_ez_model_mhz, "coherent_at_idle_dez": dev.coherent_at_idle(),
        "published": pub.as_dict(),
        "best": {s_: b.as_dict() for s_, b in best.items()},
        "sweep": sweep,
    }
    if n_grape and grape_needed(pub_cos):
        t_lo = shortest_gate_time("square", dev.j_cap_mhz) if dev.j_cap_mhz else 0.25 * dev.cz_time_ns
        TG = np.geomspace(t_lo * 1.0005, 2.0 * dev.cz_time_ns, int(n_grape))
        g = []
        for t in TG:
            j, dt = grape_cz_pulse(nm.delta_ez_mhz, float(t), j_max_mhz=dev.j_cap_mhz, max_iter=400)
            g.append(cz_error_budget_for_pulse(nm, j, dt, label="grape"))
        k = int(np.argmin([b.total for b in g]))
        out["sweep"]["grape"] = {"duration_ns": [b.duration_ns for b in g], "total": [b.total for b in g],
                                 "coherent": [b.coherent for b in g]}
        out["best"]["grape"] = g[k].as_dict()
        out["best"]["grape"]["note"] = (f"best of {len(g)} GRAPE runs between the shortest allowed time "
                                        f"({TG[0]:.0f} ns) and {TG[-1]:.0f} ns")
    return out


def panel(ax, r):
    T_all = np.asarray(r["sweep"]["duration_ns"])
    t_min = r.get("t_min_ns") or T_all[0]
    keep = T_all >= t_min * (1 - 1e-9)                 # durations the demonstrated exchange can reach
    T = T_all[keep]
    base = np.zeros_like(T)
    meas = 1.0 - r["measured_fidelity"]
    for k in STACK:
        y = 100 * np.asarray(r["sweep"]["stack"][k])[keep]
        ax.fill_between(T, base, base + y, color=COLORS[k], lw=0, alpha=0.85, label=LABELS[k])
        ax.plot(T, base + y, color="white", lw=0.6)
        base = base + y
    shape_name = r["model_shape"]
    if t_min > T_all[0]:
        ax.axvspan(T_all[0], t_min, color="#f4f4f4", lw=0, zorder=0, label="needs more exchange than the model allows")
    if shape_name != "cosine":
        ax.plot(T, 100 * np.asarray(r["sweep"]["total_cosine"])[keep], color=INK2, lw=0.9, ls=(0, (1, 1.5)),
                label="total, cosine window")
    if "grape" in r["sweep"]:
        g = r["sweep"]["grape"]
        ax.plot(g["duration_ns"], 100 * np.asarray(g["total"]), "o", ms=4.5, mfc="white", mec=INK, mew=1.0,
                label="total, GRAPE pulse", zorder=4)
    ax.errorbar([r["gate_time_ns"]], [100 * meas], yerr=[100 * r["measured_uncertainty"]], fmt="o", ms=6,
                color=INK, ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=5, label="measured")
    ax.annotate(f"measured {100 * meas:.2f}%\n({r['method']})", (r["gate_time_ns"], 100 * meas),
                xytext=(-9, 0), textcoords="offset points", fontsize=8, color=INK, ha="right", va="center")
    pub = r["published"]["total"]
    ax.annotate(f"model {100 * pub:.3f}%", (r["gate_time_ns"], 100 * pub), xytext=(-9, 9),
                textcoords="offset points", fontsize=8, color=INK2, ha="right",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=1))
    ax.axvline(r["gate_time_ns"], color="#b5b5b5", lw=0.7, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(T_all[0], T_all[-1])
    top = 100 * (meas + r["measured_uncertainty"]) * 1.45
    ax.set_ylim(0, top)
    shape = {"cosine": "cosine", "tukey": "Tukey", "square": "square"}[shape_name]
    ax.set_title(f"{r['short']}: {shape}-window CZ", loc="left", color=INK)
    ax.set_xlabel("CZ gate time (ns)")
    ax.set_ylabel("infidelity (%)")


def budget_figure(results, path):
    n = len(results)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, results):
        panel(ax, r)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = [], []
    for ax in axes[:n]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summary_figure(results, path):
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8, 0.62 * len(results) + 1.7))
    ys = np.arange(len(results))[::-1]
    for y, r in zip(ys, results):
        c = r["published"]
        for k in STACK:
            if c[k] > 1e-6:
                ax.plot(c[k], y + 0.2, MARKERS[k], ms=5.5, color=COLORS[k], mec="white", mew=0.6)
        meas, err = 1 - r["measured_fidelity"], r["measured_uncertainty"]
        ax.plot([c["total"], meas], [y, y], color="#c8c8c8", lw=1.0, zorder=0)
        ax.plot(c["total"], y, "o", ms=8, mfc="white", mec=INK, mew=1.4)
        ax.errorbar(meas, y, xerr=[[min(err, 0.9 * meas)], [err]], fmt="o", ms=8, color=INK, elinewidth=1.0, capsize=2.5)
        note = " (model inside the error bar)" if meas - err <= c["total"] else ""
        ratio = meas / c["total"]
        ratio = f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.1f}"
        ax.text(np.sqrt(c["total"] * meas), y - 0.32, f"{ratio} times{note}", ha="center", fontsize=8, color=INK2)
        idle = r.get("coherent_at_idle_dez")
        if idle and idle > 10 * max(c["coherent"], 1e-12):
            ax.plot(idle, y + 0.2, MARKERS["coherent"], ms=5.5, mfc="white", mec=COLORS["coherent"], mew=1.0)
            ax.text(idle, y + 0.42, "coherent at the idle ΔEz", fontsize=7.5, color=INK2, ha="center", va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['short']}\n{r['gate']}, {r['gate_time_ns']:.0f} ns" for r in results], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 2.5e-2)
    ax.set_ylim(-0.7, len(results) - 0.3)
    ax.grid(False)
    ax.grid(True, axis="x", which="major", color=GRID, lw=0.8)
    ax.set_xlabel("average gate infidelity at the published gate time")
    handles = [Line2D([], [], ls="", marker=MARKERS[k], ms=5.5, color=COLORS[k], mec="white", label=LABELS[k]) for k in STACK]
    handles += [Line2D([], [], ls="", marker="o", ms=7, mfc="white", mec=INK, mew=1.4, label="model total"),
                Line2D([], [], ls="", marker="o", ms=7, color=INK, label="measured")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def calibration_results(dev, n_runs: int = 120, seed: int = 1):
    """
    Recalibration interval for one device: once with the drift and measurement
    errors from its data entries (all assumed), and once with a J drift sized
    so that plain recalibration at the paper's interval adds the gap between
    the measured and the modelled infidelity.
    """
    from spin_optimal_control.calibration import drift_rate_for_infidelity, longest_interval, recalibration_interval

    base = dev.budget().total
    every = dev.value("recalibration_interval_h")
    common = dict(drift_dez_mhz_per_sqrt_h=dev.value("drift_dez_mhz_per_sqrt_h"),
                  meas_std_j_rel=dev.value("calibration_std_j_rel"), meas_std_dez_mhz=dev.value("calibration_std_dez_mhz"),
                  intervals_h=np.geomspace(0.05, 48.0, 22), n_runs=n_runs, seed=seed)
    d_gap = drift_rate_for_infidelity((1 - dev.measured_fidelity) - base, every, dev.value("calibration_std_j_rel"))
    out = {"key": dev.key, "baseline_infidelity": base, "reported_interval_h": every, "scenarios": {}}
    for name, dj in (("assumed", dev.value("drift_j_rel_per_sqrt_h")), ("gap", d_gap)):
        r = recalibration_interval(base, dev.cz_time_ns, dev.value("j_on_mhz"), drift_j_rel_per_sqrt_h=dj, **common)
        sc = {"drift_j_rel_per_sqrt_h": dj, "intervals_h": r["intervals_h"].tolist(),
              "infidelity_plain": r["infidelity_plain"].tolist(), "infidelity_kalman": r["infidelity_kalman"].tolist(),
              "floor_plain": r["floor_plain"], "targets": {}}
        for tgt in (0.999, 0.9995):
            sc["targets"][str(tgt)] = {"plain_h": longest_interval(r["intervals_h"], r["infidelity_plain"], tgt),
                                       "kalman_h": longest_interval(r["intervals_h"], r["infidelity_kalman"], tgt)}
        out["scenarios"][name] = sc
    return out


def calibration_figure(c, short, path):
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    a, g = c["scenarios"]["assumed"], c["scenarios"]["gap"]
    x = np.asarray(a["intervals_h"])
    ja, jg = 100 * a["drift_j_rel_per_sqrt_h"], 100 * g["drift_j_rel_per_sqrt_h"]
    ax.plot(x, 100 * np.asarray(a["infidelity_plain"]), color=INK2, lw=1.6)
    ax.plot(x, 100 * np.asarray(a["infidelity_kalman"]), color=COLORS["zeeman"], lw=1.6)
    ax.plot(x, 100 * np.asarray(g["infidelity_plain"]), color=INK2, lw=1.2, ls=(0, (4, 2)))
    ax.plot(x, 100 * np.asarray(g["infidelity_kalman"]), color=COLORS["zeeman"], lw=1.2, ls=(0, (4, 2)))
    handles = [Line2D([], [], color=INK2, lw=1.6, label="set to the newest measurement"),
               Line2D([], [], color=COLORS["zeeman"], lw=1.6, label="Kalman tracker"),
               Line2D([], [], color=INK, lw=1.2, label=f"J drift {ja:.0f} %/√h (assumed)"),
               Line2D([], [], color=INK, lw=1.2, ls=(0, (4, 2)), label=f"J drift {jg:.0f} %/√h (sized to the measured gap)")]
    ax.axhline(100 * c["baseline_infidelity"], color="#b5b5b5", lw=0.8, ls=(0, (4, 3)))
    ax.text(x[0] * 1.1, 100 * c["baseline_infidelity"] - 0.002, "modelled CZ without drift", fontsize=8, color=INK2, va="top")
    for tgt in (0.999, 0.9995):
        ax.axhline(100 * (1 - tgt), color=INK, lw=0.8)
        ax.text(x[-1] / 1.1, 100 * (1 - tgt), f"target {100 * tgt:g}%", fontsize=8, color=INK, va="bottom", ha="right")
    if c.get("reported_interval_h"):
        ax.axvline(c["reported_interval_h"], color="#b5b5b5", lw=0.8)
        ax.text(c["reported_interval_h"] * 1.08, 0.094, "Ramsey recalibration\ninterval in the paper", fontsize=8, color=INK2, va="top")
    ax.set_xscale("log")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 0.11)
    ax.set_xlabel("time between recalibrations (h)")
    ax.set_ylabel("mean CZ infidelity (%)")
    ax.set_title(f"{short}: recalibration of the modelled CZ under drift", loc="left", color=INK)
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, frameon=True, framealpha=1, facecolor="white", edgecolor="none")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


GAP_COLORS = {"gate_zeeman": "#2a9d74", "slow_zeeman": COLORS["zeeman"], "slow_exchange": COLORS["exchange"],
              "calibration_residual": "#7a68c8"}
DEVICE_STYLE = {"xue2022": ("-", "o"), "noiri2022": ((0, (5, 2)), "s"), "mills2022": ((0, (1, 1.5)), "D"),
                "tanttu2024": ((0, (6, 2, 1, 2)), "^")}


def _first_word(short: str) -> str:
    return short.split(" (")[0]


def gap_bars(ax, gaps, legend_ax):
    from matplotlib.patches import Patch

    ys = np.arange(len(gaps))[::-1]
    for y, g in zip(ys, gaps):
        left = 0.0
        for k in ("calibration_residual", "gate_zeeman", "slow_exchange", "slow_zeeman"):
            t = g["terms"][k]
            if t["status"] not in ("modelled", "upper bound", "measured") or t["share"] <= 0:
                continue
            w = 100 * t["share"]
            ax.barh(y, w, left=left, height=0.55, color=GAP_COLORS[k], lw=0,
                    hatch="////" if t["status"] == "upper bound" else None, edgecolor="white")
            left += w
        ax.barh(y, 100 - left, left=left, height=0.55, color="#ececec", lw=0)
        ax.text(101.5, y, f"{100 * g['closed_share']:.0f}% of {g['gap']:.1e}", va="center", fontsize=8, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([_first_word(g["short"]) for g in gaps], fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(gaps) - 0.4)
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("share of the gap (measured minus model), %")
    ax.set_title("Gap closed by terms with published inputs", loc="left", color=INK)
    short = {"slow_zeeman": "Zeeman noise below the Ramsey band", "slow_exchange": "exchange noise below the Ramsey band",
             "gate_zeeman": "qubit frequencies moved by the gate voltage",
             "calibration_residual": "coherent residual measured by GST"}
    handles = [Patch(color=GAP_COLORS[k], label=v) for k, v in short.items()]
    handles += [Patch(facecolor="#bdbdbd", hatch="////", edgecolor="white", label="upper bound (fed back during the run)"),
                Patch(color="#ececec", label="not closed by anything published")]
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center left", fontsize=7.5)


def _spread(points, min_dex=0.22):
    """Nudge label heights (log scale) apart so that labels at the same x do not overlap."""
    order = sorted(range(len(points)), key=lambda i: points[i])
    out = list(points)
    for a, b in zip(order[:-1], order[1:]):
        if np.log10(out[b]) - np.log10(out[a]) < min_dex:
            out[b] = out[a] * 10**min_dex
    return out


def gap_sweep(ax, gaps, term, title, xlabel, sum_with=None, marks=None, flip=False, star_color=None):
    """One term as a function of an unpublished input, divided by each device's gap; 1 closes it."""
    ends, names = [], []
    for g in gaps:
        t = g["terms"][term]
        if not t.get("sweep"):
            continue
        x = np.asarray(t["sweep"]["x"])
        y = np.asarray(t["sweep"]["y"], dtype=float)
        if sum_with:
            y = y + np.asarray(g["terms"][sum_with]["sweep"]["y"], dtype=float)
        r = y / g["gap"]
        ls, mk = DEVICE_STYLE.get(g["key"], ("-", "o"))
        ax.plot(x, np.abs(r), color=INK2, lw=1.1, ls=ls)
        pos = r > 0
        ax.plot(x[pos], r[pos], mk, ms=3.8, color=INK, mec=INK, ls="")
        ax.plot(x[~pos], -r[~pos], mk, ms=3.8, mfc="white", mec=INK2, mew=0.9, ls="")
        i = int(np.argmin(x)) if flip else int(np.argmax(x))
        ends.append((x[i], min(max(abs(r[i]), 2e-5), 20.0)))
        names.append(_first_word(g["short"]))
    heights = _spread([e[1] for e in ends])
    for (xe, _), h, name in zip(ends, heights, names):
        ax.annotate(name, (xe, h), xytext=(5, 0), textcoords="offset points", fontsize=7.5, color=INK2,
                    va="center", ha="left", annotation_clip=False)
    for xm, ym in (marks or []):
        ax.plot([xm], [ym], "*", ms=10, color=star_color or COLORS["zeeman"], mec="white", mew=0.5, zorder=5)
    ax.axhline(1.0, color=INK, lw=0.8)
    ax.text(0.03, 1.0, "closes the gap", transform=ax.get_yaxis_transform(), fontsize=7.5, color=INK, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 30)
    ax.grid(True, axis="both", which="major", color=GRID, lw=0.8)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("added error / gap")
    if flip:
        ax.invert_xaxis()


def gap_figure(gaps, path):
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(11.5, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.0], hspace=0.55, wspace=0.42)
    gap_bars(fig.add_subplot(gs[0, :2]), gaps, fig.add_subplot(gs[0, 2]))
    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    marks = [(g["terms"]["slow_exchange"]["hours"],
              (g["terms"]["slow_zeeman"]["value"] + g["terms"]["slow_exchange"]["value"]) / g["gap"])
             for g in gaps if g["terms"]["slow_zeeman"]["value"] is not None and g["terms"]["slow_exchange"]["value"] is not None]
    gap_sweep(axes[0], gaps, "slow_zeeman", "Noise slower than the Ramsey average",
              "lab time without recalibration (h)", sum_with="slow_exchange", marks=marks)
    gap_sweep(axes[1], gaps, "distortion", "Pulse distortion (single-pole low-pass)",
              "bandwidth of the exchange line (MHz)", flip=True)
    mk = [(g["terms"]["gate_zeeman"]["df_dlnj_peak_mhz"][1], g["terms"]["gate_zeeman"]["share"])
          for g in gaps if g["terms"]["gate_zeeman"]["status"] == "modelled"]
    gap_sweep(axes[2], gaps, "gate_zeeman", "Qubit frequency moved by the exchange gate",
              "df/d(ln J) of one qubit (MHz)", marks=mk, star_color=GAP_COLORS["gate_zeeman"])
    handles = [Line2D([], [], ls="", marker="o", ms=4, color=INK, label="adds error"),
               Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=INK2, label="removes error (absolute value)"),
               Line2D([], [], ls="", marker="*", ms=9, color=INK2, mec="white",
                      label="at the published or derived input (Xue, Tanttu)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.085, right=0.95, top=0.95, bottom=0.14)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _rounded(x, digits: int = 6):
    """Six significant digits in comparison.json; the figures use the full values."""
    if isinstance(x, dict):
        return {k: _rounded(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_rounded(v, digits) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.{digits}g}") if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description="Device-comparison figures and figures/comparison.json (a few minutes).")
    ap.add_argument("--out", type=Path, default=ROOT / "figures", help="output directory (default: figures/)")
    ap.add_argument("--quick", action="store_true", help="coarse grids, no GRAPE, fewer calibration runs; for a fast check")
    args = ap.parse_args(argv)
    style()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(n_coarse=16, n_fine=200, n_grape=0) if args.quick else {}
    t0 = time.time()
    devs = load_devices()
    results = []
    for dev in devs.values():
        t1 = time.time()
        results.append(device_results(dev, **kw))
        print(f"{dev.key}: {time.time() - t1:.1f} s", flush=True)
    table = comparison_table(devs)
    budget_figure(results, out_dir / "error_budget_vs_gate_time.png")
    summary_figure(results, out_dir / "summary.png")
    cal = calibration_results(devs["xue2022"], n_runs=30 if args.quick else 120)
    calibration_figure(cal, devs["xue2022"].short, out_dir / "calibration_interval.png")
    gaps = []
    for dev in devs.values():
        t1 = time.time()
        gaps.append(gap_analysis(dev, n_points=7 if args.quick else 13))
        print(f"{dev.key} gap mechanisms: {time.time() - t1:.1f} s", flush=True)
    gap_figure(gaps, out_dir / "gap_mechanisms.png")
    payload = {"table": table, "devices": results, "calibration": cal, "gap": gaps}
    (out_dir / "comparison.json").write_text(json.dumps(_rounded(payload), indent=1) + "\n")
    for r in results:
        b = r["best"]
        print(f"{r['label']:42s} published {100 * r['published']['fidelity']:.3f}% ({r['published']['dominant']}) | "
              + " | ".join(f"best {s_} {v['duration_ns']:.1f} ns {100 * v['fidelity']:.4f}% ({v['dominant']})" for s_, v in b.items()))
    for name, sc in cal["scenarios"].items():
        for tgt, v in sc["targets"].items():
            print(f"recalibration ({name}, J drift {100 * sc['drift_j_rel_per_sqrt_h']:.1f} %/sqrt h), target {tgt}: "
                  f"every {v['plain_h']:.2f} h (newest measurement), {v['kalman_h']:.2f} h (Kalman)")
    from spin_optimal_control.devices import format_gap_table

    print(format_gap_table(gaps))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
