"""
Visualization and plotting routines for pulse waveforms, state trajectories,
noise spectra, and Randomized Benchmarking curves.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def plot_pulse_and_detuning(
    time_grid: np.ndarray,
    j_pulse: np.ndarray,
    detuning_pulse: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plots exchange coupling J(t) and detuning voltage epsilon(t)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(time_grid, j_pulse, color="#00f0ff", lw=2.5, label="Exchange $J(t)$")
    ax1.set_ylabel("$J(t)$ (MHz $\\cdot 2\\pi$)", color="#00f0ff")
    ax1.grid(True, alpha=0.3, ls="--")
    ax1.legend(loc="upper right")
    ax1.set_title("Optimal Control Exchange Pulse Profile", fontsize=13, fontweight="bold")

    ax2.plot(time_grid, detuning_pulse, color="#ff0055", lw=2.0, label="Detuning $\\epsilon(t)$")
    ax2.set_xlabel("Time $t$ (ns)")
    ax2.set_ylabel("Detuning $\\epsilon$ (mV)", color="#ff0055")
    ax2.grid(True, alpha=0.3, ls="--")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_singlet_triplet_dynamics(
    time_grid: np.ndarray,
    populations: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    """Plots singlet/triplet state populations during exchange evolution."""
    plt.figure(figsize=(8, 5))
    colors = {"P_S": "#00f0ff", "P_T0": "#a855f7", "P_T+": "#3b82f6", "P_T-": "#10b981"}

    for key, vals in populations.items():
        plt.plot(time_grid, vals, label=f"${key}$", color=colors.get(key, "#ffffff"), lw=2.0)

    plt.xlabel("Time $t$ (ns)", fontsize=11)
    plt.ylabel("Population $|\\langle \\psi | \\phi \\rangle|^2$", fontsize=11)
    plt.title("Two-Electron Singlet-Triplet State Evolution", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3, ls="--")
    plt.legend(loc="center right")
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_rb_decay_curves(
    rb_results: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """Plots reference vs interleaved Randomized Benchmarking decay curves."""
    lengths = rb_results["lengths"]
    ref_fids = rb_results.get("ref_fidelities", rb_results.get("fidelities"))
    p_ref = rb_results.get("decay_p_ref", rb_results.get("decay_p"))

    plt.figure(figsize=(8, 5))
    m_fine = np.linspace(min(lengths), max(lengths), 200)

    # Reference curve
    plt.scatter(lengths, ref_fids, color="#00f0ff", label="Reference RB Data", zorder=3)
    plt.plot(m_fine, 0.5 * (p_ref**m_fine) + 0.25, color="#00f0ff", ls="--", label=f"Ref Fit ($p={p_ref:.4f}$)")

    if "interleaved_fidelities" in rb_results:
        intl_fids = rb_results["interleaved_fidelities"]
        p_intl = rb_results["decay_p_interleaved"]
        gate_fid = rb_results["gate_fidelity"]
        plt.scatter(lengths, intl_fids, color="#ff007f", label="Interleaved $\\sqrt{\\mathrm{SWAP}}$", zorder=3)
        plt.plot(m_fine, 0.5 * (p_intl**m_fine) + 0.25, color="#ff007f", ls="-", label=f"Interleaved Fit ($F={gate_fid*100:.2f}\\%$)")

    plt.xlabel("Clifford Sequence Length $m$", fontsize=11)
    plt.ylabel("Ground State Survival $P(|00\\rangle)$", fontsize=11)
    plt.title("Cirq Randomized Benchmarking under 1/f Charge Noise", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3, ls="--")
    plt.legend(loc="upper right")
    plt.ylim(0.2, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
