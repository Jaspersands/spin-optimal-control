"""
Window-shaped exchange pulses for an adiabatic CZ.

With a Zeeman gradient ΔB_z ≫ J, the two-electron eigenstates are close to
|↑↓⟩ and |↓↑⟩ and an exchange pulse J(t) only shifts their energies: the gate
is a controlled phase, φ = arg U₀₀ − arg U₀₁ − arg U₁₀ + arg U₁₁, and a CZ
needs φ = π. The price of the exchange term is that it also couples |↑↓⟩ and
|↓↑⟩ with strength J/2 across a gap ≈ ΔB_z; if J(t) switches faster than the
gap allows, population is transferred between them (a SWAP-like error that no
single-qubit phase can undo).

Smooth windows suppress that transfer because the non-adiabatic amplitude is,
to first order, the Fourier component of J(t) at the gap frequency: a square
pulse has a sinc spectrum (∝ 1/f tails), a cosine window falls off as 1/f³.
This is the pulse shape used for high-fidelity CZ gates in silicon
(e.g. Xue et al., Nature 601, 343 (2022); Noiri et al., Nature 601, 338 (2022)).

Calibration is exact and shape-independent. |↑↑⟩ and |↓↓⟩ only acquire the
phase ∓2π·10⁻³·∫J/4 dt, while the (|↑↓⟩, |↓↑⟩) block is e^{+iχ} times an SU(2)
matrix [[a, b], [−b*, a*]], so U₀₁*·U₁₀* contributes |a|² (real) to φ and

    φ = −2π·10⁻³ ∫ J(t) dt       (J in MHz, t in ns)

for *any* pulse shape and any ΔB_z. A CZ therefore needs pulse area
∫J dt = 500 MHz·ns, and the only thing the shape changes is the leakage |b|².
``AdiabaticCZDesigner`` sets that area, evolves the full 4×4 Hamiltonian and
reports the leakage and the CZ fidelity up to single-qubit Z phases.

(v0.3.0 shipped a "DRAG quadrature" for exchange pulses; exchange is a scalar
baseband control with no quadrature, so that module was removed.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .hamiltonian import ExchangeDynamics, SiliconSpinHamiltonian

WINDOWS = ("square", "cosine", "tukey", "blackman")
CZ_AREA_MHZ_NS = 500.0            # ∫J dt for a π conditional phase: 2π·10⁻³·500 = π


def window(shape: str, n_steps: int, tukey_alpha: float = 0.5) -> np.ndarray:
    """Unit-peak window sampled at the midpoints of ``n_steps`` slices."""
    x = (np.arange(n_steps) + 0.5) / n_steps          # (0, 1)
    if shape == "square":
        return np.ones(n_steps)
    if shape == "cosine":                              # Hann
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * x))
    if shape == "blackman":
        return 0.42 - 0.5 * np.cos(2.0 * np.pi * x) + 0.08 * np.cos(4.0 * np.pi * x)
    if shape == "tukey":
        a = float(tukey_alpha)
        w = np.ones(n_steps)
        edge = x < a / 2
        w[edge] = 0.5 * (1.0 - np.cos(2.0 * np.pi * x[edge] / a))
        edge = x > 1 - a / 2
        w[edge] = 0.5 * (1.0 - np.cos(2.0 * np.pi * (1.0 - x[edge]) / a))
        return w
    raise ValueError(f"unknown window {shape!r}; choose from {WINDOWS}")


def conditional_phase(U: np.ndarray) -> float:
    """φ = arg U₀₀ − arg U₀₁ − arg U₁₀ + arg U₁₁ ∈ (−π, π] (diagonal elements)."""
    d = np.diag(U)
    return float(np.angle(d[0] * d[3] * np.conj(d[1]) * np.conj(d[2])))


def swap_leakage(U: np.ndarray) -> float:
    """Mean population transferred between |↑↓⟩ and |↓↑⟩: (|U₁₂|² + |U₂₁|²)/2."""
    return float(0.5 * (abs(U[1, 2]) ** 2 + abs(U[2, 1]) ** 2))


def cz_fidelity_up_to_z(U: np.ndarray) -> float:
    """
    Average gate fidelity to CZ after the best single-qubit Z corrections
    (closed form: the corrections cancel the phases of U₀₁, U₁₀ and U₀₀).
    """
    d = np.diag(U)
    a = np.angle(d[0])
    # remove global phase and the two local Z phases, then compare with diag(1, 1, 1, −1)
    corr = np.array([1.0, np.exp(-1j * (np.angle(d[1]) - a)), np.exp(-1j * (np.angle(d[2]) - a)),
                     np.exp(-1j * (np.angle(d[1]) + np.angle(d[2]) - 2 * a))]) * np.exp(-1j * a)
    V = np.diag(corr) @ U
    target = np.diag([1.0, 1.0, 1.0, -1.0])
    tr = np.trace(target.conj().T @ V)
    return float((abs(tr) ** 2 + 4.0) / 20.0)


@dataclass
class ShapedCZ:
    shape: str
    duration_ns: float
    j_max_mhz: float
    j_pulse: np.ndarray
    dt_ns: float
    conditional_phase: float
    swap_leakage: float
    fidelity: float

    @property
    def infidelity(self) -> float:
        return 1.0 - self.fidelity


class AdiabaticCZDesigner:
    """
    Calibrates window-shaped exchange pulses to a CZ.

    Parameters
    ----------
    hamiltonian : the two-electron model (only ΔB_z and the rotating frame matter)
    n_steps : time slices per pulse
    """

    def __init__(self, hamiltonian: Optional[SiliconSpinHamiltonian] = None, n_steps: int = 400):
        self.h = hamiltonian or SiliconSpinHamiltonian(delta_bz=20.0)
        self.dyn = ExchangeDynamics(self.h)
        self.n_steps = int(n_steps)

    def _unitary(self, shape: str, duration_ns: float, j_max: float, tukey_alpha: float) -> np.ndarray:
        dt = duration_ns / self.n_steps
        return self.dyn.propagate_unitary(j_max * window(shape, self.n_steps, tukey_alpha), dt)

    def calibrate(self, shape: str, duration_ns: float, tukey_alpha: float = 0.5) -> ShapedCZ:
        """Amplitude giving pulse area 500 MHz·ns (φ = π), then the exact propagator."""
        w = window(shape, self.n_steps, tukey_alpha)
        dt = duration_ns / self.n_steps
        j_max = CZ_AREA_MHZ_NS / (np.sum(w) * dt)
        U = self._unitary(shape, duration_ns, j_max, tukey_alpha)
        return ShapedCZ(shape=shape, duration_ns=float(duration_ns), j_max_mhz=float(j_max),
                        j_pulse=j_max * w, dt_ns=dt, conditional_phase=conditional_phase(U),
                        swap_leakage=swap_leakage(U), fidelity=cz_fidelity_up_to_z(U))

    def sweep(self, shapes: Sequence[str] = WINDOWS, durations_ns: Sequence[float] = (25, 50, 100, 200)) -> Dict[str, List[ShapedCZ]]:
        return {s: [self.calibrate(s, T) for T in durations_ns] for s in shapes}
