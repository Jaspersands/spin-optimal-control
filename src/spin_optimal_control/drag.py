"""
Derivative-based (DRAG-style) corrections for exchange / detuning ramps.

For a singlet–triplet qubit driven by J(t) in the presence of a fixed Zeeman
gradient ΔB_z, non-adiabatic ramps produce phase errors proportional to
dJ/dt / ΔB_z. The first-order derivative correction adds a quadrature
component and the second-order (super-adiabatic) term shifts the in-phase
amplitude:

    Ω_y(t)   = −β · (dJ/dt) / ΔB_z
    J_eff(t) = J(t) + (d²J/dt²) / (2 ΔB_z²)

These are analytical adiabatic-expansion corrections (Motzoi et al. 2009
applied to the S–T₀ subspace), not a full optimal-control solution. Both
vanish identically for a constant pulse.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class DRAGPulseSynthesizer:
    """Computes derivative corrections for a nominal exchange waveform."""

    def __init__(self, delta_bz_mhz: float = 15.0, drag_coefficient: float = 0.5):
        self.delta_bz = float(delta_bz_mhz)
        self.drag_coeff = float(drag_coefficient)

    def apply_drag_correction(self, nominal_pulse: np.ndarray, dt_ns: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns ``(in_phase, quadrature)`` in MHz for a pulse sampled every ``dt_ns``.
        ``quadrature = −β · J'/ΔB_z`` and ``in_phase = max(0, J + J''/(2ΔB_z²))``.
        """
        pulse = np.asarray(nominal_pulse, dtype=float)
        if pulse.size < 2:
            return pulse.copy(), np.zeros_like(pulse)
        d_pulse = np.gradient(pulse, dt_ns)
        d2_pulse = np.gradient(d_pulse, dt_ns)
        db = max(abs(self.delta_bz), 1e-3)
        quadrature = -(self.drag_coeff / db) * d_pulse
        in_phase = np.maximum(0.0, pulse + d2_pulse / (2.0 * db**2))
        return in_phase, quadrature
