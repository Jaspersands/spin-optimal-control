"""
Derivative-Removal via Adiabatic Gate (DRAG) Corrections for Silicon Spin Qubits.

Suppresses phase errors and non-adiabatic transitions during rapid detuning ramps
by calculating analytical derivative corrections:
    epsilon_DRAG(t) = epsilon(t) - (d(epsilon)/dt) / Delta_Bz
    J_DRAG(t) = J(t) * (1 - (dJ/dt) / (J_0 * Delta_Bz))
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


class DRAGPulseSynthesizer:
    """
    Computes analytical DRAG corrections on exchange and detuning waveforms.
    """

    def __init__(self, delta_bz_mhz: float = 15.0, drag_coefficient: float = 0.5):
        self.delta_bz = delta_bz_mhz
        self.drag_coeff = drag_coefficient

    def apply_drag_correction(
        self,
        nominal_pulse: np.ndarray,
        dt_ns: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the DRAG-corrected envelope and the derivative quadrature correction.

        Returns:
            (in_phase_corrected, quadrature_correction)
        """
        # Central difference time derivative: dJ/dt (MHz / ns)
        d_pulse = np.gradient(nominal_pulse, dt_ns)

        # DRAG quadrature correction: - (drag_coeff / Delta_Bz) * (dJ/dt)
        quadrature = - (self.drag_coeff / max(abs(self.delta_bz), 1e-3)) * d_pulse

        # In-phase amplitude correction (2nd order Stark shift / adiabatic correction)
        # J_eff(t) = J(t) + (d^2 J / dt^2) / (2 * Delta_Bz^2)
        d2_pulse = np.gradient(d_pulse, dt_ns)
        in_phase = nominal_pulse + (d2_pulse / (2.0 * max(self.delta_bz**2, 1e-3)))
        in_phase = np.maximum(0.0, in_phase)

        return in_phase, quadrature
