"""
Valley physics: an effective two-level valley degree of freedom coupled to
the two-electron spin manifold.

Silicon's conduction band has two low-lying valleys split by E_v (tens to
hundreds of µeV). This module models the *lowest valley-orbit excitation* as
a single τ (pseudo-spin) coupled to the 4-dimensional two-spin space through
an inter-valley spin-orbit term, giving an 8×8 Hamiltonian

    H₈ = H_spin ⊗ 1 + 1 ⊗ (E_v/2) τ_z + Δ_soc · (Z₁ − Z₂) ⊗ (cos φ τ_x + sin φ τ_y).

It is not a full 16-dimensional (2 electrons × 2 spins × 2 valleys) model; it
captures the leading leakage channel out of the computational valley during
fast exchange pulses. Units: MHz and ns; E_v is given in µeV.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import scipy.linalg

from .hamiltonian import HEISENBERG_EXCHANGE, Z_DIFF, SiliconSpinHamiltonian
from .units import MHZ_NS_TO_RAD, UEV_TO_MHZ

_TAU_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_TAU_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_TAU_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_I2 = np.eye(2, dtype=np.complex128)
_I4 = np.eye(4, dtype=np.complex128)


class SiliconValleyModel:
    """
    Parameters
    ----------
    valley_splitting_uev : E_v in µeV (typ. 50–300)
    valley_phase : inter-valley coupling phase φ
    inter_valley_soc_mhz : spin-orbit coupling between valleys, Δ_soc (MHz)
    """

    def __init__(
        self,
        valley_splitting_uev: float = 120.0,
        valley_phase: float = 0.35,
        inter_valley_soc_mhz: float = 2.5,
    ):
        self.valley_splitting_uev = float(valley_splitting_uev)
        self.e_valley_mhz = self.valley_splitting_uev * UEV_TO_MHZ
        self.valley_phase = float(valley_phase)
        self.soc_mhz = float(inter_valley_soc_mhz)

    def get_valley_hamiltonian(self, j_val: float, delta_bz: float = 15.0, b_0: float = 0.0) -> np.ndarray:
        """8×8 valley-spin Hamiltonian (MHz), rotating frame unless ``b_0`` ≠ 0."""
        H_spin = (j_val / 4.0) * HEISENBERG_EXCHANGE + (delta_bz / 2.0) * Z_DIFF
        if b_0:
            from .hamiltonian import Z_SUM
            H_spin = H_spin + (b_0 / 2.0) * Z_SUM
        H_val = 0.5 * self.e_valley_mhz * _TAU_Z
        coupling = np.cos(self.valley_phase) * _TAU_X + np.sin(self.valley_phase) * _TAU_Y
        return np.kron(H_spin, _I2) + np.kron(_I4, H_val) + self.soc_mhz * np.kron(Z_DIFF, coupling)

    @staticmethod
    def initial_state(j_val: float = 0.0, delta_bz: float = 0.0) -> np.ndarray:
        """|S⟩ ⊗ |v_ground⟩ — singlet in the computational (lower) valley."""
        s, _, _, _ = SiliconSpinHamiltonian.get_singlet_triplet_basis()
        ground_valley = np.array([0.0, 1.0], dtype=np.complex128)  # τ_z = −1
        return np.kron(s, ground_valley)

    def compute_valley_leakage(
        self,
        j_pulse: np.ndarray,
        dt_ns: float = 0.5,
        delta_bz: float = 15.0,
        psi_0: Optional[np.ndarray] = None,
        dt: Optional[float] = None,   # v0.2 alias
    ) -> Dict[str, Any]:
        """
        Propagate from the singlet in the ground valley and record the
        population that leaks into the excited valley at each step.
        """
        if dt is not None:
            dt_ns = dt
        psi = self.initial_state() if psi_0 is None else np.asarray(psi_0, dtype=np.complex128)
        psi = psi / np.linalg.norm(psi)
        leak = np.zeros(len(j_pulse))
        ground = np.zeros(len(j_pulse))
        for k, j_k in enumerate(j_pulse):
            H_k = self.get_valley_hamiltonian(float(j_k), delta_bz)
            psi = scipy.linalg.expm(-1.0j * MHZ_NS_TO_RAD * H_k * dt_ns) @ psi
            pops = np.abs(psi.reshape(4, 2)) ** 2
            leak[k] = float(np.clip(pops[:, 0].sum(), 0.0, 1.0))    # τ_z = +1 (excited valley)
            ground[k] = float(np.clip(pops[:, 1].sum(), 0.0, 1.0))
        return {
            "final_valley_leakage": float(leak[-1]) if len(leak) else 0.0,
            "max_valley_leakage": float(leak.max()) if len(leak) else 0.0,
            "leakage_vs_time": leak,
            "ground_valley_population": ground,
            "excited_valley_population": leak,
            "valley_splitting_mhz": self.e_valley_mhz,
            "valley_splitting_uev": self.valley_splitting_uev,
        }
