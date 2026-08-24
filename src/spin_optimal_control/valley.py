"""
Silicon Valley Physics and Multi-Valley Leakage Modeling.

Models the six-fold conduction band valley degeneracy in silicon, valley splitting E_v,
valley-orbit coupling Delta_v, and non-adiabatic leakage between valley-spin manifolds
{|S, v_0>, |T_0, v_0>, |T_+, v_0>, |T_-, v_0>, |S, v_1>, |T_0, v_1>}.
"""

from __future__ import annotations
import numpy as np
import scipy.linalg
from typing import Tuple, List, Optional, Dict, Any


class SiliconValleyModel:
    """
    Simulates valley-spin dynamics in silicon quantum dots with finite valley splitting.

    Hilbert space: 8 dimensions (2 electrons x 2 spin states x 2 valley states v_-, v_+)
    or the low-energy 6-state manifold consisting of the two lowest valley orbitals.
    """

    def __init__(
        self,
        valley_splitting_uev: float = 120.0, # Valley splitting E_v in micro-electronvolts (typically 50-300 ueV)
        valley_phase: float = 0.35,          # Inter-valley coupling phase phi_v
        inter_valley_soc_mhz: float = 2.5,   # Inter-valley spin-orbit coupling (MHz)
    ):
        # 1 ueV corresponds to ~ 241.8 MHz
        self.e_valley_mhz = valley_splitting_uev * 241.7989
        self.valley_phase = valley_phase
        self.soc_mhz = inter_valley_soc_mhz

    def get_valley_hamiltonian(
        self,
        j_val: float,
        delta_bz: float = 15.0,
        b_0: float = 100.0,
    ) -> np.ndarray:
        """
        Builds the 8x8 valley-spin Hamiltonian:
            H_8 = H_spin (x) I_valley + I_spin (x) H_valley + H_spin_valley_coupling
        """
        from .hamiltonian import HEISENBERG_EXCHANGE, Z_DIFF, Z_SUM

        # 4x4 Spin Hamiltonian
        H_spin = (
            (j_val / 4.0) * HEISENBERG_EXCHANGE
            + (delta_bz / 2.0) * Z_DIFF
            + (b_0 / 2.0) * Z_SUM
        )

        # 2x2 Valley Hamiltonian (valley splitting E_v along z and inter-valley tunnel Delta_v along x/y)
        tau_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        tau_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        tau_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        id2 = np.eye(2, dtype=np.complex128)
        id4 = np.eye(4, dtype=np.complex128)

        H_valley_single = 0.5 * self.e_valley_mhz * tau_z
        # Single-particle valley projection for dot 1
        H_v = np.kron(id4, H_valley_single)

        # Spin part
        H_total = np.kron(H_spin, id2) + H_v

        # Inter-valley spin-orbit coupling
        H_soc = self.soc_mhz * (
            np.kron(Z_DIFF, np.cos(self.valley_phase) * tau_x + np.sin(self.valley_phase) * tau_y)
        )
        H_total += H_soc

        return H_total

    def compute_valley_leakage(
        self,
        j_pulse: np.ndarray,
        dt: float,
        delta_bz: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Simulates time evolution starting in the lowest valley ground state |S, v_0>
        and calculates non-adiabatic leakage probability into excited valley states |v_1>.
        """
        H_0 = self.get_valley_hamiltonian(j_pulse[0], delta_bz)
        evals, evecs = np.linalg.eigh(H_0)

        # Initial state: Ground valley Singlet state |S, v_0>
        psi = evecs[:, 0]
        n_steps = len(j_pulse)

        excited_valley_pop = []
        ground_valley_pop = []

        for step in range(n_steps):
            H_k = self.get_valley_hamiltonian(j_pulse[step], delta_bz)
            U_k = scipy.linalg.expm(-1.0j * H_k * dt)
            psi = U_k @ psi

            # Population in ground valley subspace (first 4 components) vs excited valley (last 4)
            p_ground = float(np.clip(np.sum(np.abs(psi[:4]) ** 2), 0.0, 1.0))
            p_excited = float(np.clip(np.sum(np.abs(psi[4:]) ** 2), 0.0, 1.0))
            ground_valley_pop.append(p_ground)
            excited_valley_pop.append(p_excited)

        final_leakage = float(np.clip(excited_valley_pop[-1], 0.0, 1.0))

        return {
            "final_valley_leakage": final_leakage,
            "ground_valley_population": np.array(ground_valley_pop),
            "excited_valley_population": np.array(excited_valley_pop),
            "valley_splitting_mhz": self.e_valley_mhz,
        }
