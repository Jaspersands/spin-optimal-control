"""
Silicon Spin Qubit Exchange Hamiltonian and Dynamics.

Implements exact two-electron spin exchange dynamics in silicon quantum dots,
including detuning-dependent exchange J(epsilon), Zeeman gradients Delta B_z,
homogeneous field B_0, and differentiable unitary propagation in JAX and NumPy.
"""

from __future__ import annotations
import numpy as np
import scipy.linalg
from typing import Tuple, List, Optional, Union, Dict, Any

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = np


# Pauli matrices (2x2)
SIGMA_I = np.eye(2, dtype=np.complex128)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
SIGMA_PLUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
SIGMA_MINUS = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)

# 2-qubit tensor operators (4x4)
# Ordering: |00>, |01>, |10>, |11> (i.e. |up up>, |up down>, |down up>, |down down>)
I_4 = np.kron(SIGMA_I, SIGMA_I)
XX = np.kron(SIGMA_X, SIGMA_X)
YY = np.kron(SIGMA_Y, SIGMA_Y)
ZZ = np.kron(SIGMA_Z, SIGMA_Z)
HEISENBERG_EXCHANGE = XX + YY + ZZ  # 4 * (S1 . S2)

Z_DIFF = np.kron(SIGMA_Z, SIGMA_I) - np.kron(SIGMA_I, SIGMA_Z)  # sigma_1^z - sigma_2^z
Z_SUM = np.kron(SIGMA_Z, SIGMA_I) + np.kron(SIGMA_I, SIGMA_Z)   # sigma_1^z + sigma_2^z

X_DIFF = np.kron(SIGMA_X, SIGMA_I) - np.kron(SIGMA_I, SIGMA_X)
Y_DIFF = np.kron(SIGMA_Y, SIGMA_I) - np.kron(SIGMA_I, SIGMA_Y)


class SiliconSpinHamiltonian:
    """
    Two-electron spin Hamiltonian in a silicon double quantum dot (DQD):

        H(t) / hbar = (J(t) / 4) * (X1 X2 + Y1 Y2 + Z1 Z2)
                    + (Delta_Bz(t) / 2) * (Z1 - Z2)
                    + (B_0 / 2) * (Z1 + Z2)

    Where:
        - J(t) is the Heisenberg exchange interaction (in rad/s or MHz * 2pi)
        - Delta_Bz(t) is the gradient magnetic field (micromagnet or g-factor difference)
        - B_0 is the homogeneous Zeeman energy
    """

    def __init__(
        self,
        j_0: float = 20.0,         # Baseline exchange amplitude (MHz * 2pi)
        epsilon_0: float = 1.0,    # Characteristic detuning scale (mV)
        delta_bz: float = 15.0,    # Static Zeeman gradient (MHz * 2pi)
        b_0: float = 100.0,        # Static homogeneous Zeeman field (MHz * 2pi)
        use_ghz: bool = False,     # Units: False -> MHz, True -> GHz
    ):
        self.j_0 = j_0
        self.epsilon_0 = epsilon_0
        self.delta_bz = delta_bz
        self.b_0 = b_0
        self.scale = 1e3 if use_ghz else 1.0

    def exchange_from_detuning(
        self, epsilon: Union[float, np.ndarray, "jnp.ndarray"]
    ) -> Union[float, np.ndarray, "jnp.ndarray"]:
        """
        Calculates Heisenberg exchange coupling J(epsilon) from gate detuning voltage:
            J(epsilon) = J_0 * exp(epsilon / epsilon_0)
        """
        if HAS_JAX and isinstance(epsilon, (jax.Array, jnp.ndarray)):
            return self.j_0 * jnp.exp(epsilon / self.epsilon_0)
        return self.j_0 * np.exp(epsilon / self.epsilon_0)

    def get_hamiltonian_matrix(
        self,
        j_val: float,
        delta_bz_val: Optional[float] = None,
        b_0_val: Optional[float] = None,
    ) -> np.ndarray:
        """
        Returns the instantaneous 4x4 Hamiltonian matrix in the computational basis.
        """
        dB = self.delta_bz if delta_bz_val is None else delta_bz_val
        B0 = self.b_0 if b_0_val is None else b_0_val

        H = (
            (j_val / 4.0) * HEISENBERG_EXCHANGE
            + (dB / 2.0) * Z_DIFF
            + (B0 / 2.0) * Z_SUM
        )
        return H.astype(np.complex128)

    @staticmethod
    def get_singlet_triplet_basis() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the Singlet-Triplet basis vectors in the |00>, |01>, |10>, |11> representation:
            |S>  = (|01> - |10>) / sqrt(2)
            |T0> = (|01> + |10>) / sqrt(2)
            |T+> = |00>
            |T-> = |11>
        """
        s = np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2.0)
        t0 = np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2.0)
        tp = np.array([1, 0, 0, 0], dtype=np.complex128)
        tm = np.array([0, 0, 0, 1], dtype=np.complex128)
        return s, t0, tp, tm


class ExchangeDynamics:
    """
    Simulates coherent and noisy time evolution under piecewise or continuous
    exchange control pulses. Supports both JAX automatic differentiation
    and standard high-precision SciPy integration.
    """

    def __init__(self, hamiltonian: SiliconSpinHamiltonian):
        self.h = hamiltonian

    def propagate_unitary(
        self,
        j_pulse: np.ndarray,
        dt: float,
        delta_bz_pulse: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Propagates the unitary operator U(T) = prod_{k=N-1}^0 exp(-i H(t_k) dt).
        """
        n_steps = len(j_pulse)
        U = np.eye(4, dtype=np.complex128)

        for k in range(n_steps):
            j_k = j_pulse[k]
            dB_k = self.h.delta_bz if delta_bz_pulse is None else delta_bz_pulse[k]
            H_k = self.h.get_hamiltonian_matrix(j_k, dB_k)
            # Propagator for slice k: U_k = exp(-1j * H_k * dt)
            U_k = scipy.linalg.expm(-1.0j * H_k * dt)
            U = U_k @ U

        return U

    def propagate_state_trajectory(
        self,
        psi_0: np.ndarray,
        j_pulse: np.ndarray,
        dt: float,
        delta_bz_pulse: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes the time-resolved state vector trajectory [psi(t_0), psi(t_1), ..., psi(t_N)].
        Shape: (n_steps + 1, 4)
        """
        n_steps = len(j_pulse)
        trajectory = np.zeros((n_steps + 1, 4), dtype=np.complex128)
        trajectory[0] = psi_0 / np.linalg.norm(psi_0)

        current_psi = trajectory[0].copy()
        for k in range(n_steps):
            j_k = j_pulse[k]
            dB_k = self.h.delta_bz if delta_bz_pulse is None else delta_bz_pulse[k]
            H_k = self.h.get_hamiltonian_matrix(j_k, dB_k)
            U_k = scipy.linalg.expm(-1.0j * H_k * dt)
            current_psi = U_k @ current_psi
            trajectory[k + 1] = current_psi

        return trajectory

    def compute_singlet_triplet_populations(
        self, trajectory: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Projects state trajectory onto {|S>, |T0>, |T+>, |T->} basis.
        """
        s, t0, tp, tm = SiliconSpinHamiltonian.get_singlet_triplet_basis()
        p_s = np.abs(trajectory @ s.conj()) ** 2
        p_t0 = np.abs(trajectory @ t0.conj()) ** 2
        p_tp = np.abs(trajectory @ tp.conj()) ** 2
        p_tm = np.abs(trajectory @ tm.conj()) ** 2

        return {
            "P_S": p_s,
            "P_T0": p_t0,
            "P_T+": p_tp,
            "P_T-": p_tm,
        }

    @staticmethod
    def target_gate_sqrt_swap() -> np.ndarray:
        """
        Ideal sqrt(SWAP) unitary matrix in computational basis:
            |00> -> |00>
            |01> -> (1+i)/2 |01> + (1-i)/2 |10>
            |10> -> (1-i)/2 |01> + (1+i)/2 |10>
            |11> -> |11>
        """
        u = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.5 * (1 + 1j), 0.5 * (1 - 1j), 0.0],
                [0.0, 0.5 * (1 - 1j), 0.5 * (1 + 1j), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.complex128,
        )
        return u

    @staticmethod
    def target_gate_swap() -> np.ndarray:
        """Ideal SWAP gate."""
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.complex128,
        )

    @staticmethod
    def target_gate_cz() -> np.ndarray:
        """Ideal Controlled-Z gate."""
        return np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)

    @staticmethod
    def gate_fidelity(U_actual: np.ndarray, U_target: np.ndarray) -> float:
        """
        Calculates average gate fidelity / trace fidelity on U(d=4):
            F(U, U_t) = (1 / d^2) * |Tr(U_t^dagger U)|^2
        """
        d = U_target.shape[0]
        overlap = np.trace(U_target.conj().T @ U_actual)
        return float(np.abs(overlap) ** 2 / (d * d))
