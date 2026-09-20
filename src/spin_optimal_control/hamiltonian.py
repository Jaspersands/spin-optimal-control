"""
Silicon Spin Qubit Exchange Hamiltonian and Dynamics.

Two-electron spin exchange dynamics in a silicon double quantum dot (DQD):
detuning-dependent exchange J(ε), Zeeman gradient ΔB_z, homogeneous field B_0,
unitary propagation (NumPy and JAX), gate targets and fidelity measures.

Units: frequencies in MHz, times in ns (see :mod:`spin_optimal_control.units`).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.linalg
import scipy.optimize

from .units import MHZ_NS_TO_RAD

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only without JAX installed
    HAS_JAX = False
    jax = None
    jnp = np


# --------------------------------------------------------------------------- #
# Pauli matrices and two-qubit operators
# --------------------------------------------------------------------------- #
SIGMA_I = np.eye(2, dtype=np.complex128)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
SIGMA_PLUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
SIGMA_MINUS = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)

# Computational basis ordering: |00>, |01>, |10>, |11>  (|↑↑>, |↑↓>, |↓↑>, |↓↓>)
I_4 = np.kron(SIGMA_I, SIGMA_I)
XX = np.kron(SIGMA_X, SIGMA_X)
YY = np.kron(SIGMA_Y, SIGMA_Y)
ZZ = np.kron(SIGMA_Z, SIGMA_Z)
HEISENBERG_EXCHANGE = XX + YY + ZZ  # = 4 (S1 · S2)

Z_DIFF = np.kron(SIGMA_Z, SIGMA_I) - np.kron(SIGMA_I, SIGMA_Z)  # σ1^z − σ2^z
Z_SUM = np.kron(SIGMA_Z, SIGMA_I) + np.kron(SIGMA_I, SIGMA_Z)   # σ1^z + σ2^z
X_DIFF = np.kron(SIGMA_X, SIGMA_I) - np.kron(SIGMA_I, SIGMA_X)
Y_DIFF = np.kron(SIGMA_Y, SIGMA_I) - np.kron(SIGMA_I, SIGMA_Y)


def _rz(theta: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


class SiliconSpinHamiltonian:
    """
    Two-electron spin Hamiltonian in a silicon double quantum dot:

        H(t)/h = (J(t)/4) (X1X2 + Y1Y2 + Z1Z2) + (ΔB_z/2)(Z1 − Z2) + (B_0/2)(Z1 + Z2)

    with ``J`` the Heisenberg exchange, ``ΔB_z`` the Zeeman gradient
    (micromagnet / g-factor difference) and ``B_0`` the homogeneous Zeeman
    energy, all in MHz.

    ``rotating_frame=True`` (default) works in the frame rotating at the mean
    Zeeman frequency, where the ``B_0`` term - which commutes with every other
    term and only produces known single-qubit Z phases - is dropped. This is the
    frame in which gate fidelities are quoted experimentally.
    """

    def __init__(
        self,
        j_0: float = 20.0,        # Baseline exchange amplitude at ε = 0 (MHz)
        epsilon_0: float = 1.0,   # Characteristic detuning lever-arm scale (mV)
        delta_bz: float = 15.0,   # Static Zeeman gradient (MHz)
        b_0: float = 1000.0,      # Homogeneous Zeeman splitting (MHz), lab frame only
        rotating_frame: bool = True,
    ):
        self.j_0 = float(j_0)
        self.epsilon_0 = float(epsilon_0)
        self.delta_bz = float(delta_bz)
        self.b_0 = float(b_0)
        self.rotating_frame = bool(rotating_frame)

    # ----------------------------------------------------------------- #
    def exchange_from_detuning(self, epsilon):
        """J(ε) = J_0 · exp(ε / ε_0)  (MHz)."""
        if HAS_JAX and isinstance(epsilon, jax.Array):
            return self.j_0 * jnp.exp(epsilon / self.epsilon_0)
        return self.j_0 * np.exp(np.asarray(epsilon) / self.epsilon_0)

    def detuning_from_exchange(self, j_mhz, j_floor: float = 1e-4):
        """Inverse map ε(J) = ε_0 · ln(J / J_0), floored so ln stays finite."""
        return self.epsilon_0 * np.log(np.maximum(np.asarray(j_mhz) / self.j_0, j_floor))

    def get_hamiltonian_matrix(
        self,
        j_val: float,
        delta_bz_val: Optional[float] = None,
        b_0_val: Optional[float] = None,
        lab_frame: bool = False,
    ) -> np.ndarray:
        """
        Instantaneous 4×4 Hamiltonian (MHz) in the computational basis.
        The ``B_0`` term is included only in the lab frame
        (``lab_frame=True`` or ``rotating_frame=False``).
        """
        dB = self.delta_bz if delta_bz_val is None else delta_bz_val
        B0 = self.b_0 if b_0_val is None else b_0_val

        H = (j_val / 4.0) * HEISENBERG_EXCHANGE + (dB / 2.0) * Z_DIFF
        if lab_frame or not self.rotating_frame:
            H = H + (B0 / 2.0) * Z_SUM
        return H.astype(np.complex128)

    def spectrum(self, j_val: float, delta_bz_val: Optional[float] = None) -> np.ndarray:
        """Eigenvalues (MHz) of the rotating-frame Hamiltonian at exchange ``j_val``."""
        return np.linalg.eigvalsh(self.get_hamiltonian_matrix(j_val, delta_bz_val))

    @staticmethod
    def get_singlet_triplet_basis() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Singlet-triplet basis in the |00>,|01>,|10>,|11> representation:
            |S>  = (|01> − |10>)/√2,  |T0> = (|01> + |10>)/√2,  |T+> = |00>,  |T-> = |11>
        """
        s = np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2.0)
        t0 = np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2.0)
        tp = np.array([1, 0, 0, 0], dtype=np.complex128)
        tm = np.array([0, 0, 0, 1], dtype=np.complex128)
        return s, t0, tp, tm


class ExchangeDynamics:
    """
    Coherent time evolution under piecewise-constant exchange pulses, gate
    targets, and fidelity measures.
    """

    def __init__(self, hamiltonian: SiliconSpinHamiltonian):
        self.h = hamiltonian

    # ----------------------------------------------------------------- #
    # Propagation
    # ----------------------------------------------------------------- #
    def step_unitary(self, j_val: float, dt_ns: float, delta_bz_val: Optional[float] = None) -> np.ndarray:
        """U_k = exp(−i · 2π·1e-3 · H(J_k) · dt)."""
        H_k = self.h.get_hamiltonian_matrix(j_val, delta_bz_val)
        return scipy.linalg.expm(-1.0j * MHZ_NS_TO_RAD * H_k * dt_ns)

    def propagate_unitary(
        self,
        j_pulse: np.ndarray,
        dt_ns: float,
        delta_bz_pulse: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Total propagator U(T) = U_{N-1} ⋯ U_1 U_0 for a piecewise-constant pulse."""
        U = np.eye(4, dtype=np.complex128)
        for k, j_k in enumerate(j_pulse):
            dB_k = None if delta_bz_pulse is None else delta_bz_pulse[k]
            U = self.step_unitary(float(j_k), dt_ns, dB_k) @ U
        return U

    def propagate_state_trajectory(
        self,
        psi_0: np.ndarray,
        j_pulse: np.ndarray,
        dt_ns: float,
        delta_bz_pulse: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Time-resolved state vector trajectory, shape ``(n_steps + 1, 4)``."""
        n_steps = len(j_pulse)
        trajectory = np.zeros((n_steps + 1, 4), dtype=np.complex128)
        trajectory[0] = psi_0 / np.linalg.norm(psi_0)
        psi = trajectory[0].copy()
        for k, j_k in enumerate(j_pulse):
            dB_k = None if delta_bz_pulse is None else delta_bz_pulse[k]
            psi = self.step_unitary(float(j_k), dt_ns, dB_k) @ psi
            trajectory[k + 1] = psi
        return trajectory

    def compute_singlet_triplet_populations(self, trajectory: np.ndarray) -> Dict[str, np.ndarray]:
        """Projects a state trajectory onto {|S>, |T0>, |T+>, |T->}."""
        s, t0, tp, tm = SiliconSpinHamiltonian.get_singlet_triplet_basis()
        return {
            "P_S": np.abs(trajectory @ s.conj()) ** 2,
            "P_T0": np.abs(trajectory @ t0.conj()) ** 2,
            "P_T+": np.abs(trajectory @ tp.conj()) ** 2,
            "P_T-": np.abs(trajectory @ tm.conj()) ** 2,
        }

    # ----------------------------------------------------------------- #
    # Gate targets
    # ----------------------------------------------------------------- #
    @staticmethod
    def exchange_gate(angle_rad: float) -> np.ndarray:
        """
        Pure exchange gate exp(−i·angle/4·(σ1·σ2 − 1)); ``angle=π`` is SWAP,
        ``π/2`` is √SWAP, ``π/4`` is SWAP^(1/4).
        """
        return scipy.linalg.expm(-0.25j * angle_rad * (HEISENBERG_EXCHANGE - I_4))

    @staticmethod
    def target_gate_sqrt_swap() -> np.ndarray:
        """√SWAP:  |01> → (1+i)/2 |01> + (1−i)/2 |10>, |10> ↔ likewise."""
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.5 * (1 + 1j), 0.5 * (1 - 1j), 0.0],
                [0.0, 0.5 * (1 - 1j), 0.5 * (1 + 1j), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.complex128,
        )

    @staticmethod
    def target_gate_swap() -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
        )

    @staticmethod
    def target_gate_fourth_swap() -> np.ndarray:
        """SWAP^(1/4): exchange angle π/4."""
        return ExchangeDynamics.exchange_gate(np.pi / 4.0)

    @staticmethod
    def target_gate_cz() -> np.ndarray:
        return np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)

    # ----------------------------------------------------------------- #
    # Fidelity measures
    # ----------------------------------------------------------------- #
    @staticmethod
    def gate_fidelity(U_actual: np.ndarray, U_target: np.ndarray) -> float:
        """
        Global-phase-invariant process (entanglement) fidelity
            F_pro = |Tr(U_t† U)|² / d².
        """
        d = U_target.shape[0]
        overlap = np.trace(U_target.conj().T @ U_actual)
        return float(np.abs(overlap) ** 2 / (d * d))

    @staticmethod
    def average_gate_fidelity(U_actual: np.ndarray, U_target: np.ndarray) -> float:
        """Nielsen's relation  F_avg = (d·F_pro + 1)/(d + 1)."""
        d = U_target.shape[0]
        return (d * ExchangeDynamics.gate_fidelity(U_actual, U_target) + 1.0) / (d + 1.0)

    @staticmethod
    def gate_fidelity_local_z_free(
        U_actual: np.ndarray, U_target: np.ndarray, n_starts: int = 4
    ) -> Tuple[float, np.ndarray]:
        """
        Process fidelity maximised over virtual single-qubit Z rotations applied
        before and after the gate:

            max_{a,b,c,d}  F( Rz(c)⊗Rz(d) · U · Rz(a)⊗Rz(b),  U_target ).

        Virtual-Z gates are free (frame updates) on spin qubits, so this is the
        experimentally relevant figure of merit for CZ-type gates synthesised
        under a Zeeman gradient. Returns ``(F, [a, b, c, d])``.
        """

        def neg_fid(x: np.ndarray) -> float:
            pre = np.kron(_rz(x[0]), _rz(x[1]))
            post = np.kron(_rz(x[2]), _rz(x[3]))
            return -ExchangeDynamics.gate_fidelity(post @ U_actual @ pre, U_target)

        best_f, best_x = -1.0, np.zeros(4)
        rng = np.random.default_rng(0)
        starts = [np.zeros(4)] + [rng.uniform(-np.pi, np.pi, 4) for _ in range(n_starts - 1)]
        for x0 in starts:
            res = scipy.optimize.minimize(neg_fid, x0, method="Nelder-Mead",
                                          options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000})
            res = scipy.optimize.minimize(neg_fid, res.x, method="BFGS",
                                          options={"gtol": 1e-12})
            if -res.fun > best_f:
                best_f, best_x = -res.fun, res.x
        return float(min(best_f, 1.0)), np.asarray(best_x)


# --------------------------------------------------------------------------- #
# JAX propagation helpers (shared with the GRAPE optimiser)
# --------------------------------------------------------------------------- #
if HAS_JAX:

    def propagate_unitary_jax(
        j_vals: "jnp.ndarray", dt_ns: float, delta_bz: float
    ) -> "jnp.ndarray":
        """Differentiable propagator for a piecewise-constant pulse (rotating frame)."""
        H_ex = jnp.asarray(HEISENBERG_EXCHANGE)
        H_db = jnp.asarray(Z_DIFF)

        def body(U, j_k):
            H_k = (j_k / 4.0) * H_ex + (delta_bz / 2.0) * H_db
            U_k = jax.scipy.linalg.expm(-1.0j * MHZ_NS_TO_RAD * H_k * dt_ns)
            return U_k @ U, None

        U0 = jnp.eye(4, dtype=jnp.complex128)
        U, _ = jax.lax.scan(body, U0, j_vals)
        return U
