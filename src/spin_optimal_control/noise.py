"""
Realistic Noise Modeling for Silicon Spin Qubits.

Implements:
1. 1/f^alpha pink charge noise (Lorentzian multi-trap & spectral synthesis).
2. Quasistatic Overhauser nuclear field fluctuations (residual 29Si spin bath).
3. Lindblad master equation for open quantum system dynamics (T1 relaxation, T2* dephasing).
4. SiliconNoiseModel for integration with Cirq and pulse simulation.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import scipy.linalg


class PinkNoiseGenerator:
    """
    Generates 1/f^alpha pink charge noise time series S(f) = A_eps / f^alpha.
    Can use either:
    1. Spectral filtering via colored Gaussian noise in frequency domain.
    2. Sum of fluctuating Two-Level Fluctuators (TLFs / charge traps) undergoing
       Ornstein-Uhlenbeck or random telegraph switching.
    """

    def __init__(
        self,
        alpha: float = 1.0,           # Exponent alpha in 1/f^alpha (typically 0.8 - 1.2)
        amplitude: float = 0.05,      # RMS noise amplitude (in detuning units, e.g. mV or MHz)
        f_min: float = 1e-4,          # Low-frequency cutoff (Hz)
        f_max: float = 1e8,           # High-frequency cutoff (Hz)
        seed: Optional[int] = None,
    ):
        self.alpha = alpha
        self.amplitude = amplitude
        self.f_min = f_min
        self.f_max = f_max
        self.rng = np.random.default_rng(seed)

    def generate_spectral_trace(self, n_steps: int, dt: float) -> np.ndarray:
        """
        Generates a 1/f^alpha noise trajectory using spectral synthesis with inverse FFT.
        """
        if n_steps <= 1:
            return np.zeros(n_steps)

        # Frequencies
        freqs = np.fft.rfftfreq(n_steps, d=dt)
        freqs[0] = 1e-9  # Avoid division by zero at DC

        # Power spectral density S(f) ~ 1 / f^alpha
        # Amplitude spectrum ~ 1 / f^(alpha/2)
        amp_filter = 1.0 / (freqs ** (self.alpha / 2.0))

        # Filter band limits
        amp_filter[freqs < self.f_min] = 0.0
        amp_filter[freqs > self.f_max] = 0.0

        # Complex Gaussian white noise
        white_noise = self.rng.standard_normal(len(freqs)) + 1j * self.rng.standard_normal(len(freqs))
        colored_freq = white_noise * amp_filter

        # Inverse FFT
        noise_trace = np.fft.irfft(colored_freq, n=n_steps)

        # Normalize to target RMS amplitude
        current_std = np.std(noise_trace)
        if current_std > 1e-12:
            noise_trace = (noise_trace / current_std) * self.amplitude
        else:
            noise_trace = np.zeros(n_steps)

        return noise_trace

    def generate_multitrap_charge_noise(
        self, n_steps: int, dt: float, n_traps: int = 15
    ) -> np.ndarray:
        """
        Generates charge noise from a sum of independent Two-Level Fluctuators (TLFs)
        with log-uniformly distributed switching rates gamma_i in [f_min, f_max].
        """
        gamma_rates = np.logspace(
            np.log10(self.f_min), np.log10(self.f_max), n_traps
        )
        total_noise = np.zeros(n_steps)

        for gamma in gamma_rates:
            # Discrete-time Ornstein-Uhlenbeck / telegraph update
            decay = np.exp(-gamma * dt)
            var_increment = 1.0 - decay**2
            trap_trace = np.zeros(n_steps)
            val = self.rng.standard_normal()

            for step in range(n_steps):
                val = val * decay + np.sqrt(var_increment) * self.rng.standard_normal()
                trap_trace[step] = val

            total_noise += trap_trace

        # Rescale
        std_val = np.std(total_noise)
        if std_val > 1e-12:
            total_noise = (total_noise / std_val) * self.amplitude

        return total_noise


class OverhauserNoise:
    """
    Simulates quasistatic nuclear spin fluctuations in isotopically enriched Silicon-28.
    The Overhauser field delta_Bz fluctuates slowly across experiment runs according
    to a Gaussian distribution delta(Delta Bz) ~ N(0, sigma_N^2).
    """

    def __init__(
        self,
        sigma_overhauser: float = 0.8,  # Standard deviation of nuclear field (MHz * 2pi)
        correlation_time_s: float = 1.0, # Slow diffusion correlation time
        seed: Optional[int] = None,
    ):
        self.sigma = sigma_overhauser
        self.t_corr = correlation_time_s
        self.rng = np.random.default_rng(seed)

    def sample_quasistatic_shift(self) -> float:
        """Draws a random Zeeman gradient shift for a single shot or Ramsey sequence."""
        return float(self.rng.normal(0.0, self.sigma))

    def sample_ensemble_shifts(self, n_samples: int) -> np.ndarray:
        """Draws an ensemble of quasistatic Overhauser gradient shifts."""
        return self.rng.normal(0.0, self.sigma, size=n_samples)


class SiliconNoiseModel:
    """
    Unified noise configuration for Silicon Quantum Dots, combining:
    - Charge noise (detuning fluctuations delta_epsilon)
    - Nuclear spin noise (Overhauser Zeeman drift)
    - T1 relaxation time and T2* dephasing time
    """

    def __init__(
        self,
        t1_us: float = 1000.0,        # Longitudinal relaxation time T1 (microseconds)
        t2_star_us: float = 20.0,     # Inhomogeneous dephasing time T2* (microseconds)
        charge_noise_amp: float = 0.03, # RMS charge noise amplitude (mV)
        overhauser_sigma: float = 0.5, # Overhauser gradient std (MHz * 2pi)
        seed: Optional[int] = None,
    ):
        self.t1 = t1_us
        self.t2_star = t2_star_us
        self.charge_noise_amp = charge_noise_amp
        self.overhauser_sigma = overhauser_sigma

        self.pink_gen = PinkNoiseGenerator(
            alpha=1.0, amplitude=charge_noise_amp, seed=seed
        )
        self.overhauser = OverhauserNoise(
            sigma_overhauser=overhauser_sigma, seed=seed
        )

    def apply_noisy_pulses(
        self,
        j_nominal: np.ndarray,
        epsilon_0: float,
        dt: float,
        delta_bz_nominal: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the instantaneous noisy exchange J_noisy(t) and delta_Bz_noisy(t)
        resulting from charge noise and nuclear spin fluctuations.
        """
        n_steps = len(j_nominal)
        delta_eps = self.pink_gen.generate_spectral_trace(n_steps, dt)

        # delta_J ~ J * (delta_eps / epsilon_0)
        # J_noisy(t) = J(t) * exp(delta_eps(t) / epsilon_0)
        j_noisy = j_nominal * np.exp(delta_eps / max(epsilon_0, 1e-4))

        # Overhauser shift
        oh_shift = self.overhauser.sample_quasistatic_shift()
        delta_bz_noisy = np.full(n_steps, delta_bz_nominal + oh_shift)

        return j_noisy, delta_bz_noisy

    def get_lindblad_jump_operators(self) -> List[Tuple[np.ndarray, float]]:
        """
        Returns the set of Lindblad collapse operators (L_k, rate_k) for both qubits.
        Rates in MHz (1 / us).
        """
        gamma_1 = 1.0 / self.t1 if self.t1 > 0 else 0.0
        # Pure dephasing rate: 1/T_phi = 1/T2* - 1/(2*T1)
        gamma_phi = max(0.0, (1.0 / self.t2_star) - (0.5 * gamma_1))

        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        sigma_m = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)
        id2 = np.eye(2, dtype=np.complex128)

        # Qubit 1 jump operators
        z1 = np.kron(sigma_z, id2)
        m1 = np.kron(sigma_m, id2)

        # Qubit 2 jump operators
        z2 = np.kron(id2, sigma_z)
        m2 = np.kron(id2, sigma_m)

        operators = [
            (z1, gamma_phi),
            (z2, gamma_phi),
            (m1, gamma_1),
            (m2, gamma_1),
        ]
        return operators

    def evolve_density_matrix_lindblad(
        self,
        rho_0: np.ndarray,
        hamiltonian_func,
        j_pulse: np.ndarray,
        dt: float,
        delta_bz_pulse: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solves Lindblad master equation:
        d(rho)/dt = -i [H, rho] + sum_k gamma_k (L_k rho L_k^dagger - 1/2 {L_k^dagger L_k, rho})
        using 4th-order Runge-Kutta.
        """
        n_steps = len(j_pulse)
        rho = rho_0.copy().astype(np.complex128)
        jumps = self.get_lindblad_jump_operators()

        def d_rho_dt(current_rho: np.ndarray, step_idx: int) -> np.ndarray:
            j_k = j_pulse[step_idx]
            dB_k = None if delta_bz_pulse is None else delta_bz_pulse[step_idx]
            H = hamiltonian_func(j_k, dB_k)

            # Commutator -i [H, rho]
            comm = -1.0j * (H @ current_rho - current_rho @ H)

            # Dissipator
            diss = np.zeros((4, 4), dtype=np.complex128)
            for L_k, rate in jumps:
                if rate > 0:
                    L_dag_L = L_k.conj().T @ L_k
                    term = L_k @ current_rho @ L_k.conj().T - 0.5 * (L_dag_L @ current_rho + current_rho @ L_dag_L)
                    diss += rate * term

            return comm + diss

        for k in range(n_steps):
            k1 = d_rho_dt(rho, k)
            k2 = d_rho_dt(rho + 0.5 * dt * k1, k)
            k3 = d_rho_dt(rho + 0.5 * dt * k2, k)
            k4 = d_rho_dt(rho + dt * k3, k)
            rho += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return rho
