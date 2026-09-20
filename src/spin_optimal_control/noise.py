"""
Noise models for silicon spin qubits.

1. 1/f^α charge noise on the detuning (spectral synthesis, or a sum of
   Ornstein–Uhlenbeck two-level fluctuators with log-uniform switching rates).
2. Quasi-static Overhauser (nuclear-spin) shifts of the Zeeman gradient.
3. Lindblad master equation with T1 relaxation and pure dephasing, solved
   exactly per time slice by exponentiating the 16×16 Liouvillian.
4. First-order filter-function estimate of dephasing infidelity.

Units: J and ΔB_z in MHz, time steps in ns, T1 / T2* in µs, PSD cut-offs in Hz.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg
import scipy.signal

from .units import MHZ_NS_TO_RAD, PER_US_TO_PER_NS


# --------------------------------------------------------------------------- #
# 1/f charge noise
# --------------------------------------------------------------------------- #
class PinkNoiseGenerator:
    """
    Generates 1/f^α noise traces with a prescribed RMS amplitude.

    Parameters
    ----------
    alpha : spectral exponent (0 = white, 1 = pink, 2 = Brownian)
    amplitude : RMS of the returned trace (detuning units, e.g. mV)
    f_min_hz, f_max_hz : band limits of the synthesised spectrum (Hz)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        amplitude: float = 0.05,
        f_min_hz: float = 1e2,
        f_max_hz: float = 1e9,
        seed: Optional[int] = None,
        f_min: Optional[float] = None,   # v0.2 aliases
        f_max: Optional[float] = None,
    ):
        self.alpha = float(alpha)
        self.amplitude = float(amplitude)
        self.f_min_hz = float(f_min if f_min is not None else f_min_hz)
        self.f_max_hz = float(f_max if f_max is not None else f_max_hz)
        self.rng = np.random.default_rng(seed)

    def generate_spectral_trace(self, n_steps: int, dt_ns: float) -> np.ndarray:
        """1/f^α trace of length ``n_steps`` sampled every ``dt_ns`` (spectral synthesis)."""
        n_steps = int(n_steps)
        if n_steps <= 1:
            return np.zeros(n_steps)
        dt_s = float(dt_ns) * 1e-9
        freqs = np.fft.rfftfreq(n_steps, d=dt_s)             # Hz
        amp = np.zeros_like(freqs)
        band = (freqs > 0) & (freqs >= self.f_min_hz) & (freqs <= self.f_max_hz)
        amp[band] = freqs[band] ** (-self.alpha / 2.0)       # amplitude ∝ f^{-α/2}
        white = self.rng.standard_normal(len(freqs)) + 1j * self.rng.standard_normal(len(freqs))
        trace = np.fft.irfft(white * amp, n=n_steps)
        std = np.std(trace)
        if std < 1e-15:
            return np.zeros(n_steps)
        return trace / std * self.amplitude

    def generate_multitrap_charge_noise(self, n_steps: int, dt_ns: float, n_traps: int = 15) -> np.ndarray:
        """
        Sum of ``n_traps`` Ornstein–Uhlenbeck fluctuators with switching rates
        log-spaced in [f_min, f_max] Hz; a superposition of Lorentzians whose
        envelope approximates 1/f.
        """
        n_steps = int(n_steps)
        if n_steps <= 0:
            return np.zeros(0)
        dt_s = float(dt_ns) * 1e-9
        rates = np.logspace(np.log10(self.f_min_hz), np.log10(self.f_max_hz), int(n_traps))
        total = np.zeros(n_steps)
        for gamma in rates:
            decay = np.exp(-gamma * dt_s)
            xi = self.rng.standard_normal(n_steps)
            # x_t = decay · x_{t-1} + sqrt(1 - decay²) · ξ_t   (stationary OU)
            x = scipy.signal.lfilter([np.sqrt(max(1.0 - decay**2, 0.0))], [1.0, -decay], xi)
            x[0] = self.rng.standard_normal()
            total += x
        std = np.std(total)
        return total / std * self.amplitude if std > 1e-15 else total

    @staticmethod
    def psd_estimate(trace: np.ndarray, dt_ns: float) -> Tuple[np.ndarray, np.ndarray]:
        """One-sided periodogram (Hz, units²/Hz) excluding DC."""
        n = len(trace)
        dt_s = float(dt_ns) * 1e-9
        f = np.fft.rfftfreq(n, d=dt_s)
        spec = np.fft.rfft(trace - np.mean(trace))
        psd = (np.abs(spec) ** 2) * dt_s / n
        psd[1:-1] *= 2.0
        return f[1:], psd[1:]


# --------------------------------------------------------------------------- #
# Overhauser field
# --------------------------------------------------------------------------- #
class OverhauserNoise:
    """
    Quasi-static Zeeman-gradient shifts from residual ²⁹Si nuclear spins,
    δ(ΔB_z) ~ N(0, σ_N²), constant during one shot and re-drawn between shots.
    """

    def __init__(self, sigma_overhauser: float = 0.8, correlation_time_s: float = 1.0, seed: Optional[int] = None):
        self.sigma = float(sigma_overhauser)
        self.t_corr = float(correlation_time_s)
        self.rng = np.random.default_rng(seed)

    def sample_quasistatic_shift(self) -> float:
        return float(self.rng.normal(0.0, self.sigma))

    def sample_ensemble_shifts(self, n_samples: int) -> np.ndarray:
        return self.rng.normal(0.0, self.sigma, size=int(n_samples))


# --------------------------------------------------------------------------- #
# Lindblad open-system model
# --------------------------------------------------------------------------- #
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_SM = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)
_I2 = np.eye(2, dtype=np.complex128)
_I4 = np.eye(4, dtype=np.complex128)


def _superop_hamiltonian(H: np.ndarray) -> np.ndarray:
    """Column-stacking superoperator of ρ ↦ −i[H, ρ]."""
    return -1.0j * (np.kron(_I4, H) - np.kron(H.T, _I4))


def _superop_dissipator(L: np.ndarray, rate: float) -> np.ndarray:
    """Column-stacking superoperator of ρ ↦ γ (LρL† − ½{L†L, ρ})."""
    LdL = L.conj().T @ L
    return rate * (np.kron(L.conj(), L) - 0.5 * (np.kron(_I4, LdL) + np.kron(LdL.T, _I4)))


class SiliconNoiseModel:
    """
    Combined noise configuration: charge noise on the detuning, Overhauser
    gradient shifts, and T1 / T2* decoherence for both electrons.
    """

    def __init__(
        self,
        t1_us: float = 1000.0,
        t2_star_us: float = 20.0,
        charge_noise_amp: float = 0.03,   # RMS detuning noise (mV)
        overhauser_sigma: float = 0.5,    # MHz
        seed: Optional[int] = None,
        alpha: float = 1.0,
    ):
        self.t1 = float(t1_us)
        self.t2_star = float(t2_star_us)
        self.charge_noise_amp = float(charge_noise_amp)
        self.overhauser_sigma = float(overhauser_sigma)
        self.pink_gen = PinkNoiseGenerator(alpha=alpha, amplitude=charge_noise_amp, seed=seed)
        self.overhauser = OverhauserNoise(sigma_overhauser=overhauser_sigma, seed=seed)

    # ----------------------------------------------------------------- #
    def lindblad_rates_per_ns(self) -> Dict[str, float]:
        """γ₁ = 1/T1 and pure-dephasing γ_φ = 1/T2* − 1/(2T1), converted to 1/ns."""
        gamma_1 = PER_US_TO_PER_NS / self.t1 if self.t1 > 0 else 0.0
        gamma_2 = PER_US_TO_PER_NS / self.t2_star if self.t2_star > 0 else 0.0
        gamma_phi = max(0.0, gamma_2 - 0.5 * gamma_1)
        return {"gamma_1": gamma_1, "gamma_phi": gamma_phi}

    def get_lindblad_jump_operators(self) -> List[Tuple[np.ndarray, float]]:
        """[(L_k, γ_k in 1/ns)] — σ⁻ relaxation and σ_z/2 dephasing on each electron."""
        r = self.lindblad_rates_per_ns()
        # For pure dephasing with L = σ_z/√2 the coherence decays as exp(−γ_φ t).
        z1 = np.kron(_SZ, _I2) / np.sqrt(2.0)
        z2 = np.kron(_I2, _SZ) / np.sqrt(2.0)
        m1 = np.kron(_SM, _I2)
        m2 = np.kron(_I2, _SM)
        return [(z1, r["gamma_phi"]), (z2, r["gamma_phi"]), (m1, r["gamma_1"]), (m2, r["gamma_1"])]

    def apply_noisy_pulses(
        self, j_nominal: np.ndarray, epsilon_0: float, dt_ns: float, delta_bz_nominal: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One noise realisation: J(t)·exp(δε(t)/ε₀) and a quasi-static ΔB_z shift."""
        n_steps = len(j_nominal)
        delta_eps = self.pink_gen.generate_spectral_trace(n_steps, dt_ns)
        j_noisy = np.asarray(j_nominal) * np.exp(delta_eps / max(float(epsilon_0), 1e-4))
        delta_bz_noisy = np.full(n_steps, float(delta_bz_nominal) + self.overhauser.sample_quasistatic_shift())
        return j_noisy, delta_bz_noisy

    # ----------------------------------------------------------------- #
    def _step_superoperator(self, H_mhz: np.ndarray, dt_ns: float) -> np.ndarray:
        Lsup = _superop_hamiltonian(MHZ_NS_TO_RAD * H_mhz)
        for L, rate in self.get_lindblad_jump_operators():
            if rate > 0:
                Lsup = Lsup + _superop_dissipator(L, rate)
        return scipy.linalg.expm(Lsup * dt_ns)

    def channel_superoperator(
        self,
        hamiltonian_func: Callable[..., np.ndarray],
        j_pulse: np.ndarray,
        dt_ns: float,
        delta_bz_pulse: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """16×16 (column-stacking) superoperator of the full noisy gate."""
        S = np.eye(16, dtype=np.complex128)
        cache: Dict[Tuple[float, Optional[float]], np.ndarray] = {}
        for k, j_k in enumerate(j_pulse):
            dB_k = None if delta_bz_pulse is None else float(delta_bz_pulse[k])
            key = (round(float(j_k), 12), None if dB_k is None else round(dB_k, 12))
            if key not in cache:
                cache[key] = self._step_superoperator(hamiltonian_func(float(j_k), dB_k), dt_ns)
            S = cache[key] @ S
        return S

    def evolve_density_matrix_lindblad(
        self,
        rho_0: np.ndarray,
        hamiltonian_func: Callable[..., np.ndarray],
        j_pulse: np.ndarray,
        dt_ns: float,
        delta_bz_pulse: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve dρ/dt = −i·2π·1e-3·[H, ρ] + Σ_k γ_k (L_k ρ L_k† − ½{L_k†L_k, ρ})
        exactly for a piecewise-constant pulse.
        """
        S = self.channel_superoperator(hamiltonian_func, j_pulse, dt_ns, delta_bz_pulse)
        rho_vec = S @ np.asarray(rho_0, dtype=np.complex128).reshape(-1, order="F")
        return rho_vec.reshape(4, 4, order="F")

    def channel_process_fidelity(
        self,
        j_pulse: np.ndarray,
        dt_ns: float,
        target_unitary: np.ndarray,
        hamiltonian_func: Callable[..., np.ndarray],
        delta_bz_pulse: Optional[np.ndarray] = None,
    ) -> float:
        """Process (entanglement) fidelity Tr(S_U† S_E)/d² of the noisy gate against ``target_unitary``."""
        S_E = self.channel_superoperator(hamiltonian_func, j_pulse, dt_ns, delta_bz_pulse)
        U = np.asarray(target_unitary, dtype=np.complex128)
        S_U = np.kron(U.conj(), U)
        return float(np.real(np.trace(S_U.conj().T @ S_E)) / 16.0)


# --------------------------------------------------------------------------- #
# Filter functions
# --------------------------------------------------------------------------- #
def filter_function(control: np.ndarray, dt_ns: float, omega: np.ndarray) -> np.ndarray:
    """
    |F(ω)|² = |Σ_k s_k e^{iω t_k} dt|² for a piecewise-constant control s(t)
    (ω in rad/ns, midpoint time grid).
    """
    t = (np.arange(len(control)) + 0.5) * dt_ns
    phases = np.exp(1j * np.outer(omega, t))            # (n_w, n_t)
    F = phases @ (np.asarray(control, dtype=float) * dt_ns)
    return np.abs(F) ** 2


def filter_function_infidelity(
    j_pulse: np.ndarray,
    dt_ns: float,
    psd_func: Callable[[np.ndarray], np.ndarray],
    sensitivity: float = 1.0,
    n_omega: int = 2000,
) -> float:
    """
    First-order infidelity from multiplicative noise δJ = sensitivity·J·δε:

        1 − F ≈ (sensitivity² / 2π) ∫₀^∞ S(ω) |F_J(ω)|² dω,

    with ``psd_func(ω)`` the one-sided noise PSD (ω in rad/ns).
    """
    T = len(j_pulse) * dt_ns
    omega = np.logspace(np.log10(2 * np.pi / (100.0 * T)), np.log10(np.pi / dt_ns), int(n_omega))
    ff = filter_function(j_pulse, dt_ns, omega)
    integrand = np.asarray(psd_func(omega)) * ff
    return float(sensitivity**2 / (2 * np.pi) * np.trapezoid(integrand, omega))
