"""
Noise models for silicon spin qubits.

1. 1/f^α charge noise on the detuning (spectral synthesis, or a sum of
   Ornstein–Uhlenbeck two-level fluctuators with log-uniform switching rates).
2. Quasi-static Overhauser (nuclear-spin) shifts of the Zeeman gradient.
3. Lindblad master equation with T1 relaxation (thermal: both σ⁻ and σ⁺ with
   detailed balance at the electron temperature) and pure dephasing, solved
   exactly per time slice by exponentiating the 16×16 Liouvillian.
4. Filter functions in the toggle frame of the gate, and the first-order
   infidelity they predict for an arbitrary noise spectrum.

Units: J and ΔB_z in MHz, time steps in ns, T1 / T2* in µs, PSD cut-offs in Hz.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg
import scipy.signal

from .units import MHZ_NS_TO_RAD, PER_US_TO_PER_NS

H_OVER_KB_K_PER_MHZ = 6.62607015e-34 * 1e6 / 1.380649e-23   # h·(1 MHz)/k_B in kelvin (4.8e-5 K)


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
        temperature_mk: float = 0.0,
        larmor_frequency_mhz: float = 18000.0,
    ):
        self.t1 = float(t1_us)
        self.temperature_mk = float(temperature_mk)
        self.larmor_frequency_mhz = float(larmor_frequency_mhz)
        self.t2_star = float(t2_star_us)
        self.charge_noise_amp = float(charge_noise_amp)
        self.overhauser_sigma = float(overhauser_sigma)
        self.pink_gen = PinkNoiseGenerator(alpha=alpha, amplitude=charge_noise_amp, seed=seed)
        self.overhauser = OverhauserNoise(sigma_overhauser=overhauser_sigma, seed=seed)

    # ----------------------------------------------------------------- #
    def excited_state_population(self) -> float:
        """Thermal |↑⟩ population 1/(1 + e^{hf/kT}) at the Larmor frequency."""
        if self.temperature_mk <= 0:
            return 0.0
        x = H_OVER_KB_K_PER_MHZ * self.larmor_frequency_mhz / (1e-3 * self.temperature_mk)
        return float(1.0 / (1.0 + np.exp(x)))

    def lindblad_rates_per_ns(self) -> Dict[str, float]:
        """
        Rates in 1/ns. T1 is the measured relaxation time, 1/T1 = γ↓ + γ↑, split
        by detailed balance γ↑/γ↓ = e^{−hf/kT}; pure dephasing
        γ_φ = 1/T2* − 1/(2T1).
        """
        gamma_1 = PER_US_TO_PER_NS / self.t1 if self.t1 > 0 else 0.0
        gamma_2 = PER_US_TO_PER_NS / self.t2_star if self.t2_star > 0 else 0.0
        gamma_phi = max(0.0, gamma_2 - 0.5 * gamma_1)
        p_up = self.excited_state_population()
        return {"gamma_1": gamma_1, "gamma_down": gamma_1 * (1.0 - p_up), "gamma_up": gamma_1 * p_up, "gamma_phi": gamma_phi}

    def get_lindblad_jump_operators(self) -> List[Tuple[np.ndarray, float]]:
        """
        [(L_k, γ_k in 1/ns)] — σ⁻ relaxation, σ⁺ thermal excitation and σ_z/√2
        dephasing on each electron. |0⟩ = |↑⟩ is the upper Zeeman level
        (H ∋ +B₀/2·Z), so relaxation is |0⟩ → |1⟩.
        """
        r = self.lindblad_rates_per_ns()
        # For pure dephasing with L = σ_z/√2 the coherence decays as exp(−γ_φ t).
        z1 = np.kron(_SZ, _I2) / np.sqrt(2.0)
        z2 = np.kron(_I2, _SZ) / np.sqrt(2.0)
        m1, m2 = np.kron(_SM, _I2), np.kron(_I2, _SM)
        p1, p2 = m1.conj().T, m2.conj().T
        ops = [(z1, r["gamma_phi"]), (z2, r["gamma_phi"]), (m1, r["gamma_down"]), (m2, r["gamma_down"])]
        if r["gamma_up"] > 0:
            ops += [(p1, r["gamma_up"]), (p2, r["gamma_up"])]
        return ops

    def idle_superoperator(self, duration_ns: float) -> np.ndarray:
        """16×16 channel of pure decoherence (no Hamiltonian, rotating frame) for ``duration_ns``."""
        Lsup = np.zeros((16, 16), dtype=np.complex128)
        for L, rate in self.get_lindblad_jump_operators():
            if rate > 0:
                Lsup = Lsup + _superop_dissipator(L, rate)
        return scipy.linalg.expm(Lsup * float(duration_ns))

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
def _pauli_basis_2q() -> np.ndarray:
    """The 15 traceless two-qubit Paulis, normalised so Tr(C_a C_b) = δ_ab."""
    P = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    return np.array([np.kron(P[a], P[b]) / 2.0 for a in range(4) for b in range(4) if (a, b) != (0, 0)], dtype=np.complex128)


def control_matrix(
    hamiltonians_mhz: Sequence[np.ndarray],
    dt_ns: float,
    noise_operators: Sequence[Tuple[np.ndarray, np.ndarray]],
    omega: np.ndarray,
) -> np.ndarray:
    """
    Toggle-frame control matrix R_αβ(ω) for a piecewise-constant gate.

    The noise Hamiltonian is Σ_α b_α(t) a_α(t) N_α with b_α the noise process
    (same units as the Hamiltonian, MHz) and a_α(t) a known per-slice
    sensitivity. In the interaction picture of the ideal gate U₀(t),

        R_αβ(ω) = ∫₀ᵀ dt e^{iωt} a_α(t) Tr[C_β U₀†(t) N_α U₀(t)],

    evaluated exactly inside every slice: with H_k = V diag(Ω) V†,
    U₀†N U₀ has elements Ñ_mn e^{i(Ω_m−Ω_n)τ} and the τ-integral is closed form
    (the same construction as Hangleiter et al., PRR 3, 043047 (2021)).

    ``noise_operators`` is a list of ``(N_α, a_α)`` with ``a_α`` of length n_steps.
    Returns an array of shape (n_noise, n_omega, 15).
    """
    omega = np.asarray(omega, dtype=float)
    n_steps = len(hamiltonians_mhz)
    C = _pauli_basis_2q()
    R = np.zeros((len(noise_operators), len(omega), len(C)), dtype=np.complex128)
    Q = np.eye(4, dtype=np.complex128)                       # U₀ at the start of the slice
    for k in range(n_steps):
        Hk = MHZ_NS_TO_RAD * np.asarray(hamiltonians_mhz[k], dtype=np.complex128)
        Om, V = np.linalg.eigh(0.5 * (Hk + Hk.conj().T))
        W = V.conj().T @ Q                                   # maps the lab frame into slice eigenbasis
        G = np.einsum("ia,bak,jk->bij", W, C, W.conj())            # W C_β W†
        # x_mn(ω) = ω + Ω_m − Ω_n ;  I_mn = ∫₀^dt e^{i x τ} dτ
        x = omega[:, None, None] + (Om[:, None] - Om[None, :])[None]
        small = np.abs(x * dt_ns) < 1e-8
        I = np.where(small, dt_ns + 0.5j * x * dt_ns**2, (np.exp(1j * x * dt_ns) - 1.0) / np.where(small, 1.0, 1j * x))
        phase = np.exp(1j * omega * k * dt_ns)
        for a, (N, amp) in enumerate(noise_operators):
            Nt = V.conj().T @ (MHZ_NS_TO_RAD * float(amp[k]) * np.asarray(N, dtype=np.complex128)) @ V
            # Tr[C_β W† (Ñ∘I) W] = Σ_mn (Ñ∘I)_mn (W C_β W†)_nm
            R[a] += phase[:, None] * np.einsum("wmn,mn,bnm->wb", I, Nt, G, optimize=True)
        Q = V @ np.diag(np.exp(-1j * Om * dt_ns)) @ V.conj().T @ Q
    return R


def gate_filter_functions(
    hamiltonians_mhz: Sequence[np.ndarray],
    dt_ns: float,
    noise_operators: Sequence[Tuple[np.ndarray, np.ndarray]],
    omega: np.ndarray,
) -> np.ndarray:
    """F_α(ω) = Σ_β |R_αβ(ω)|², shape (n_noise, n_omega), units rad²·… (see ``control_matrix``)."""
    R = control_matrix(hamiltonians_mhz, dt_ns, noise_operators, omega)
    return np.sum(np.abs(R) ** 2, axis=-1)


def exchange_gate_noise_operators(j_pulse: np.ndarray, kind: str = "exchange") -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Standard noise channels for the two-electron gate:

    * ``"exchange"`` — multiplicative charge noise δJ(t) = J(t)·ε(t): N = (X₁X₂+Y₁Y₂+Z₁Z₂)/4, a = J(t)
    * ``"zeeman1"`` / ``"zeeman2"`` — frequency noise on one electron: N = Z_i/2, a = 1
    """
    from .hamiltonian import HEISENBERG_EXCHANGE, SIGMA_I, SIGMA_Z

    j = np.asarray(j_pulse, dtype=float)
    if kind == "exchange":
        return [(HEISENBERG_EXCHANGE / 4.0, j)]
    if kind == "zeeman1":
        return [(np.kron(SIGMA_Z, SIGMA_I) / 2.0, np.ones_like(j))]
    if kind == "zeeman2":
        return [(np.kron(SIGMA_I, SIGMA_Z) / 2.0, np.ones_like(j))]
    raise ValueError(f"unknown noise kind {kind!r}")


def infidelity_from_filter_function(omega: np.ndarray, F: np.ndarray, psd_one_sided: np.ndarray, d: int = 4) -> float:
    """
    First-order entanglement infidelity

        1 − F_e ≈ (1 / 2πd) ∫₀^∞ S(ω) F(ω) dω

    for a one-sided PSD S (∫₀^∞ S dω/2π = ⟨b²⟩, b in MHz, ω in rad/ns, S in MHz²·ns).
    Valid while the result is ≪ 1.
    """
    _trapz = getattr(np, "trapezoid", None) or np.trapz   # NumPy < 2.0 compatibility
    return float(_trapz(np.asarray(psd_one_sided) * np.asarray(F), omega) / (2.0 * np.pi * d))


def filter_function(control: np.ndarray, dt_ns: float, omega: np.ndarray) -> np.ndarray:
    """
    Scalar spectrum |Σ_k s_k e^{iω t_k} dt|² of a piecewise-constant control
    (ω in rad/ns, midpoint grid).

    This equals the gate filter function only when the noise operator commutes
    with the control Hamiltonian at all times (e.g. exchange noise with
    ΔB_z = 0). For a real gate use ``gate_filter_functions``, which works in the
    toggle frame of U₀(t).
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
    n_omega: int = 600,
    hamiltonian=None,
    noise: str = "exchange",
) -> float:
    """
    First-order infidelity of an exchange gate under a noise process with
    one-sided PSD ``psd_func(ω)`` (ω in rad/ns), in the toggle frame of the
    ideal evolution under ``hamiltonian`` (default: ΔB_z = 15 MHz model).

    ``noise="exchange"``: δJ(t) = sensitivity·J(t)·ε(t) with ε the noise process.
    ``noise="zeeman1"``/``"zeeman2"``: δf(t) = sensitivity·ε(t) MHz on one electron.
    """
    from .hamiltonian import SiliconSpinHamiltonian

    h = hamiltonian if hamiltonian is not None else SiliconSpinHamiltonian()
    j = np.asarray(j_pulse, dtype=float)
    T = len(j) * dt_ns
    omega = np.concatenate([[0.0], np.logspace(np.log10(2 * np.pi / (100.0 * T)), np.log10(np.pi / dt_ns), int(n_omega))])
    Hs = [h.get_hamiltonian_matrix(float(jk)) for jk in j]
    ops = [(N, sensitivity * a) for N, a in exchange_gate_noise_operators(j, noise)]
    F = gate_filter_functions(Hs, dt_ns, ops, omega)[0]
    return infidelity_from_filter_function(omega, F, psd_func(np.maximum(omega, omega[1])))
