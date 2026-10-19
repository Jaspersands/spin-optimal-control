"""
Error budget of an exchange-pulsed CZ gate.

For a pulse J(t) under a Zeeman difference ΔE_z the CZ infidelity is split into
four contributions, each computed with the machinery already in the package:

(a) coherent: the exact noiseless propagator, after the best virtual-Z
    correction (``pulse_shaping.cz_fidelity_up_to_z``). For a window-shaped
    pulse this is the non-adiabatic transfer between |↑↓⟩ and |↓↑⟩.
(b) Zeeman noise on each qubit frequency, with a 1/f^α spectrum. The
    quasi-static part has the Gaussian width σ = √2 / (2π T2*) that a Ramsey
    experiment measures; it enters through the toggle-frame filter function at
    ω = 0. The same spectrum above the Ramsey band gives a fast part. Where a
    Hahn-echo time is known, α and the amplitude are fitted per qubit to T2*
    and T2 (``fit_zeeman_alpha``); otherwise α = 1.
(c) exchange (charge) noise, δJ(t) = J(t)·η(t), i.e. dJ/dε = J/ε₀ for the
    exponential J(ε). η has a quasi-static width σ_J and a 1/f^α spectrum
    above the Ramsey band, integrated with the toggle-frame filter function.
(d) Markovian relaxation and dephasing: the Lindblad channel with per-qubit
    T1 (detailed balance at the electron temperature) and white dephasing.
    White dephasing is only the part of a measured echo decay that the fitted
    1/f^α spectrum cannot explain (``white_dephasing_t2``); it is nonzero only
    when the fit hits the lower end of the allowed α range.

Every contribution is reported as an average gate infidelity
1 − F_avg = (d/(d+1))·(1 − F_e), d = 4, the quantity randomized benchmarking and
gate set tomography quote. To first order the four add. ``combined_simulation``
checks that: it propagates the Lindblad equation with all noise acting at once
(Gauss–Hermite quadrature over the quasi-static offsets, Monte-Carlo over the
fast 1/f traces) and returns the infidelity of the averaged channel.

Conventions: MHz and ns; ``delta_ez_mhz`` is the qubit frequency difference
f₁ − f₂. The package Hamiltonian is written with Pauli matrices,
(ΔB_z/2)(Z₁ − Z₂), so its ``delta_bz`` is ΔE_z / 2.

Candidate mechanisms for the gap between this budget and measured errors are
separate terms, each switched on through ``GapMechanisms`` (see ``GAP_TERMS``):
the exchange-gate voltage moving the qubit frequencies, noise slower than the
Ramsey average, pulse distortion from a finite bandwidth, residual exchange
outside the pulse, uncalibrated frequency and exchange offsets, and a measured
coherent residual. With no mechanisms the budget is the four parts above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg

from .hamiltonian import HEISENBERG_EXCHANGE, SIGMA_I, SIGMA_Z, Z_DIFF, ExchangeDynamics, SiliconSpinHamiltonian
from .noise import H_OVER_KB_K_PER_MHZ, _superop_dissipator, control_matrix
from .pulse_shaping import CZ_AREA_MHZ_NS, cz_fidelity_up_to_z, window
from .units import MHZ_NS_TO_RAD, PER_US_TO_PER_NS

D = 4
E_TO_AVG = D / (D + 1.0)          # 1 − F_avg = (d/(d+1))·(1 − F_e)

_Z1 = np.kron(SIGMA_Z, SIGMA_I)
_Z2 = np.kron(SIGMA_I, SIGMA_Z)
_SM = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)   # lowers |0⟩=|↑⟩ to |1⟩=|↓⟩
_CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)

CONTRIBUTIONS = ("coherent", "zeeman", "exchange", "markovian")
GAP_TERMS = ("gate_zeeman", "slow_zeeman", "slow_exchange", "distortion", "residual_exchange",
             "frequency_offset", "exchange_offset", "calibration_residual")
LABELS = {
    "coherent": "coherent (non-adiabatic)",
    "zeeman": "Zeeman noise (T2*)",
    "exchange": "exchange (charge) noise",
    "markovian": "Markovian T1/T2",
    "gate_zeeman": "qubit frequencies moved by the exchange-gate voltage",
    "slow_zeeman": "Zeeman noise slower than the Ramsey average",
    "slow_exchange": "exchange noise slower than the Ramsey average",
    "distortion": "pulse distortion from a finite bandwidth",
    "residual_exchange": "residual exchange outside the pulse",
    "frequency_offset": "uncalibrated qubit-frequency offset",
    "exchange_offset": "uncalibrated exchange offset",
    "calibration_residual": "coherent calibration residual (measured)",
}


def zeeman_sigma_from_t2_star(t2_star_us: float) -> float:
    """Quasi-static Gaussian width (MHz) of a qubit frequency with Ramsey time T2* (µs): √2/(2π T2*)."""
    return float(np.sqrt(2.0) / (2.0 * np.pi * float(t2_star_us)))


def _band_integral(f_lo: float, f_hi: float, alpha: float) -> float:
    """∫ f^-α df from f_lo to f_hi (Hz)."""
    if f_hi <= f_lo:
        return 0.0
    if abs(alpha - 1.0) < 1e-12:
        return float(np.log(f_hi / f_lo))
    return float((f_hi ** (1.0 - alpha) - f_lo ** (1.0 - alpha)) / (1.0 - alpha))


_ALPHA_RANGE = (0.5, 2.5)          # exponents allowed in the T2*/T2 fit


def _echo_integral(alpha: float) -> float:
    """I(α) = ∫₀^∞ x^(−α−2) sin⁴x dx (finite for −1 < α < 3; I(1) = ln 2)."""
    import scipy.integrate

    a = float(alpha)
    X = 400.0 * np.pi
    f = lambda x: x ** (-a - 2.0) * np.sin(x) ** 4 if x > 0 else 0.0
    edges = np.linspace(0.0, X, 401)
    head = sum(scipy.integrate.quad(f, lo, hi, limit=100)[0] for lo, hi in zip(edges[:-1], edges[1:]))
    tail = 0.375 * X ** (-a - 1.0) / (a + 1.0)                 # ⟨sin⁴⟩ = 3/8 beyond X
    return float(head + tail)


_ECHO_I_CACHE: Dict[float, float] = {}


def echo_integral(alpha: float) -> float:
    key = round(float(alpha), 6)
    if key not in _ECHO_I_CACHE:
        _ECHO_I_CACHE[key] = float(np.log(2.0)) if key == 1.0 else _echo_integral(key)
    return _ECHO_I_CACHE[key]


def psd_amplitude(sigma_mhz: float, f_low_hz: float, f_ramsey_hz: float, alpha: float = 1.0) -> float:
    """A in S(f) = A f^-α (MHz²·Hz^(α−1), f in Hz) such that ∫ S df over [f_low, f_ramsey] = σ²."""
    band = _band_integral(f_low_hz, f_ramsey_hz, alpha)
    return float(sigma_mhz**2 / band) if band > 0 else 0.0


def echo_time_from_1f(sigma_mhz: float, f_low_hz: float, f_ramsey_hz: float, alpha: float = 1.0) -> float:
    """
    Hahn-echo 1/e time (µs) of a qubit frequency with a 1/f^α spectrum whose
    quasi-static width over [f_low, f_ramsey] is ``sigma_mhz``.

    With S(f) = A f^-α the echo phase variance is
    ⟨φ²⟩ = 16·A·(πt/2)^(α+1)·I(α), I(α) = ∫₀^∞ x^(−α−2) sin⁴x dx, so the echo
    decays as exp(−(t/T2)^(α+1)) with T2 = (2/π)(8·A·I(α))^(−1/(α+1)). For
    α = 1, I = ln 2 and T2 = 1/(π √(2 A ln 2)). The low cut-off is negligible
    for the echo filter.
    """
    if sigma_mhz <= 0:
        return float("inf")
    A_hz2 = 1e12 * psd_amplitude(sigma_mhz, f_low_hz, f_ramsey_hz, alpha)      # Hz²·Hz^(α−1)
    t_s = (2.0 / np.pi) * (8.0 * A_hz2 * echo_integral(alpha)) ** (-1.0 / (alpha + 1.0))
    return float(1e6 * t_s)


def white_dephasing_t2(t2_echo_us: float, sigma_mhz: float, f_low_hz: float, f_ramsey_hz: float,
                       t1_us: float = np.inf, alpha: float = 1.0) -> float:
    """
    Coherence time (µs) for the Markovian (white) dephasing in the Lindblad term.

    Only the part of the measured echo decay that the 1/f^α spectrum fixed by
    T2* does not already explain is treated as white noise. At t = T2_echo the
    measured decay exponent is 1; the 1/f^α noise contributes
    (T2_echo/T2_1/f)^(α+1), and the rest, if positive, is a pure-dephasing rate
    γ_φ = (1 − (T2_echo/T2_1/f)^(α+1))/T2_echo. Returns T2 with
    1/T2 = γ_φ + 1/(2T1), or inf when the 1/f^α noise explains the whole echo
    decay and T1 is infinite. Reading the full echo decay as white noise would
    double count the low-frequency noise that T2* already contains.
    """
    if not np.isfinite(t2_echo_us) or t2_echo_us <= 0:
        gphi = 0.0
    else:
        t2f = echo_time_from_1f(sigma_mhz, f_low_hz, f_ramsey_hz, alpha)
        resid = 1.0 - (t2_echo_us / t2f) ** (alpha + 1.0)
        gphi = resid / t2_echo_us if resid > 1e-6 else 0.0           # below 1e-6: fit tolerance, not white noise
    g = gphi + (0.5 / t1_us if np.isfinite(t1_us) and t1_us > 0 else 0.0)
    return float(1.0 / g) if g > 0 else float("inf")


def fit_zeeman_alpha(sigma_mhz: float, t2_echo_us: Optional[float], f_low_hz: float, f_ramsey_hz: float,
                     alpha_range: Tuple[float, float] = _ALPHA_RANGE, default_alpha: float = 1.0) -> Dict[str, float]:
    """
    Exponent α of S(f) = A f^-α for one qubit frequency from two numbers: the
    quasi-static width σ (from T2*) and the Hahn-echo time.

    σ fixes A for each α (``psd_amplitude``); the echo time rises with α, so
    the α that reproduces the measured echo is found by bisection. If even the
    lowest allowed α predicts a longer echo than measured, α is clipped there
    and the shortfall becomes white dephasing (``white_dephasing_t2``); if the
    highest allowed α predicts a shorter one, α is clipped at the top. Without
    an echo time α = ``default_alpha``. T1 is ignored (it is milliseconds
    against echo times of tens of µs in every device here).

    Returns ``alpha``, ``status`` ('fitted', 'clipped at lowest', 'clipped at
    highest', 'no echo'), ``echo_us_at_alpha1`` and ``echo_us_model``.
    """
    import scipy.optimize

    out = {"alpha": float(default_alpha), "status": "no echo",
           "echo_us_at_alpha1": echo_time_from_1f(sigma_mhz, f_low_hz, f_ramsey_hz, 1.0)}
    if t2_echo_us is None or not np.isfinite(t2_echo_us) or t2_echo_us <= 0 or sigma_mhz <= 0:
        out["echo_us_model"] = echo_time_from_1f(sigma_mhz, f_low_hz, f_ramsey_hz, out["alpha"])
        return out
    g = lambda a: np.log(echo_time_from_1f(sigma_mhz, f_low_hz, f_ramsey_hz, a) / t2_echo_us)
    lo, hi = alpha_range
    if g(lo) >= 0:
        out.update(alpha=float(lo), status="clipped at lowest")
    elif g(hi) <= 0:
        out.update(alpha=float(hi), status="clipped at highest")
    else:
        out.update(alpha=float(scipy.optimize.brentq(g, lo, hi, xtol=1e-6)), status="fitted")
    out["echo_us_model"] = echo_time_from_1f(sigma_mhz, f_low_hz, f_ramsey_hz, out["alpha"])
    return out


def _pair(x) -> Tuple[float, float]:
    if np.ndim(x) == 0:
        return float(x), float(x)
    a, b = x
    return float(a), float(b)


@dataclass
class CZNoiseParameters:
    """
    Noise environment of a two-qubit exchange gate.

    Parameters
    ----------
    delta_ez_mhz : qubit frequency difference f₁ − f₂ (MHz)
    zeeman_sigma_mhz : (σ₁, σ₂) quasi-static Gaussian widths of the two qubit
        frequencies (MHz), e.g. ``zeeman_sigma_from_t2_star(T2*)``
    exchange_sigma_rel : quasi-static width of δJ/J (dimensionless)
    t1_us : (T1₁, T1₂) in µs; ``np.inf`` switches relaxation off
    t2_markov_us : (T2₁, T2₂) in µs used for white dephasing in the Lindblad
        term, with 1/T_φ = 1/T2 − 1/(2T1); ``np.inf`` switches it off
    zeeman_alpha : (α₁, α₂) exponents of the 1/f^α spectra of the two qubit
        frequencies (``fit_zeeman_alpha``)
    exchange_alpha : exponent of the 1/f^α spectrum of δJ/J
    f_low_hz : lowest noise frequency, 1/(averaging time) of the measurements
        that gave the quasi-static widths
    f_ramsey_hz : upper edge of the band contained in the quasi-static widths
        (≈ 1/T2*); noise above it is integrated with filter functions
    temperature_mk : electron temperature for detailed balance in T1
    larmor_mhz : (f₁, f₂) qubit frequencies for detailed balance
    """

    delta_ez_mhz: float
    zeeman_sigma_mhz: Tuple[float, float] = (0.0, 0.0)
    exchange_sigma_rel: float = 0.0
    t1_us: Tuple[float, float] = (np.inf, np.inf)
    t2_markov_us: Tuple[float, float] = (np.inf, np.inf)
    zeeman_alpha: Tuple[float, float] = (1.0, 1.0)
    exchange_alpha: float = 1.0
    f_low_hz: float = 1e-2
    f_ramsey_hz: float = 1e5
    temperature_mk: float = 0.0
    larmor_mhz: Tuple[float, float] = (18000.0, 18000.0)

    def __post_init__(self):
        self.zeeman_sigma_mhz = _pair(self.zeeman_sigma_mhz)
        self.t1_us = _pair(self.t1_us)
        self.t2_markov_us = _pair(self.t2_markov_us)
        self.larmor_mhz = _pair(self.larmor_mhz)
        self.zeeman_alpha = _pair(self.zeeman_alpha)

    # ----------------------------------------------------------------- #
    @property
    def delta_bz_mhz(self) -> float:
        """The package's ΔB_z (Pauli convention): half the qubit frequency difference."""
        return 0.5 * float(self.delta_ez_mhz)

    def hamiltonian(self) -> SiliconSpinHamiltonian:
        return SiliconSpinHamiltonian(delta_bz=self.delta_bz_mhz)

    def noiseless(self) -> "CZNoiseParameters":
        """Same device with every noise source switched off."""
        return CZNoiseParameters(delta_ez_mhz=self.delta_ez_mhz, zeeman_alpha=self.zeeman_alpha,
                                 exchange_alpha=self.exchange_alpha, f_low_hz=self.f_low_hz,
                                 f_ramsey_hz=self.f_ramsey_hz)

    def replace(self, **kw) -> "CZNoiseParameters":
        d = dict(self.__dict__)
        d.update(kw)
        return CZNoiseParameters(**d)

    @property
    def alphas(self) -> np.ndarray:
        """Spectral exponents of [Zeeman 1, Zeeman 2, δJ/J]."""
        return np.array([self.zeeman_alpha[0], self.zeeman_alpha[1], self.exchange_alpha], dtype=float)

    def psd_amplitudes(self) -> np.ndarray:
        """
        One-sided 1/f^α amplitudes A (S(f) = A f^-α, f in Hz) for
        [Zeeman 1 (MHz²·Hz^(α−1)), Zeeman 2, δJ/J], fixed so that the
        quasi-static widths equal ∫ S df over [f_low, f_ramsey].
        """
        s = np.array([self.zeeman_sigma_mhz[0], self.zeeman_sigma_mhz[1], self.exchange_sigma_rel])
        return np.array([psd_amplitude(x, self.f_low_hz, self.f_ramsey_hz, a) for x, a in zip(s, self.alphas)])

    # ----------------------------------------------------------------- #
    def dissipator(self) -> np.ndarray:
        """16×16 Lindblad dissipator (1/ns): relaxation, thermal excitation and white dephasing per qubit."""
        Lsup = np.zeros((16, 16), dtype=np.complex128)
        for q in range(2):
            t1, t2 = self.t1_us[q], self.t2_markov_us[q]
            g1 = PER_US_TO_PER_NS / t1 if np.isfinite(t1) and t1 > 0 else 0.0
            g2 = PER_US_TO_PER_NS / t2 if np.isfinite(t2) and t2 > 0 else 0.0
            gphi = max(0.0, g2 - 0.5 * g1)
            p_up = 0.0
            if self.temperature_mk > 0:
                x = H_OVER_KB_K_PER_MHZ * self.larmor_mhz[q] / (1e-3 * self.temperature_mk)
                p_up = float(1.0 / (1.0 + np.exp(x)))
            sm = np.kron(_SM, SIGMA_I) if q == 0 else np.kron(SIGMA_I, _SM)
            z = (_Z1 if q == 0 else _Z2) / np.sqrt(2.0)
            for L, rate in ((sm, g1 * (1 - p_up)), (sm.conj().T, g1 * p_up), (z, gphi)):
                if rate > 0:
                    Lsup = Lsup + _superop_dissipator(L, rate)
        return Lsup


# --------------------------------------------------------------------------- #
# Pulses
# --------------------------------------------------------------------------- #
def default_n_steps(duration_ns: float, delta_ez_mhz: float, j_max_mhz: float, min_steps: int = 120,
                    steps_per_period: float = 12.0) -> int:
    """Slices so the fastest eigenfrequency (≈ √(ΔE_z² + J²) + J) is sampled ``steps_per_period`` times."""
    f_max = np.hypot(delta_ez_mhz, j_max_mhz) + j_max_mhz          # MHz
    return int(max(min_steps, np.ceil(duration_ns * f_max * 1e-3 * steps_per_period)))


def shaped_cz_pulse(shape: str, duration_ns: float, delta_ez_mhz: float, n_steps: Optional[int] = None,
                    tukey_alpha: float = 0.5) -> Tuple[np.ndarray, float]:
    """Window of area 500 MHz·ns (conditional phase π) on a grid fine enough for ΔE_z. Returns (J, dt)."""
    w_coarse = window(shape, 200, tukey_alpha)
    j_peak = CZ_AREA_MHZ_NS / (np.mean(w_coarse) * duration_ns)
    n = int(n_steps) if n_steps is not None else default_n_steps(duration_ns, delta_ez_mhz, j_peak)
    w = window(shape, n, tukey_alpha)
    dt = duration_ns / n
    return CZ_AREA_MHZ_NS / (np.sum(w) * dt) * w, dt


def grape_cz_pulse(delta_ez_mhz: float, duration_ns: float, n_steps: Optional[int] = None, n_harmonics: int = 8,
                   j_max_mhz: Optional[float] = None, max_iter: int = 300, seed: int = 0) -> Tuple[np.ndarray, float]:
    """CZ pulse from ``GRAPEOptimizer`` with virtual-Z co-optimisation (coherent error only). Returns (J, dt)."""
    from .grape import GRAPEOptimizer

    j_cos = 2.0 * CZ_AREA_MHZ_NS / duration_ns
    j_max = float(j_max_mhz) if j_max_mhz is not None else 2.0 * j_cos
    n = int(n_steps) if n_steps is not None else default_n_steps(duration_ns, delta_ez_mhz, j_max, min_steps=80, steps_per_period=8.0)
    h = SiliconSpinHamiltonian(delta_bz=0.5 * delta_ez_mhz)
    opt = GRAPEOptimizer(h, t_gate_ns=duration_ns, n_steps=n, n_harmonics=n_harmonics, j_max=j_max,
                         local_z_free=True, slew_penalty=0.0, target_infidelity=1e-9)
    # seed: half-sine of area 500 MHz·ns in the first harmonic
    p0 = np.zeros(opt.num_params())
    p0[0] = min(0.9 * j_max, CZ_AREA_MHZ_NS * np.pi / (2.0 * duration_ns))
    res = opt.optimize_pulse(ExchangeDynamics.target_gate_cz(), initial_params=p0, max_iter=max_iter, seed=seed)
    return np.asarray(res.j_pulse, dtype=float), float(res.dt)


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
def z_corrected_cz_target(U0: np.ndarray) -> np.ndarray:
    """T = D†·CZ, where D is the virtual-Z (and global phase) correction that brings U0 closest to CZ."""
    d = np.diag(U0)
    a = np.angle(d[0])
    corr = np.array([1.0, np.exp(-1j * (np.angle(d[1]) - a)), np.exp(-1j * (np.angle(d[2]) - a)),
                     np.exp(-1j * (np.angle(d[1]) + np.angle(d[2]) - 2 * a))]) * np.exp(-1j * a)
    return np.diag(corr).conj().T @ _CZ


@dataclass
class CZErrorBudget:
    """Average-gate-infidelity contributions of one CZ pulse (see module docstring)."""

    label: str
    duration_ns: float
    j_max_mhz: float
    delta_ez_mhz: float
    coherent: float
    zeeman_quasistatic: float
    zeeman_fast: float
    exchange_quasistatic: float
    exchange_fast: float
    markovian: float
    dt_ns: float = 0.0
    j_pulse: np.ndarray = field(default=None, repr=False)
    gap: Dict[str, float] = field(default_factory=dict)     # the GapMechanisms terms that were switched on

    @property
    def zeeman(self) -> float:
        return self.zeeman_quasistatic + self.zeeman_fast

    @property
    def exchange(self) -> float:
        return self.exchange_quasistatic + self.exchange_fast

    @property
    def base_total(self) -> float:
        """The four-part budget without any gap terms."""
        return float(self.coherent + self.zeeman + self.exchange + self.markovian)

    @property
    def contributions(self) -> Dict[str, float]:
        c = {"coherent": self.coherent, "zeeman": self.zeeman, "exchange": self.exchange, "markovian": self.markovian}
        c.update(self.gap)
        return c

    @property
    def total(self) -> float:
        return float(sum(self.contributions.values()))

    @property
    def fidelity(self) -> float:
        return 1.0 - self.total

    @property
    def dominant(self) -> str:
        c = self.contributions
        return max(c, key=c.get)

    def as_dict(self) -> Dict[str, float]:
        out = {"label": self.label, "duration_ns": self.duration_ns, "j_max_mhz": self.j_max_mhz,
               "delta_ez_mhz": self.delta_ez_mhz, "zeeman_quasistatic": self.zeeman_quasistatic,
               "zeeman_fast": self.zeeman_fast, "exchange_quasistatic": self.exchange_quasistatic,
               "exchange_fast": self.exchange_fast}
        out.update(self.contributions)
        out.update({"total": self.total, "fidelity": self.fidelity, "dominant": self.dominant})
        return out


def _hamiltonians(h: SiliconSpinHamiltonian, j_pulse: np.ndarray) -> List[np.ndarray]:
    return [h.get_hamiltonian_matrix(float(j)) for j in j_pulse]


# --------------------------------------------------------------------------- #
# Candidate mechanisms for the gap
# --------------------------------------------------------------------------- #
@dataclass
class ExchangeVoltageMap:
    """J(v) = J_ref·e^{κv} on the exchange control line, v measured from the point where J = J_ref (V)."""

    kappa_per_v: float
    j_ref_mhz: float

    def voltage(self, j_mhz) -> np.ndarray:
        j = np.maximum(np.asarray(j_mhz, dtype=float), self.j_ref_mhz)
        return np.log(j / self.j_ref_mhz) / self.kappa_per_v

    def exchange(self, v) -> np.ndarray:
        return self.j_ref_mhz * np.exp(self.kappa_per_v * np.asarray(v, dtype=float))


@dataclass
class GateZeemanCoupling:
    """
    Qubit frequencies that follow the exchange-gate voltage, f_j(v) = f_j(0) + β_j v^γ
    (Xue 2022, Methods, with J(v) = J_res e^{2αv}, so κ = 2α).

    A voltage fluctuation δv changes J by δJ/J = κ δv and qubit j by β_j γ v^(γ−1) δv,
    so the exchange-noise source η = δJ/J also shifts qubit j by k_j(t)·η with
    k_j = β_j γ v(t)^(γ−1) / κ (MHz). The mean shift β_j v^γ during the pulse is a
    calibrated single-qubit phase and is not modelled.
    """

    beta_mhz: Tuple[float, float]          # MHz V^-γ
    gamma: float
    voltage_map: ExchangeVoltageMap

    @classmethod
    def linear(cls, df_dlnj_mhz: Tuple[float, float], j_floor_mhz: float = 1e-6) -> "GateZeemanCoupling":
        """
        A frequency shift linear in the gate voltage (γ = 1), given as df_j/d(ln J) in MHz: the shift of
        qubit j when J changes by a factor e. It needs no lever arm, since κ cancels.
        """
        return cls(beta_mhz=_pair(df_dlnj_mhz), gamma=1.0, voltage_map=ExchangeVoltageMap(1.0, float(j_floor_mhz)))

    def sensitivity_mhz(self, j_mhz) -> np.ndarray:
        """(2, n): frequency shift of each qubit per unit δJ/J along the pulse."""
        v = self.voltage_map.voltage(j_mhz)
        g = np.zeros_like(v)
        on = v > 0
        g[on] = self.gamma * v[on] ** (self.gamma - 1.0)
        return np.array([float(b) * g / self.voltage_map.kappa_per_v for b in self.beta_mhz])


@dataclass
class GapMechanisms:
    """
    Error terms outside the four-part budget. Each is off while its parameter is None or zero.

    gate_zeeman : ``GateZeemanCoupling``; the exchange noise then also moves the qubit frequencies
        (term ``gate_zeeman``, quasi-static and 1/f, correlated with the exchange noise).
    zeeman_f_run_hz, exchange_f_run_hz : lowest noise frequency a benchmark averages over,
        1/(run time), or 1/(recalibration interval) for a recalibrated quantity. The fitted
        1/f^α spectra are extended from ``f_low_hz`` down to it; the added variance is
        quasi-static within a gate (terms ``slow_zeeman``, ``slow_exchange``).
    bandwidth_mhz : −3 dB bandwidth of a single-pole low-pass on the exchange control. The pulse
        is filtered (the voltage when ``voltage_map`` is given, otherwise J itself), recalibrated
        to the CZ area, and the term ``distortion`` is the change of the whole budget.
    j_off_mhz, idle_ns : residual exchange, and the time outside the pulse that the benchmark
        attributes to the CZ (term ``residual_exchange``). No paper here reports such a time.
    frequency_offset_mhz : rms offsets (δf₁, δf₂) of the qubit frequencies during the gate that
        no calibration removes: drift between recalibrations, shifts after microwave bursts or
        exchange pulses (term ``frequency_offset``).
    exchange_offset_rel : the same for δJ/J, e.g. barrier drift or an exchange-level calibration
        error (term ``exchange_offset``). A static miscalibration of size x and a slow Gaussian
        offset of rms x give the same average infidelity to leading order.
    calibration_residual : a measured coherent residual, average infidelity, added as given
        (term ``calibration_residual``).

    With ``gate_zeeman`` on, the slow exchange noise and the exchange offset move the qubit
    frequencies too, as the fast exchange noise does.
    """

    gate_zeeman: Optional[GateZeemanCoupling] = None
    zeeman_f_run_hz: Optional[float] = None
    exchange_f_run_hz: Optional[float] = None
    bandwidth_mhz: Optional[float] = None
    voltage_map: Optional[ExchangeVoltageMap] = None
    j_off_mhz: float = 0.0
    idle_ns: float = 0.0
    frequency_offset_mhz: Tuple[float, float] = (0.0, 0.0)
    exchange_offset_rel: float = 0.0
    calibration_residual: float = 0.0

    def __post_init__(self):
        self.frequency_offset_mhz = _pair(self.frequency_offset_mhz)

    def active(self) -> Tuple[str, ...]:
        on = {
            "gate_zeeman": self.gate_zeeman is not None,
            "slow_zeeman": self.zeeman_f_run_hz is not None,
            "slow_exchange": self.exchange_f_run_hz is not None,
            "distortion": self.bandwidth_mhz is not None,
            "residual_exchange": self.j_off_mhz > 0 and self.idle_ns > 0,
            "frequency_offset": any(x != 0 for x in self.frequency_offset_mhz),
            "exchange_offset": self.exchange_offset_rel != 0,
            "calibration_residual": self.calibration_residual != 0,
        }
        return tuple(k for k in GAP_TERMS if on[k])

    def only(self, *names: str) -> "GapMechanisms":
        """Copy with every term not in ``names`` switched off."""
        unknown = set(names) - set(GAP_TERMS)
        if unknown:
            raise ValueError(f"unknown gap terms {sorted(unknown)}")
        keep = set(names)
        return GapMechanisms(
            gate_zeeman=self.gate_zeeman if "gate_zeeman" in keep else None,
            zeeman_f_run_hz=self.zeeman_f_run_hz if "slow_zeeman" in keep else None,
            exchange_f_run_hz=self.exchange_f_run_hz if "slow_exchange" in keep else None,
            bandwidth_mhz=self.bandwidth_mhz if "distortion" in keep else None,
            voltage_map=self.voltage_map,
            j_off_mhz=self.j_off_mhz if "residual_exchange" in keep else 0.0,
            idle_ns=self.idle_ns if "residual_exchange" in keep else 0.0,
            frequency_offset_mhz=self.frequency_offset_mhz if "frequency_offset" in keep else (0.0, 0.0),
            exchange_offset_rel=self.exchange_offset_rel if "exchange_offset" in keep else 0.0,
            calibration_residual=self.calibration_residual if "calibration_residual" in keep else 0.0,
        )


def residual_exchange_error(delta_ez_mhz: float, j_off_mhz: float, duration_ns: float) -> float:
    """
    Average infidelity of an idle slot under a constant residual exchange, after the best
    virtual-Z correction. For J ≪ ΔE_z it is a conditional phase φ = 2π·10⁻³·J·t, and
    1 − F = (4/5)·sin²(φ/4).
    """
    if j_off_mhz <= 0 or duration_ns <= 0:
        return 0.0
    U = ExchangeDynamics(SiliconSpinHamiltonian(delta_bz=0.5 * float(delta_ez_mhz))).propagate_unitary(
        np.array([float(j_off_mhz)]), float(duration_ns))
    # Local Z phases cannot remove φ = arg(d₀d₃/d₁d₂); the best correction leaves ±φ/4 on each entry.
    # (pulse_shaping.cz_fidelity_up_to_z puts all of it on d₃, which is exact only when φ is calibrated.)
    d = np.diag(U)
    phi = float(np.angle(d[0] * d[3] / (d[1] * d[2])))
    a = np.abs(d)
    tr = (a[0] + a[3]) * np.exp(0.25j * phi) + (a[1] + a[2]) * np.exp(-0.25j * phi)
    return float(1.0 - (abs(tr) ** 2 + 4.0) / 20.0)


def distorted_pulse(j_pulse: np.ndarray, dt_ns: float, bandwidth_mhz: float,
                    voltage_map: Optional[ExchangeVoltageMap] = None, tail_tau: float = 5.0,
                    max_dt_tau: float = 0.5) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    A pulse through a single-pole low-pass with −3 dB bandwidth ``bandwidth_mhz`` (τ = 1/2πf).

    The slices are split until dt ≤ ``max_dt_tau``·τ (exact for a piecewise-constant pulse),
    and ``tail_tau``·τ is appended for the decay after the pulse. With a ``voltage_map`` the
    voltage v(J) is filtered and mapped back to J, otherwise J is filtered directly. Both the
    filtered and the reference pulse (same grid and floor, not filtered) are rescaled to the CZ
    area, as the experiments recalibrate the exchange amplitude. Returns (filtered, reference, dt).
    """
    import scipy.signal

    tau = 1e3 / (2.0 * np.pi * float(bandwidth_mhz))                    # ns
    m = int(max(1, np.ceil(dt_ns / (max_dt_tau * tau))))
    dt = dt_ns / m
    j = np.repeat(np.asarray(j_pulse, dtype=float), m)
    n_tail = int(np.ceil(tail_tau * tau / dt))
    x = voltage_map.voltage(j) if voltage_map is not None else j
    x = np.concatenate([x, np.zeros(n_tail)])
    a = np.exp(-dt / tau)
    y_end = scipy.signal.lfilter([1.0 - a], [1.0, -a], x)               # output at the end of each slice
    y_start = np.concatenate([[0.0], y_end[:-1]])
    y = x + (y_start - x) * (tau / dt) * (1.0 - a)                       # slice average of the response
    if voltage_map is not None:
        filt, ref = voltage_map.exchange(y), voltage_map.exchange(x)
    else:
        filt, ref = y, x
    filt = filt * CZ_AREA_MHZ_NS / (np.sum(filt) * dt)
    ref = ref * CZ_AREA_MHZ_NS / (np.sum(ref) * dt)
    return filt, ref, dt


def _omega_grid(noise: CZNoiseParameters, dt_ns: float, n_omega: int) -> np.ndarray:
    w_r = 2.0 * np.pi * noise.f_ramsey_hz * 1e-9                     # rad/ns
    w_max = np.pi / dt_ns
    return np.concatenate([[0.0], np.logspace(np.log10(w_r), np.log10(w_max), int(n_omega))])


def _fast_part(F: np.ndarray, A: np.ndarray, alphas: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """∫ S(ω) F(ω) dω / (2π d) above the Ramsey band, per noise source (rows of F)."""
    w = omega[1:]
    f_hz = w * 1e9 / (2.0 * np.pi)
    S = 1e9 * A[:, None] * f_hz[None, :] ** (-alphas[:, None])       # one-sided, per (rad/ns)
    _trapz = getattr(np, "trapezoid", None) or np.trapz               # NumPy < 2.0
    return _trapz(S * F[:, 1:] * w[None, :], np.log(w), axis=1) / (2.0 * np.pi * D)


def cz_error_budget_for_pulse(noise: CZNoiseParameters, j_pulse: np.ndarray, dt_ns: float, label: str = "pulse",
                              n_omega: int = 240, mechanisms: Optional[GapMechanisms] = None) -> CZErrorBudget:
    """
    Error budget of an arbitrary exchange pulse J(t) (MHz, slices of ``dt_ns``) calibrated to a CZ.
    ``mechanisms`` switches on gap terms (``GapMechanisms``); they are reported in ``gap`` and
    included in ``total``.
    """
    j = np.asarray(j_pulse, dtype=float)
    T = len(j) * dt_ns
    h = noise.hamiltonian()
    dyn = ExchangeDynamics(h)
    Hs = _hamiltonians(h, j)
    U0 = dyn.propagate_unitary(j, dt_ns)

    # (a) coherent, after the closed-form virtual-Z correction
    coherent = 1.0 - cz_fidelity_up_to_z(U0)

    # (b), (c) filter functions for Z₁/2, Z₂/2 (b in MHz) and exchange N = σ·σ/4 with a = J(t) (b = δJ/J)
    ops = [(_Z1 / 2.0, np.ones_like(j)), (_Z2 / 2.0, np.ones_like(j)), (HEISENBERG_EXCHANGE / 4.0, j)]
    coupling = mechanisms.gate_zeeman if mechanisms is not None else None
    if coupling is not None:                                          # the same δJ/J source, on Z₁/2 and Z₂/2
        k = coupling.sensitivity_mhz(j)
        ops += [(_Z1 / 2.0, k[0]), (_Z2 / 2.0, k[1])]
    omega = _omega_grid(noise, dt_ns, n_omega)
    R = control_matrix(Hs, dt_ns, ops, omega)                         # (n_ops, n_omega + 1, 15)
    F = np.sum(np.abs(R[:3]) ** 2, axis=-1)                           # (3, n_omega + 1)
    sig = np.array([noise.zeeman_sigma_mhz[0], noise.zeeman_sigma_mhz[1], noise.exchange_sigma_rel])
    qs = sig**2 * F[:, 0] / D                                         # entanglement infidelity
    A = noise.psd_amplitudes()
    fast = _fast_part(F, A, noise.alphas, omega)

    # (d) Lindblad with the noiseless Hamiltonian, against the noiseless propagator
    Dsup = noise.dissipator()
    if np.any(Dsup):
        S_E = _channel(Hs, dt_ns, Dsup)
        S_U = np.kron(U0.conj(), U0)
        markov_e = 1.0 - float(np.real(np.trace(S_U.conj().T @ S_E)) / 16.0)
    else:
        markov_e = 0.0

    gap: Dict[str, float] = {}
    if mechanisms is not None and mechanisms.active():
        F_ex = F[2]
        if coupling is not None:
            F_ex = np.sum(np.abs(R[2] + R[3] + R[4]) ** 2, axis=-1)     # exchange noise with the frequency coupling
        gap = _gap_terms(noise, mechanisms, j, dt_ns, F, F_ex, A, omega, n_omega)

    return CZErrorBudget(
        label=label, duration_ns=float(T), j_max_mhz=float(j.max()), delta_ez_mhz=float(noise.delta_ez_mhz),
        coherent=float(coherent),
        zeeman_quasistatic=float(E_TO_AVG * (qs[0] + qs[1])), zeeman_fast=float(E_TO_AVG * (fast[0] + fast[1])),
        exchange_quasistatic=float(E_TO_AVG * qs[2]), exchange_fast=float(E_TO_AVG * fast[2]),
        markovian=float(E_TO_AVG * markov_e), dt_ns=float(dt_ns), j_pulse=j, gap=gap,
    )


def _gap_terms(noise: CZNoiseParameters, mech: GapMechanisms, j: np.ndarray, dt_ns: float, F: np.ndarray,
               F_ex: np.ndarray, A: np.ndarray, omega: np.ndarray, n_omega: int) -> Dict[str, float]:
    """The switched-on ``GapMechanisms`` terms (average infidelity), see ``cz_error_budget_for_pulse``."""
    on = mech.active()
    out: Dict[str, float] = {}
    if "gate_zeeman" in on:
        dF = (F_ex - F[2])[None, :]
        qs = noise.exchange_sigma_rel**2 * dF[0, 0] / D
        fast = _fast_part(dF, A[2:3], noise.alphas[2:3], omega)[0]
        out["gate_zeeman"] = float(E_TO_AVG * (qs + fast))
    if "slow_zeeman" in on:
        f_run = float(mech.zeeman_f_run_hz)
        var = [A[q] * _band_integral(f_run, noise.f_low_hz, noise.alphas[q]) for q in range(2)]
        out["slow_zeeman"] = float(E_TO_AVG * (var[0] * F[0, 0] + var[1] * F[1, 0]) / D)
    if "slow_exchange" in on:
        var = A[2] * _band_integral(float(mech.exchange_f_run_hz), noise.f_low_hz, noise.alphas[2])
        out["slow_exchange"] = float(E_TO_AVG * var * F_ex[0] / D)
    if "distortion" in on:
        filt, ref, dt = distorted_pulse(j, dt_ns, float(mech.bandwidth_mhz), mech.voltage_map)
        b_f = cz_error_budget_for_pulse(noise, filt, dt, n_omega=n_omega)
        b_r = cz_error_budget_for_pulse(noise, ref, dt, n_omega=n_omega)
        out["distortion"] = float(b_f.total - b_r.total)
    if "residual_exchange" in on:
        out["residual_exchange"] = residual_exchange_error(noise.delta_ez_mhz, mech.j_off_mhz, mech.idle_ns)
    if "frequency_offset" in on:
        d1, d2 = mech.frequency_offset_mhz
        out["frequency_offset"] = float(E_TO_AVG * (d1**2 * F[0, 0] + d2**2 * F[1, 0]) / D)
    if "exchange_offset" in on:
        out["exchange_offset"] = float(E_TO_AVG * mech.exchange_offset_rel**2 * F_ex[0] / D)
    if "calibration_residual" in on:
        out["calibration_residual"] = float(mech.calibration_residual)
    return out


def cz_error_budget(noise: CZNoiseParameters, shape: str = "cosine", duration_ns: float = 100.0,
                    n_steps: Optional[int] = None, tukey_alpha: float = 0.5, n_omega: int = 240,
                    mechanisms: Optional[GapMechanisms] = None) -> CZErrorBudget:
    """Error budget of a window-shaped CZ (area 500 MHz·ns) of the given duration."""
    j, dt = shaped_cz_pulse(shape, duration_ns, noise.delta_ez_mhz, n_steps, tukey_alpha)
    return cz_error_budget_for_pulse(noise, j, dt, label=shape, n_omega=n_omega, mechanisms=mechanisms)


# --------------------------------------------------------------------------- #
# What the benchmarks report
# --------------------------------------------------------------------------- #
def irb_error_bound(reference_clifford_fidelity: float, interleaved_gate_fidelity: float, d: int = 4) -> Dict[str, float]:
    """
    Range that interleaved RB allows for the true gate error (Magesan et al., PRL 109, 080505
    (2012)): r_est = (d−1)(1 − p_int/p)/d lies within ±E of it, with

        E = min{ (d−1)[|p − p_int/p| + (1 − p)]/d ,  2(d²−1)(1 − p)/(p d²) + 4√(1−p)√(d²−1)/p }.

    The point estimate is exact when the errors are depolarising; E is the worst case over
    error models. When the reference error is larger than the gate error the first form is
    E = 2 r_ref − r_est, so the range is set by the reference. Inputs are average fidelities
    of the reference Clifford and of the interleaved gate as published.
    """
    r_ref = 1.0 - float(reference_clifford_fidelity)
    r_est = 1.0 - float(interleaved_gate_fidelity)
    p = 1.0 - d * r_ref / (d - 1.0)
    ratio = 1.0 - d * r_est / (d - 1.0)                               # p_int / p
    e1 = (d - 1.0) * (abs(p - ratio) + (1.0 - p)) / d
    e2 = 2.0 * (d**2 - 1.0) * (1.0 - p) / (p * d**2) + 4.0 * np.sqrt(1.0 - p) * np.sqrt(d**2 - 1.0) / p
    E = float(min(e1, e2))
    return {"estimate": r_est, "bound": E, "lower": max(0.0, r_est - E), "upper": r_est + E, "p_ref": p}


def sweep_gate_time(noise: CZNoiseParameters, durations_ns: Sequence[float], shape: str = "cosine",
                    tukey_alpha: float = 0.5, n_omega: int = 160) -> List[CZErrorBudget]:
    """``cz_error_budget`` for each duration."""
    return [cz_error_budget(noise, shape, float(T), tukey_alpha=tukey_alpha, n_omega=n_omega) for T in durations_ns]


def shortest_gate_time(shape: str, j_cap_mhz: float, tukey_alpha: float = 0.5) -> float:
    """Shortest duration (ns) whose area-500 MHz·ns window stays below the exchange ``j_cap_mhz``."""
    return float(CZ_AREA_MHZ_NS / (np.mean(window(shape, 2000, tukey_alpha)) * float(j_cap_mhz)))


def coherent_error(noise: CZNoiseParameters, shape: str, duration_ns: float, tukey_alpha: float = 0.5) -> float:
    """Coherent (noiseless) CZ infidelity after virtual-Z correction; cheap enough for fine scans."""
    j, dt = shaped_cz_pulse(shape, duration_ns, noise.delta_ez_mhz, tukey_alpha=tukey_alpha)
    U0 = ExchangeDynamics(noise.hamiltonian()).propagate_unitary(j, dt)
    return float(1.0 - cz_fidelity_up_to_z(U0))


def square_sync_times(delta_ez_mhz: float, t_min_ns: float, t_max_ns: float) -> np.ndarray:
    """
    Durations (ns) at which a square CZ of area 500 MHz·ns leaves no coherent
    error: the |↑↓⟩, |↓↑⟩ precession at √(J² + ΔE_z²) completes n whole cycles,
    with J = 500/T, so T_n = √(n² − ¼)/ΔE_z.
    """
    dez = float(delta_ez_mhz)
    n_lo = max(1, int(np.floor(np.sqrt((1e-3 * t_min_ns * dez) ** 2 + 0.25))))
    n_hi = int(np.ceil(np.sqrt((1e-3 * t_max_ns * dez) ** 2 + 0.25)))
    n = np.arange(n_lo, n_hi + 1)
    T = 1e3 * np.sqrt(n**2 - 0.25) / dez
    return T[(T >= t_min_ns * (1 - 1e-9)) & (T <= t_max_ns)]


def optimal_gate_time(noise: CZNoiseParameters, shape: str = "cosine", bounds_ns: Tuple[float, float] = (10.0, 1000.0),
                      n_grid: int = 60, tukey_alpha: float = 0.5, j_cap_mhz: Optional[float] = None,
                      n_omega: int = 100) -> CZErrorBudget:
    """
    Duration that minimises the modelled CZ infidelity, with durations whose
    peak exchange would exceed ``j_cap_mhz`` excluded.

    Square pulses have sharp minima at the synchronised times
    (``square_sync_times``); those are enumerated, the first few and a
    log-spaced selection of the rest are evaluated, and the best is refined
    within ±1 %. Smooth windows use a log grid of ``n_grid`` points and a
    bounded search between the neighbours of the best one.
    """
    import scipy.optimize

    lo = bounds_ns[0] if j_cap_mhz is None else max(bounds_ns[0], shortest_gate_time(shape, j_cap_mhz, tukey_alpha))
    hi = float(bounds_ns[1])
    f = lambda T: cz_error_budget(noise, shape, float(T), tukey_alpha=tukey_alpha, n_omega=n_omega).total
    if shape == "square":
        Tn = square_sync_times(noise.delta_ez_mhz, lo, hi)
        if len(Tn) > 16:
            Tn = np.unique(np.concatenate([Tn[:8], Tn[np.unique(np.geomspace(1, len(Tn) - 1, 8).astype(int))]]))
        grid = np.unique(np.concatenate([[lo], Tn]))
        tot = [f(T) for T in grid]
        k = int(np.argmin(tot))
        a, b = max(lo, 0.99 * grid[k]), min(hi, 1.01 * grid[k])
    else:
        grid = np.geomspace(lo, hi, int(n_grid))
        tot = [f(T) for T in grid]
        k = int(np.argmin(tot))
        a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    best_T = float(grid[k])
    if b > a:
        res = scipy.optimize.minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 0.01})
        if res.fun < tot[k]:
            best_T = float(res.x)
    return cz_error_budget(noise, shape, best_T, tukey_alpha=tukey_alpha)


# --------------------------------------------------------------------------- #
# Combined simulation (all errors at once)
# --------------------------------------------------------------------------- #
def _channel(Hs: Sequence[np.ndarray], dt_ns: float, Dsup: np.ndarray) -> np.ndarray:
    Hb = np.asarray(Hs, dtype=np.complex128)[:, None]                 # (n, 1, 4, 4)
    S = np.eye(16, dtype=np.complex128)
    for E in _slice_propagators(Hb, dt_ns, Dsup):
        S = E[0] @ S
    return S


def _batched_liouvillian(H: np.ndarray, Dsup: np.ndarray) -> np.ndarray:
    """Column-stacking Liouvillian −i[2π·1e-3·H, ·] + D for a batch of Hamiltonians (B, 4, 4)."""
    Hr = MHZ_NS_TO_RAD * H
    I4 = np.eye(4)
    B = H.shape[0]
    left = np.einsum("ij,bkl->bikjl", I4, Hr).reshape(B, 16, 16)
    right = np.einsum("bji,kl->bikjl", Hr, I4).reshape(B, 16, 16)
    return -1j * (left - right) + Dsup[None]


def _slice_propagators(H_slices: np.ndarray, dt_ns: float, Dsup: np.ndarray):
    for Hk in H_slices:
        yield scipy.linalg.expm(_batched_liouvillian(Hk, Dsup) * dt_ns)


def _fast_traces(A: float, alpha: float, f_lo: float, f_hi: float, t_ns: np.ndarray, n_traces: int,
                 rng: np.random.Generator, n_freq: int = 48) -> np.ndarray:
    """Gaussian traces with one-sided PSD A·f^-α on [f_lo, f_hi] Hz (log-spaced sinusoids, random quadratures)."""
    if A <= 0 or f_hi <= f_lo:
        return np.zeros((n_traces, len(t_ns)))
    edges = np.geomspace(f_lo, f_hi, n_freq + 1)
    fc = np.sqrt(edges[:-1] * edges[1:])
    var = np.array([A * _band_integral(a, b, alpha) for a, b in zip(edges[:-1], edges[1:])])
    ph = 2.0 * np.pi * np.outer(fc, t_ns * 1e-9)                       # (n_freq, n_t)
    a = rng.standard_normal((n_traces, n_freq)) * np.sqrt(var)
    b = rng.standard_normal((n_traces, n_freq)) * np.sqrt(var)
    return a @ np.cos(ph) + b @ np.sin(ph)


def combined_simulation(noise: CZNoiseParameters, j_pulse: np.ndarray, dt_ns: float, n_quadrature: int = 3,
                        n_fast: int = 4, seed: int = 0) -> Dict[str, float]:
    """
    Average gate infidelity of the CZ with every error acting at once.

    Each sample propagates the Lindblad equation (per-qubit T1, white dephasing)
    with the Hamiltonian

        H = J(t)(1 + η_qs + η(t))·σ·σ/4 + (ΔE_z/4)(Z₁ − Z₂) + Σ_i (δf_i + ξ_i(t)) Z_i/2,

    where the quasi-static offsets (η_qs, δf₁, δf₂) are integrated with an
    ``n_quadrature``-point Gauss–Hermite rule in each variable and the fast 1/f
    parts (η, ξ₁, ξ₂) above the Ramsey band are drawn ``n_fast`` times per
    node. The channel is averaged and compared with the virtually-Z-corrected
    CZ calibrated on the noiseless pulse. Returns the infidelity and its
    standard error over the fast-noise draws.
    """
    j = np.asarray(j_pulse, dtype=float)
    n = len(j)
    h = noise.hamiltonian()
    U0 = ExchangeDynamics(h).propagate_unitary(j, dt_ns)
    target = z_corrected_cz_target(U0)
    S_T = np.kron(target.conj(), target)

    x, wq = np.polynomial.hermite_e.hermegauss(int(n_quadrature))
    wq = wq / wq.sum()
    sig = np.array([noise.zeeman_sigma_mhz[0], noise.zeeman_sigma_mhz[1], noise.exchange_sigma_rel])
    grids = np.meshgrid(x, x, x, indexing="ij")
    nodes = np.stack([g.ravel() for g in grids], axis=1) * sig[None, :]     # (n_nodes, 3)
    wn = np.prod(np.stack(np.meshgrid(wq, wq, wq, indexing="ij"), axis=0).reshape(3, -1), axis=0)
    n_nodes = len(nodes)

    rng = np.random.default_rng(seed)
    t = (np.arange(n) + 0.5) * dt_ns
    A = noise.psd_amplitudes()
    f_hi = 0.5 / (dt_ns * 1e-9)                                              # Nyquist of the slicing
    B = n_nodes * int(n_fast)
    fast = [_fast_traces(A[c], noise.alphas[c], noise.f_ramsey_hz, f_hi, t, B, rng) for c in range(3)]

    rep = np.repeat(np.arange(n_nodes), int(n_fast))
    dz1 = nodes[rep, 0][:, None] + fast[0]                                    # (B, n)
    dz2 = nodes[rep, 1][:, None] + fast[1]
    eta = nodes[rep, 2][:, None] + fast[2]

    H_static = (noise.delta_bz_mhz / 2.0) * Z_DIFF
    Dsup = noise.dissipator()
    S = np.broadcast_to(np.eye(16, dtype=np.complex128), (B, 16, 16)).copy()
    for k in range(n):
        Hk = (j[k] * (1.0 + eta[:, k]))[:, None, None] * (HEISENBERG_EXCHANGE / 4.0)[None] + H_static[None] \
            + dz1[:, k, None, None] * (_Z1 / 2.0)[None] + dz2[:, k, None, None] * (_Z2 / 2.0)[None]
        S = scipy.linalg.expm(_batched_liouvillian(Hk, Dsup) * dt_ns) @ S

    fe = np.real(np.einsum("ij,bij->b", S_T.conj(), S)) / 16.0                # per-sample F_e
    w = np.repeat(wn, int(n_fast)) / int(n_fast)
    infid = E_TO_AVG * (1.0 - float(np.sum(w * fe)))
    per_draw = E_TO_AVG * (1.0 - (fe.reshape(n_nodes, int(n_fast)) * wn[:, None]).sum(axis=0))
    stderr = float(np.std(per_draw, ddof=1) / np.sqrt(len(per_draw))) if int(n_fast) > 1 else 0.0
    return {"infidelity": infid, "stderr": stderr, "n_samples": B}
