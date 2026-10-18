"""
CZ error budget: first-order contributions against a combined simulation,
physical limits, and closed forms.
"""

import numpy as np
import pytest
import scipy.integrate

from spin_optimal_control.error_budget import (
    CZNoiseParameters,
    coherent_error,
    combined_simulation,
    cz_error_budget,
    cz_error_budget_for_pulse,
    echo_integral,
    echo_time_from_1f,
    fit_zeeman_alpha,
    psd_amplitude,
    shaped_cz_pulse,
    square_sync_times,
    white_dephasing_t2,
    zeeman_sigma_from_t2_star,
)
from spin_optimal_control.hamiltonian import ExchangeDynamics, SiliconSpinHamiltonian
from spin_optimal_control.pulse_shaping import cz_fidelity_up_to_z

# A noise environment where all four contributions are of similar size (≈ 1e-4 – 1e-3)
NOISY = CZNoiseParameters(
    delta_ez_mhz=30.0,
    zeeman_sigma_mhz=(zeeman_sigma_from_t2_star(3.0), zeeman_sigma_from_t2_star(5.0)),
    exchange_sigma_rel=0.03,
    t1_us=(200.0, 300.0),
    t2_markov_us=(60.0, 90.0),
    f_low_hz=1e-2,
    f_ramsey_hz=2.5e5,
    temperature_mk=100.0,
    larmor_mhz=(4000.0, 4000.0),
)


def test_package_hamiltonian_uses_half_the_frequency_difference():
    """f₁ − f₂ = 2 ΔB_z in the package Hamiltonian, so ΔB_z = ΔE_z / 2."""
    nm = CZNoiseParameters(delta_ez_mhz=80.0)
    H = nm.hamiltonian().get_hamiltonian_matrix(0.0, lab_frame=True)
    e = np.real(np.diag(H))
    f1, f2 = e[0] - e[2], e[0] - e[1]
    assert f1 - f2 == pytest.approx(80.0)


def test_zero_noise_leaves_only_the_coherent_error():
    nm = NOISY.noiseless()
    b = cz_error_budget(nm, "square", 60.0)
    assert b.zeeman == 0.0 and b.exchange == 0.0 and b.markovian == 0.0
    U = ExchangeDynamics(nm.hamiltonian()).propagate_unitary(b.j_pulse, b.dt_ns)
    assert b.total == pytest.approx(1.0 - cz_fidelity_up_to_z(U), rel=1e-12)
    assert b.coherent > 1e-5                     # a 60 ns square CZ at 30 MHz is visibly non-adiabatic
    sim = combined_simulation(nm, b.j_pulse, b.dt_ns, n_fast=1)
    assert sim["infidelity"] == pytest.approx(b.coherent, rel=1e-6, abs=1e-12)


def test_first_order_budget_adds_up_to_the_combined_simulation():
    """All four error sources at once (Lindblad + quasi-static + 1/f traces) against the sum of the terms."""
    b = cz_error_budget(NOISY, "cosine", 80.0)
    parts = b.contributions
    assert all(v > 3e-5 for v in parts.values()), parts
    assert max(parts.values()) < 0.6 * b.total         # no single term dominates the check
    sim = combined_simulation(NOISY, b.j_pulse, b.dt_ns, n_quadrature=3, n_fast=6, seed=3)
    assert abs(sim["infidelity"] - b.total) < 3 * sim["stderr"] + 0.04 * b.total


@pytest.mark.parametrize("which", ["zeeman", "exchange", "markovian"])
def test_each_noise_term_matches_its_own_simulation(which):
    off = dict(zeeman_sigma_mhz=(0.0, 0.0), exchange_sigma_rel=0.0, t1_us=(np.inf, np.inf), t2_markov_us=(np.inf, np.inf))
    keep = {"zeeman": ["zeeman_sigma_mhz"], "exchange": ["exchange_sigma_rel"], "markovian": ["t1_us", "t2_markov_us"]}[which]
    for k in keep:
        off.pop(k)
    nm = NOISY.replace(**off)
    b = cz_error_budget(nm, "cosine", 80.0)
    others = {k: v for k, v in b.contributions.items() if k not in (which, "coherent")}
    assert all(v == 0.0 for v in others.values())
    sim = combined_simulation(nm, b.j_pulse, b.dt_ns, n_fast=6, seed=1)
    assert abs(sim["infidelity"] - b.total) < 3 * sim["stderr"] + 0.03 * b.total


def test_quasistatic_terms_match_closed_forms_when_the_gradient_is_large():
    """ΔE_z ≫ J: Zeeman noise is a Z rotation, 1 − F_e = σ²(2π·10⁻³T)²/4 per qubit; exchange noise is a ZZ error, π²σ_J²/16."""
    T = 100.0
    nm = CZNoiseParameters(delta_ez_mhz=400.0, zeeman_sigma_mhz=(0.02, 0.03), exchange_sigma_rel=0.01,
                           f_low_hz=1e-2, f_ramsey_hz=1e5)
    b = cz_error_budget(nm, "cosine", T)
    zeeman = 0.8 * (0.02**2 + 0.03**2) * (2 * np.pi * 1e-3 * T) ** 2 / 4
    exchange = 0.8 * np.pi**2 * 0.01**2 / 16
    assert b.zeeman_quasistatic == pytest.approx(zeeman, rel=0.01)
    assert b.exchange_quasistatic == pytest.approx(exchange, rel=0.01)
    # quasi-static exchange error does not depend on the gate time (fixed area 500 MHz·ns)
    assert cz_error_budget(nm, "cosine", 2 * T).exchange_quasistatic == pytest.approx(b.exchange_quasistatic, rel=0.01)


def test_fast_noise_terms_are_small_and_scale_with_the_spectrum():
    nm = NOISY.replace(t1_us=(np.inf, np.inf), t2_markov_us=(np.inf, np.inf))
    b = cz_error_budget(nm, "cosine", 80.0)
    assert 0 < b.zeeman_fast < 0.5 * b.zeeman_quasistatic
    assert 0 < b.exchange_fast < 0.5 * b.exchange_quasistatic
    b2 = cz_error_budget(nm.replace(exchange_sigma_rel=2 * nm.exchange_sigma_rel), "cosine", 80.0)
    assert b2.exchange_fast == pytest.approx(4 * b.exchange_fast, rel=1e-9)


def test_budget_converges_with_time_slicing():
    nm = NOISY
    a = cz_error_budget(nm, "cosine", 60.0)
    j, dt = shaped_cz_pulse("cosine", 60.0, nm.delta_ez_mhz, n_steps=2 * len(a.j_pulse))
    b = cz_error_budget_for_pulse(nm, j, dt)
    for k in a.contributions:
        assert b.contributions[k] == pytest.approx(a.contributions[k], rel=0.03, abs=1e-8)


def test_echo_time_of_1f_noise_matches_numerical_filter_integral():
    """Closed form T2 = 1/(π√(2A ln 2)) against ∫ S(f)|Y_echo(f)|² df."""
    sigma, f_lo, f_r = 0.05, 1e-2, 2e5
    t2 = echo_time_from_1f(sigma, f_lo, f_r)
    A = sigma**2 / np.log(f_r / f_lo)                    # MHz², S(f) = A/f
    t = t2                                                # µs
    # ⟨φ²⟩ = (2π)² ∫ A/f · 16 sin⁴(πft/2)/(2πf)² df, f in MHz
    integrand = lambda lf: A * 16 * np.sin(np.pi * np.exp(lf) * t / 2) ** 4 / (2 * np.pi * np.exp(lf)) ** 2
    val, _ = scipy.integrate.quad(integrand, np.log(f_lo * 1e-6), np.log(1e3), limit=400)
    phi2 = (2 * np.pi) ** 2 * val
    assert phi2 / 2 == pytest.approx(1.0, rel=0.01)


@pytest.mark.parametrize("alpha", [0.7, 1.3, 1.8])
def test_echo_time_of_1f_alpha_noise_matches_numerical_filter_integral(alpha):
    """T2 = (2/π)(8 A I(α))^(−1/(α+1)) against ⟨φ²⟩ = (2π)² ∫ S(f)|Y_echo(f)|² df with the low cut-off kept."""
    sigma, f_lo, f_r = 0.05, 1e-2, 2e5
    t2 = echo_time_from_1f(sigma, f_lo, f_r, alpha)                  # µs
    A = 1e12 * psd_amplitude(sigma, f_lo, f_r, alpha)                # Hz²·Hz^(α−1), f in Hz
    t = 1e-6 * t2
    integrand = lambda lf: np.exp(lf) * A * np.exp(lf) ** (-alpha) * 16 * np.sin(np.pi * np.exp(lf) * t / 2) ** 4 \
        / (2 * np.pi * np.exp(lf)) ** 2
    edges = np.linspace(np.log(f_lo), np.log(1e9), 41)
    val = sum(scipy.integrate.quad(integrand, a, b, limit=200)[0] for a, b in zip(edges[:-1], edges[1:]))
    assert (2 * np.pi) ** 2 * val / 2 == pytest.approx(1.0, rel=0.01)


def test_echo_integral_closed_forms():
    assert echo_integral(1.0) == pytest.approx(np.log(2.0), rel=1e-9)
    assert echo_integral(2.0) == pytest.approx(np.pi / 3, rel=1e-6)


def test_t2_star_width_matches_a_simulated_ramsey_decay():
    """Quasi-static Gaussian detuning of width σ: the averaged Ramsey signal falls to 1/e at T2* = √2/(2πσ)."""
    rng = np.random.default_rng(0)
    t2s = 7.0                                                           # µs
    df = rng.normal(0.0, zeeman_sigma_from_t2_star(t2s), 200_000)       # MHz
    t = np.linspace(0.0, 2.0 * t2s, 401)
    signal = np.array([np.mean(np.cos(2 * np.pi * df * tt)) for tt in t])
    t_e = np.interp(-np.exp(-1.0), -signal, t)                          # where the signal falls to 1/e
    assert t_e == pytest.approx(t2s, rel=0.01)
    assert np.interp(t2s, t, signal) == pytest.approx(np.exp(-1.0), abs=0.005)


def test_alpha_fit_reproduces_the_echo_and_clips_at_the_ends():
    sigma, f_lo, f_r = zeeman_sigma_from_t2_star(5.0), 1e-2, 2e5
    for target in (15.0, 60.0):
        fit = fit_zeeman_alpha(sigma, target, f_lo, f_r)
        assert fit["status"] == "fitted"
        assert echo_time_from_1f(sigma, f_lo, f_r, fit["alpha"]) == pytest.approx(target, rel=1e-5)
    short = fit_zeeman_alpha(sigma, 5.5, f_lo, f_r)                     # barely longer than T2*: too short for α ≥ 0.5
    assert short["status"] == "clipped at lowest" and short["alpha"] == 0.5
    assert np.isfinite(white_dephasing_t2(5.5, sigma, f_lo, f_r, alpha=0.5))
    assert fit_zeeman_alpha(sigma, None, f_lo, f_r)["status"] == "no echo"


def test_square_pulse_is_exact_at_the_synchronised_times():
    nm = CZNoiseParameters(delta_ez_mhz=103.0)
    Tn = square_sync_times(103.0, 25.0, 80.0)
    assert Tn[0] == pytest.approx(1e3 * np.sqrt(9 - 0.25) / 103.0)           # n = 3, 28.7 ns
    assert all(coherent_error(nm, "square", T) < 1e-10 for T in Tn)
    assert coherent_error(nm, "square", 0.5 * (Tn[0] + Tn[1])) > 1e-3


def test_white_dephasing_is_only_the_part_of_the_echo_that_1f_noise_misses():
    sigma, f_lo, f_r = 0.05, 1e-2, 2e5
    t2f = echo_time_from_1f(sigma, f_lo, f_r)
    assert white_dephasing_t2(1.5 * t2f, sigma, f_lo, f_r) == np.inf      # 1/f already explains the echo
    t2w = white_dephasing_t2(0.5 * t2f, sigma, f_lo, f_r)
    assert t2w == pytest.approx(0.5 * t2f / (1 - 0.25), rel=1e-9)
    assert white_dephasing_t2(np.inf, sigma, f_lo, f_r, t1_us=100.0) == pytest.approx(200.0)
    t2f15 = echo_time_from_1f(sigma, f_lo, f_r, 1.5)                    # residual uses the exponent α + 1
    assert white_dephasing_t2(0.5 * t2f15, sigma, f_lo, f_r, alpha=1.5) == pytest.approx(0.5 * t2f15 / (1 - 0.5**2.5), rel=1e-9)


def test_gate_time_tradeoff_has_an_interior_optimum():
    nm = NOISY.replace(exchange_sigma_rel=0.0)
    totals = [cz_error_budget(nm, "cosine", T, n_omega=80).total for T in (15.0, 60.0, 600.0)]
    assert totals[1] < totals[0] and totals[1] < totals[2]
    short, long_ = cz_error_budget(nm, "cosine", 15.0), cz_error_budget(nm, "cosine", 600.0)
    assert short.dominant == "coherent" and long_.dominant in ("zeeman", "markovian")
