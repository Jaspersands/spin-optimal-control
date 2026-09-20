"""
Tests for noise generation, the Lindblad solver, and the filter-function
infidelity estimate. Units: MHz / ns; T1, T2* in µs.
"""

import numpy as np
import pytest

from spin_optimal_control.hamiltonian import SiliconSpinHamiltonian, ExchangeDynamics
from spin_optimal_control.noise import (
    PinkNoiseGenerator,
    OverhauserNoise,
    SiliconNoiseModel,
    filter_function,
    filter_function_infidelity,
)


def test_pink_noise_rms_and_psd_slope():
    gen = PinkNoiseGenerator(alpha=1.0, amplitude=0.05, f_min_hz=1e3, f_max_hz=1e12, seed=7)
    dt = 0.1  # ns
    n = 4096
    trace = gen.generate_spectral_trace(n, dt)
    assert trace.shape == (n,)
    assert np.isclose(np.std(trace), 0.05, atol=5e-3)

    # Average the periodogram over independent traces to estimate the PSD slope.
    psd_acc = None
    for _ in range(40):
        f_hz, psd = gen.psd_estimate(gen.generate_spectral_trace(n, dt), dt)
        psd_acc = psd if psd_acc is None else psd_acc + psd
    psd_acc /= 40
    band = (f_hz > 5e6) & (f_hz < 2e9)  # mid-band, away from cut-offs
    slope = np.polyfit(np.log10(f_hz[band]), np.log10(psd_acc[band]), 1)[0]
    assert -1.3 < slope < -0.7


@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_pink_noise_any_alpha_is_finite(alpha):
    gen = PinkNoiseGenerator(alpha=alpha, amplitude=0.1, seed=3)
    t = gen.generate_spectral_trace(256, 0.05)
    assert np.all(np.isfinite(t))
    assert np.isclose(np.std(t), 0.1, atol=0.02)
    assert len(gen.generate_spectral_trace(1, 0.1)) == 1
    assert len(gen.generate_spectral_trace(2, 0.1)) == 2


def test_multitrap_noise_units_are_consistent():
    # Switching rates span [f_min, f_max] in Hz while dt is in ns; every trap
    # must contribute (no exp(-1e8*0.5) underflow), so the trace is not white.
    gen = PinkNoiseGenerator(amplitude=0.03, f_min_hz=1e5, f_max_hz=1e9, seed=11)
    trace = gen.generate_multitrap_charge_noise(2000, dt_ns=0.5, n_traps=12)
    assert np.isclose(np.std(trace), 0.03, atol=5e-3)
    # lag-1 autocorrelation of a sum of OU processes with slow traps is clearly positive
    ac1 = np.corrcoef(trace[:-1], trace[1:])[0, 1]
    assert ac1 > 0.3


def test_overhauser_sampling():
    oh = OverhauserNoise(sigma_overhauser=0.8, seed=42)
    shifts = oh.sample_ensemble_shifts(2000)
    assert np.isclose(np.mean(shifts), 0.0, atol=0.08)
    assert np.isclose(np.std(shifts), 0.8, atol=0.08)


def test_lindblad_rates_conversion():
    model = SiliconNoiseModel(t1_us=100.0, t2_star_us=10.0)
    r = model.lindblad_rates_per_ns()
    assert np.isclose(r["gamma_1"], 1e-3 / 100.0)
    # 1/T_phi = 1/T2 - 1/(2 T1)
    assert np.isclose(r["gamma_phi"], 1e-3 * (1.0 / 10.0 - 0.5 / 100.0))


def test_lindblad_t1_decay_has_correct_timescale():
    # Two qubits each relaxing with T1 = 1 µs from |↑↑> = |00>; after 1000 ns the
    # population remaining in |00> is e^{-2} (independent decays), the
    # coherent Hamiltonian being switched off (J = 0, ΔBz = 0).
    model = SiliconNoiseModel(t1_us=1.0, t2_star_us=1e6, charge_noise_amp=0.0, overhauser_sigma=0.0)
    h = SiliconSpinHamiltonian(j_0=0.0, delta_bz=0.0)
    rho0 = np.zeros((4, 4), dtype=np.complex128); rho0[0, 0] = 1.0
    j_pulse = np.zeros(1000)
    rho = model.evolve_density_matrix_lindblad(rho0, h.get_hamiltonian_matrix, j_pulse, dt_ns=1.0)
    assert np.isclose(np.real(rho[0, 0]), np.exp(-2.0), rtol=0.03)
    assert np.isclose(np.real(rho[3, 3]), (1 - np.exp(-1.0)) ** 2, rtol=0.03)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-8)


def test_lindblad_dephasing_timescale():
    # Pure dephasing of a |+> ⊗ |0> coherence: T2* = 0.5 µs, T1 = ∞ →
    # off-diagonal element decays by e^{-t/T2*}.
    model = SiliconNoiseModel(t1_us=1e9, t2_star_us=0.5, charge_noise_amp=0.0, overhauser_sigma=0.0)
    h = SiliconSpinHamiltonian(j_0=0.0, delta_bz=0.0)
    plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    zero = np.array([1, 0], dtype=np.complex128)
    psi = np.kron(plus, zero)
    rho0 = np.outer(psi, psi.conj())
    rho = model.evolve_density_matrix_lindblad(rho0, h.get_hamiltonian_matrix, np.zeros(500), dt_ns=1.0)
    coherence = abs(rho[0, 2])  # <00|rho|10>
    assert np.isclose(coherence, 0.5 * np.exp(-500.0 / 500.0), rtol=0.03)


def test_lindblad_cptp_under_strong_damping():
    noise = SiliconNoiseModel(t1_us=0.05, t2_star_us=0.01, charge_noise_amp=0.2)
    h = SiliconSpinHamiltonian(j_0=30.0, delta_bz=10.0)
    rho0 = np.zeros((4, 4), dtype=np.complex128); rho0[0, 0] = 1.0
    rho = noise.evolve_density_matrix_lindblad(rho0, h.get_hamiltonian_matrix, np.full(200, 25.0), dt_ns=0.5)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-6)
    assert np.allclose(rho, rho.conj().T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(rho) >= -1e-7)


def test_channel_process_fidelity_matches_unitary_when_noiseless():
    model = SiliconNoiseModel(t1_us=1e12, t2_star_us=1e12, charge_noise_amp=0.0, overhauser_sigma=0.0)
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=0.0)
    dyn = ExchangeDynamics(h)
    j_pulse = np.full(100, 20.0)
    dt = (0.25 / 20.0 * 1e3) / 100  # sqrt(SWAP) area
    u = dyn.propagate_unitary(j_pulse, dt)
    target = ExchangeDynamics.target_gate_sqrt_swap()
    f_unitary = dyn.gate_fidelity(u, target)
    f_channel = model.channel_process_fidelity(j_pulse, dt, target, h.get_hamiltonian_matrix)
    assert np.isclose(f_channel, f_unitary, atol=1e-6)
    assert f_channel > 1 - 1e-6

    noisy = SiliconNoiseModel(t1_us=1.0, t2_star_us=0.2, charge_noise_amp=0.0, overhauser_sigma=0.0)
    f_noisy = noisy.channel_process_fidelity(j_pulse, dt, target, h.get_hamiltonian_matrix)
    assert f_noisy < f_channel - 1e-3


def test_filter_function_and_infidelity():
    dt = 0.5
    j_pulse = 20.0 * np.sin(np.pi * (np.arange(60) + 0.5) / 60)
    omega = np.linspace(1e-4, 5.0, 400)  # rad/ns
    ff = filter_function(j_pulse, dt, omega)
    assert ff.shape == omega.shape
    assert np.all(ff >= 0)
    # DC value equals |∫ s(t) dt|²
    assert np.isclose(ff[0], (np.sum(j_pulse) * dt) ** 2, rtol=1e-3)

    psd_small = lambda w: 1e-6 / np.maximum(w, 1e-4)
    psd_large = lambda w: 1e-4 / np.maximum(w, 1e-4)
    inf_small = filter_function_infidelity(j_pulse, dt, psd_small, sensitivity=1.0)
    inf_large = filter_function_infidelity(j_pulse, dt, psd_large, sensitivity=1.0)
    assert 0.0 <= inf_small < inf_large
    assert np.isclose(inf_large / inf_small, 100.0, rtol=1e-6)
