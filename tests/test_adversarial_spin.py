"""
Adversarial / stress tests: extreme parameters, degenerate inputs, and
invariants that must hold regardless of physics regime.
"""

import numpy as np
import pytest

from spin_optimal_control.hamiltonian import SiliconSpinHamiltonian, ExchangeDynamics
from spin_optimal_control.noise import PinkNoiseGenerator, SiliconNoiseModel
from spin_optimal_control.grape import GRAPEOptimizer, SmoothFourierPulse
from spin_optimal_control.valley import SiliconValleyModel
from spin_optimal_control.awg_export import export_awg_waveforms


def test_zero_and_huge_exchange_stay_unitary():
    h = SiliconSpinHamiltonian(j_0=0.0, delta_bz=50.0)
    u0 = ExchangeDynamics(h).propagate_unitary(np.zeros(20), 0.1)
    assert np.allclose(u0.conj().T @ u0, np.eye(4), atol=1e-10)

    h_huge = SiliconSpinHamiltonian(j_0=5000.0, delta_bz=0.0)
    u1 = ExchangeDynamics(h_huge).propagate_unitary(np.full(50, 5000.0), 0.01)
    assert np.allclose(u1.conj().T @ u1, np.eye(4), atol=1e-10)


def test_zero_pulse_is_identity_in_rotating_frame_without_gradient():
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=0.0, b_0=5000.0)
    u = ExchangeDynamics(h).propagate_unitary(np.zeros(30), 1.0)
    assert ExchangeDynamics.gate_fidelity(u, np.eye(4)) > 1 - 1e-12


def test_pulse_basis_handles_zero_params_and_negative_coefficients():
    basis = SmoothFourierPulse(n_harmonics=3, t_gate_ns=10.0, j_max=20.0)
    t = np.linspace(0, 10, 21)
    assert np.allclose(basis.evaluate(np.zeros(6), t), 0.0)
    pulse = basis.evaluate(np.array([-30.0, 0, 0, 0, 0, 0]), t)
    assert np.all(pulse == 0.0)  # negative raw values are clipped to zero


def test_grape_rejects_wrong_size_initial_params():
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=0.0)
    opt = GRAPEOptimizer(h, t_gate_ns=20.0, n_steps=10, n_harmonics=2)
    with pytest.raises(Exception):
        opt.optimize_pulse(ExchangeDynamics.target_gate_swap(), initial_params=np.zeros(99), max_iter=2)


def test_grape_single_step_pulse_runs():
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=0.0)
    opt = GRAPEOptimizer(h, t_gate_ns=5.0, n_steps=1, n_harmonics=1, slew_penalty=0.0)
    res = opt.optimize_pulse(np.eye(4), max_iter=5)
    assert res.j_pulse.shape == (1,)
    assert np.isfinite(res.gate_fidelity)


def test_pink_noise_degenerate_band_returns_zeros():
    gen = PinkNoiseGenerator(alpha=1.0, amplitude=0.1, f_min_hz=1e20, f_max_hz=1e21, seed=0)
    trace = gen.generate_spectral_trace(64, 0.1)
    assert np.allclose(trace, 0.0)
    assert gen.generate_multitrap_charge_noise(0, 0.1).shape == (0,)


def test_lindblad_zero_time_is_identity_channel():
    model = SiliconNoiseModel(t1_us=1.0, t2_star_us=0.1)
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=10.0)
    rho0 = np.eye(4, dtype=np.complex128) / 4
    rho = model.evolve_density_matrix_lindblad(rho0, h.get_hamiltonian_matrix, np.zeros(0), 1.0)
    assert np.allclose(rho, rho0)


def test_lindblad_extreme_damping_reaches_ground_state():
    model = SiliconNoiseModel(t1_us=0.001, t2_star_us=0.001, charge_noise_amp=0.0, overhauser_sigma=0.0)
    h = SiliconSpinHamiltonian(j_0=0.0, delta_bz=0.0)
    rho0 = np.zeros((4, 4), dtype=np.complex128); rho0[0, 0] = 1.0
    rho = model.evolve_density_matrix_lindblad(rho0, h.get_hamiltonian_matrix, np.zeros(50), 1.0)
    assert np.isclose(np.real(rho[3, 3]), 1.0, atol=1e-6)  # |11> = both electrons relaxed
    assert np.all(np.linalg.eigvalsh(rho) >= -1e-9)


def test_valley_zero_soc_never_leaks():
    vm = SiliconValleyModel(valley_splitting_uev=50.0, inter_valley_soc_mhz=0.0)
    res = vm.compute_valley_leakage(np.full(40, 40.0), dt_ns=0.5)
    assert np.allclose(res["leakage_vs_time"], 0.0, atol=1e-12)


def test_valley_empty_pulse():
    vm = SiliconValleyModel()
    res = vm.compute_valley_leakage(np.zeros(0), dt_ns=0.5)
    assert res["final_valley_leakage"] == 0.0


def test_awg_export_single_sample(tmp_path):
    data = export_awg_waveforms(np.array([0.0]), np.array([1.0]), np.array([0.0]), file_path=str(tmp_path / "a.json"))
    assert data["metadata"]["num_samples"] == 1


def test_local_z_free_fidelity_is_bounded():
    rng = np.random.default_rng(0)
    for _ in range(3):
        a = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        q, _ = np.linalg.qr(a)
        f, _ = ExchangeDynamics.gate_fidelity_local_z_free(q, ExchangeDynamics.target_gate_cz())
        assert 0.0 <= f <= 1.0 + 1e-12
