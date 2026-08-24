"""
Unit tests for Silicon Spin Optimal Control.
"""

import pytest
import numpy as np
from spin_optimal_control.hamiltonian import (
    SiliconSpinHamiltonian,
    ExchangeDynamics,
    HEISENBERG_EXCHANGE,
    Z_DIFF,
    Z_SUM,
)
from spin_optimal_control.noise import (
    PinkNoiseGenerator,
    OverhauserNoise,
    SiliconNoiseModel,
)
from spin_optimal_control.grape import GRAPEOptimizer, SmoothFourierPulse
from spin_optimal_control.cirq_backend import (
    SiliconExchangeGate,
    generate_single_qubit_cliffords,
    run_randomized_benchmarking,
)


def test_hamiltonian_hermiticity():
    h = SiliconSpinHamiltonian(j_0=25.0, delta_bz=10.0, b_0=120.0)
    mat = h.get_hamiltonian_matrix(j_val=30.0)
    assert np.allclose(mat, mat.conj().T), "Hamiltonian matrix must be Hermitian"
    assert mat.shape == (4, 4)


def test_singlet_triplet_basis_orthonormality():
    s, t0, tp, tm = SiliconSpinHamiltonian.get_singlet_triplet_basis()
    basis = [s, t0, tp, tm]
    for i in range(4):
        for j in range(4):
            inner_prod = np.vdot(basis[i], basis[j])
            expected = 1.0 if i == j else 0.0
            assert np.isclose(inner_prod, expected, atol=1e-7)


def test_unitary_propagation_preservation():
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=10.0)
    dyn = ExchangeDynamics(h)
    j_pulse = np.array([5.0, 15.0, 30.0, 20.0, 0.0])
    dt = 0.5

    u = dyn.propagate_unitary(j_pulse, dt)
    # Check U^dagger U = I
    assert np.allclose(u.conj().T @ u, np.eye(4), atol=1e-7), "Propagated operator must be unitary"


def test_pink_noise_generation():
    gen = PinkNoiseGenerator(alpha=1.0, amplitude=0.05, seed=123)
    trace = gen.generate_spectral_trace(n_steps=256, dt=0.1)
    assert len(trace) == 256
    assert np.isclose(np.std(trace), 0.05, atol=1e-2)


def test_overhauser_noise_sampling():
    oh = OverhauserNoise(sigma_overhauser=0.8, seed=42)
    shifts = oh.sample_ensemble_shifts(n_samples=500)
    assert np.isclose(np.mean(shifts), 0.0, atol=0.15)
    assert np.isclose(np.std(shifts), 0.8, atol=0.15)


def test_smooth_pulse_boundary_conditions():
    pulse_basis = SmoothFourierPulse(n_harmonics=4, t_gate_ns=20.0)
    t_grid = np.linspace(0.0, 20.0, 50)
    params = np.array([10.0, 5.0, -2.0, 1.0, 4.0, -1.0, 2.0, 0.5])
    pulse = pulse_basis.evaluate_pulse_np(params, t_grid)

    assert np.isclose(pulse[0], 0.0, atol=1e-6), "Pulse must start at 0 at t=0"
    assert np.isclose(pulse[-1], 0.0, atol=1e-6), "Pulse must end at 0 at t=T"
    assert np.all(pulse >= 0.0), "Pulse exchange must be non-negative"


def test_grape_optimization_convergence():
    h = SiliconSpinHamiltonian(j_0=20.0, epsilon_0=1.0, delta_bz=0.0, b_0=0.0)
    dyn = ExchangeDynamics(h)
    u_target = dyn.target_gate_sqrt_swap()

    opt = GRAPEOptimizer(h, t_gate_ns=25.0, n_steps=40, n_harmonics=4)
    res = opt.optimize_pulse(u_target, max_iter=50, tolerance=1e-4)

    assert res.gate_fidelity > 0.98, f"GRAPE fidelity should exceed 98%, got {res.gate_fidelity:.4f}"
    assert res.j_pulse.shape == (40,)


def test_cirq_gate_and_cliffords():
    cliffords = generate_single_qubit_cliffords()
    assert len(cliffords) == 24, "Single qubit Clifford group must contain 24 elements"

    u4 = np.eye(4, dtype=np.complex128)
    gate = SiliconExchangeGate(u4)
    assert gate.num_qubits() == 2
    assert np.allclose(gate._unitary_(), u4)


def test_cirq_rb_pipeline():
    lengths = [1, 2, 4]
    noise = SiliconNoiseModel(t1_us=1000.0, t2_star_us=50.0, charge_noise_amp=0.01)
    res = run_randomized_benchmarking(lengths, n_sequences_per_length=3, noise_model=noise, seed=42)
    assert len(res["fidelities"]) == 3
    assert res["decay_p"] > 0.8
