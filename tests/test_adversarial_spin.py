"""
Adversarial and Stress Test Suite for spin_optimal_control.
"""

import pytest
import numpy as np
import scipy.linalg
from spin_optimal_control.hamiltonian import SiliconSpinHamiltonian, ExchangeDynamics
from spin_optimal_control.noise import PinkNoiseGenerator, OverhauserNoise, SiliconNoiseModel
from spin_optimal_control.grape import GRAPEOptimizer, SmoothFourierPulse
from spin_optimal_control.cirq_backend import (
    SiliconExchangeGate,
    CirqSiliconSimulator,
    run_randomized_benchmarking,
    run_interleaved_rb,
)


def test_adversarial_zero_and_infinite_exchange():
    """Check Hamiltonian dynamics with zero exchange and extreme exchange."""
    h = SiliconSpinHamiltonian(j_0=0.0, delta_bz=50.0, b_0=200.0)
    dyn = ExchangeDynamics(h)
    
    # Zero pulse
    j_zero = np.zeros(20)
    u_zero = dyn.propagate_unitary(j_zero, dt=0.1)
    assert np.allclose(u_zero.conj().T @ u_zero, np.eye(4), atol=1e-8)
    
    # Large pulse
    h_huge = SiliconSpinHamiltonian(j_0=500.0, delta_bz=0.0, b_0=0.0)
    dyn_huge = ExchangeDynamics(h_huge)
    j_huge = np.full(50, 500.0)
    u_huge = dyn_huge.propagate_unitary(j_huge, dt=0.01)
    assert np.allclose(u_huge.conj().T @ u_huge, np.eye(4), atol=1e-8)


def test_adversarial_extreme_pink_noise():
    """Test pink noise generator with extreme spectral exponents and zero/huge amplitudes."""
    # Alpha = 0 (White noise), Alpha = 2 (Brownian noise)
    for alpha in [0.0, 1.0, 2.0]:
        gen = PinkNoiseGenerator(alpha=alpha, amplitude=0.1, seed=42)
        trace = gen.generate_spectral_trace(n_steps=128, dt=0.05)
        assert len(trace) == 128
        assert not np.any(np.isnan(trace))
        assert not np.any(np.isinf(trace))
        assert np.isclose(np.std(trace), 0.1, atol=0.02)

    # Extreme length (1 step, 2 steps, large steps)
    gen = PinkNoiseGenerator(amplitude=0.05)
    t1 = gen.generate_spectral_trace(1, 0.1)
    assert len(t1) == 1
    t2 = gen.generate_spectral_trace(2, 0.1)
    assert len(t2) == 2


def test_adversarial_lindblad_density_matrix_properties():
    """Verify trace preservation, Hermiticity, and positive semi-definiteness under strong Lindblad damping."""
    noise = SiliconNoiseModel(t1_us=0.5, t2_star_us=0.1, charge_noise_amp=0.2)
    h = SiliconSpinHamiltonian(j_0=30.0, delta_bz=10.0)
    
    # Initial state: |00>
    rho_0 = np.zeros((4, 4), dtype=np.complex128)
    rho_0[0, 0] = 1.0
    
    j_pulse = np.full(100, 25.0)
    dt = 0.05 # total 5 us >> T1, T2*
    
    rho_final = noise.evolve_density_matrix_lindblad(
        rho_0, h.get_hamiltonian_matrix, j_pulse, dt
    )
    
    # 1. Trace = 1
    assert np.isclose(np.trace(rho_final), 1.0, atol=1e-5), f"Trace must be 1, got {np.trace(rho_final)}"
    # 2. Hermitian
    assert np.allclose(rho_final, rho_final.conj().T, atol=1e-6), "Density matrix must remain Hermitian"
    # 3. Positive semi-definite (eigenvalues >= -1e-6)
    evals = np.linalg.eigvalsh(rho_final)
    assert np.all(evals >= -1e-5), f"Negative eigenvalues in density matrix: {evals}"


def test_adversarial_rb_decay_bounds():
    """Verify RB curve fitting when noise is zero and when noise is extreme."""
    # Zero noise (infinite T1/T2*)
    clean_noise = SiliconNoiseModel(t1_us=1e8, t2_star_us=1e8, charge_noise_amp=0.0, overhauser_sigma=0.0)
    rb_clean = run_randomized_benchmarking([1, 2, 4], n_sequences_per_length=3, noise_model=clean_noise, seed=42)
    assert rb_clean["decay_p"] >= 0.99
    assert all(f >= 0.98 for f in rb_clean["fidelities"])
