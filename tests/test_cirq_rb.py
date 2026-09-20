"""
Tests for the Cirq backend: Clifford group generation, noise channels, and
(interleaved) randomized benchmarking.
"""

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")

from spin_optimal_control.hamiltonian import ExchangeDynamics
from spin_optimal_control.noise import SiliconNoiseModel
from spin_optimal_control.cirq_backend import (
    SiliconExchangeGate,
    CirqSiliconSimulator,
    generate_single_qubit_cliffords,
    generate_two_qubit_cliffords,
    clifford_key,
    run_randomized_benchmarking,
    run_interleaved_rb,
    rb_fit,
    clifford_error_from_p,
)


def test_single_qubit_clifford_group():
    cl = generate_single_qubit_cliffords()
    assert len(cl) == 24
    keys = {clifford_key(u) for u in cl}
    assert len(keys) == 24


def test_two_qubit_clifford_group_order_and_closure():
    cl = generate_two_qubit_cliffords()
    assert len(cl) == 11520
    keys = {clifford_key(u) for u in cl}
    assert len(keys) == 11520
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = cl[rng.integers(len(cl))]
        b = cl[rng.integers(len(cl))]
        assert clifford_key(a @ b) in keys
    for u in cl[:500]:
        assert np.allclose(u.conj().T @ u, np.eye(4), atol=1e-9)
    # CNOT and the tensor-product Cliffords are members
    assert clifford_key(cirq.unitary(cirq.CNOT)) in keys
    assert clifford_key(np.kron(cirq.unitary(cirq.H), cirq.unitary(cirq.S))) in keys


def test_silicon_exchange_gate():
    u = ExchangeDynamics.target_gate_sqrt_swap()
    g = SiliconExchangeGate(u, name="sqrtSWAP")
    assert g.num_qubits() == 2
    assert np.allclose(cirq.unitary(g), u)
    with pytest.raises(ValueError):
        SiliconExchangeGate(np.eye(3))


def test_noisy_circuit_phase_damping_parameter():
    # phase_damp(γ) shrinks coherences by sqrt(1-γ); to reproduce exp(-t/Tφ) we
    # need γ = 1 - exp(-2 t/Tφ).
    noise = SiliconNoiseModel(t1_us=1e9, t2_star_us=0.1)  # Tφ = 100 ns
    sim = CirqSiliconSimulator(noise)
    q = cirq.LineQubit(0)
    c = cirq.Circuit(cirq.H(q))
    noisy = sim.create_noisy_circuit(c, gate_duration_ns=50.0)
    rho = cirq.DensityMatrixSimulator().simulate(noisy).final_density_matrix
    assert np.isclose(abs(rho[0, 1]), 0.5 * np.exp(-50.0 / 100.0), atol=1e-3)


def test_rb_fit_and_error_relation():
    m = np.array([1, 2, 4, 8, 16, 32])
    p_true = 0.97
    surv = 0.5 * p_true**m + 0.25
    A, p, B = rb_fit(m, surv)
    assert np.isclose(p, p_true, atol=1e-6)
    assert np.isclose(clifford_error_from_p(p, d=4), (1 - p_true) * 0.75)


def test_rb_clean_has_no_decay():
    clean = SiliconNoiseModel(t1_us=1e9, t2_star_us=1e9, charge_noise_amp=0.0, overhauser_sigma=0.0)
    res = run_randomized_benchmarking([1, 4, 8], n_sequences_per_length=3, noise_model=clean, seed=1)
    assert res["decay_p"] > 0.995
    assert all(f > 0.99 for f in res["fidelities"])
    assert res["clifford_set"] == "two_qubit"


def test_rb_decays_with_noise_and_uses_2q_cliffords():
    noisy = SiliconNoiseModel(t1_us=5.0, t2_star_us=1.0, charge_noise_amp=0.0, overhauser_sigma=0.0)
    res = run_randomized_benchmarking([1, 2, 4, 8], n_sequences_per_length=4, noise_model=noisy, seed=2)
    assert res["fidelities"][0] > res["fidelities"][-1]
    assert 0.5 < res["decay_p"] < 0.999


def test_interleaved_rb_recovers_known_depolarizing_error():
    # Clean Cliffords; the interleaved gate carries cirq.depolarize(p, n_qubits=2).
    # That channel has process fidelity 1-p, i.e. depolarising parameter
    # λ = 1 - 16p/15, so the IRB gate error is r = (1 - 1/d)(1 - λ) = 0.8 p.
    clean = SiliconNoiseModel(t1_us=1e9, t2_star_us=1e9, charge_noise_amp=0.0, overhauser_sigma=0.0)
    p_dep = 0.04
    res = run_interleaved_rb(
        ExchangeDynamics.target_gate_swap(), [1, 2, 4, 8, 16],
        n_sequences_per_length=6, noise_model=clean, seed=3, interleaved_depolarizing=p_dep,
    )
    expected = 0.8 * p_dep
    assert res["gate_error"] == pytest.approx(expected, rel=0.35)
    assert 0.9 < res["gate_fidelity"] < 1.0
    assert res["decay_p_ref"] > 0.995
