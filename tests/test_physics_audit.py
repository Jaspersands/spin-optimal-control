"""
Regression tests for the v0.3 adversarial audit: toggle-frame filter functions
(validated against Monte-Carlo), thermal Lindblad bath, and native-gate
Clifford RB.
"""

import numpy as np
import pytest
import scipy.linalg

from spin_optimal_control import SiliconSpinHamiltonian
from spin_optimal_control.noise import (
    SiliconNoiseModel,
    exchange_gate_noise_operators,
    filter_function,
    gate_filter_functions,
    infidelity_from_filter_function,
)
from spin_optimal_control.pulse_shaping import AdiabaticCZDesigner
from spin_optimal_control.units import MHZ_NS_TO_RAD

cirq = pytest.importorskip("cirq")
from spin_optimal_control.cirq_backend import CompiledCliffordRB, best_z_corrected_cz  # noqa: E402
from spin_optimal_control.clifford_compiler import (  # noqa: E402
    compiled_unitary,
    lookup_two_qubit_clifford,
    one_qubit_cliffords,
    two_qubit_cliffords,
    unitary_key,
)


# --------------------------------------------------------------------------- #
# filter functions
# --------------------------------------------------------------------------- #
def _cz_setup(n_steps=160):
    h = SiliconSpinHamiltonian(delta_bz=20.0)
    cz = AdiabaticCZDesigner(h, n_steps=n_steps).calibrate("cosine", 100.0)
    Hs = [h.get_hamiltonian_matrix(float(x)) for x in cz.j_pulse]
    return h, cz, Hs


def _propagate(Hs, extra, dt):
    U = np.eye(4, dtype=complex)
    for k in range(len(Hs)):
        U = scipy.linalg.expm(-1j * MHZ_NS_TO_RAD * (Hs[k] + extra[k]) * dt) @ U
    return U


def test_filter_function_reduces_to_scalar_spectrum_when_noise_commutes():
    """ΔBz = 0: exchange noise commutes with the control, so F = ‖N‖²·|∫J e^{iωt}|² (up to slice sinc)."""
    h0 = SiliconSpinHamiltonian(delta_bz=0.0)
    j = 20.0 * np.sin(np.pi * (np.arange(200) + 0.5) / 200)
    dt = 0.25
    om = np.linspace(0.0, 0.5, 30)
    Fg = gate_filter_functions([h0.get_hamiltonian_matrix(float(x)) for x in j], dt, exchange_gate_noise_operators(j), om)[0]
    Fs = filter_function(j, dt, om) * 0.75 * MHZ_NS_TO_RAD**2        # Σ_β |Tr(C_β N)|² = 3/4 for N = σ·σ/4
    sinc2 = np.sinc(om * dt / (2 * np.pi)) ** 2
    assert np.allclose(Fg, Fs * sinc2, rtol=1e-6)


def test_quasistatic_limit_matches_exact_perturbation():
    """S(ω) = 2πσ²δ(ω) ⇒ 1 − F_e = σ² F(0)/d; compare with the exact average over static offsets."""
    h, cz, Hs = _cz_setup()
    ops = exchange_gate_noise_operators(cz.j_pulse, "zeeman1")
    F0 = gate_filter_functions(Hs, cz.dt_ns, ops, np.array([0.0]))[0, 0]
    sigma = 0.05
    N, a = ops[0]
    U0 = _propagate(Hs, [0] * len(Hs), cz.dt_ns)
    xs = np.array([-1.0, 1.0]) * sigma                    # ±σ is exact for a quadratic response
    inf = [1 - abs(np.trace(U0.conj().T @ _propagate(Hs, [x * N] * len(Hs), cz.dt_ns))) ** 2 / 16 for x in xs]
    assert np.mean(inf) == pytest.approx(sigma**2 * F0 / 4, rel=1e-3)


def test_colored_noise_infidelity_matches_monte_carlo():
    """Ornstein–Uhlenbeck frequency noise on one electron during a shaped CZ (dynamic toggle frame)."""
    h, cz, Hs = _cz_setup()
    dt = cz.dt_ns
    sig, tc = 0.3, 5.0
    ops = exchange_gate_noise_operators(cz.j_pulse, "zeeman1")
    om = np.concatenate([[0.0], np.logspace(-4, np.log10(np.pi / dt), 2500)])
    F = gate_filter_functions(Hs, dt, ops, om)[0]
    pred = infidelity_from_filter_function(om, F, 2 * sig**2 * 2 * tc / (1 + (om * tc) ** 2))
    rng = np.random.default_rng(7)
    U0 = _propagate(Hs, [0] * len(Hs), dt)
    N = ops[0][0]
    decay = np.exp(-dt / tc)
    vals = []
    for _ in range(200):
        x = np.empty(len(Hs)); x[0] = rng.normal(0, sig)
        for k in range(1, len(Hs)):
            x[k] = decay * x[k - 1] + np.sqrt(1 - decay**2) * sig * rng.normal()
        U = _propagate(Hs, [xk * N for xk in x], dt)
        vals.append(1 - abs(np.trace(U0.conj().T @ U)) ** 2 / 16)
    mc, err = np.mean(vals), np.std(vals) / np.sqrt(len(vals))
    assert abs(pred - mc) < 3 * err + 0.05 * mc


# --------------------------------------------------------------------------- #
# thermal bath
# --------------------------------------------------------------------------- #
def test_thermal_bath_relaxes_to_boltzmann_populations():
    nm = SiliconNoiseModel(t1_us=1.0, t2_star_us=1.0, charge_noise_amp=0.0, overhauser_sigma=0.0,
                           temperature_mk=100.0, larmor_frequency_mhz=1000.0)
    p_up = nm.excited_state_population()
    x = 6.62607015e-34 * 1e9 / (1.380649e-23 * 0.1)
    assert p_up == pytest.approx(1 / (1 + np.exp(x)), rel=1e-9)    # ≈ 0.38 at 1 GHz, 100 mK
    S = nm.idle_superoperator(50_000.0)                            # 50 T1
    rho = (S @ np.eye(4, dtype=complex).reshape(-1, order="F") / 4).reshape(4, 4, order="F")
    single = np.array([p_up, 1 - p_up])
    assert np.allclose(np.real(np.diag(rho)), np.kron(single, single), atol=1e-9)
    # zero temperature relaxes fully to |↓↓⟩ = |11⟩, and T1 is still the measured 1/e time
    nm0 = SiliconNoiseModel(t1_us=1.0, t2_star_us=1e9, charge_noise_amp=0.0, overhauser_sigma=0.0)
    rho0 = np.zeros((4, 4), complex); rho0[0, 0] = 1
    r = (nm0.idle_superoperator(1000.0) @ rho0.reshape(-1, order="F")).reshape(4, 4, order="F")
    assert np.real(r[0, 0]) == pytest.approx(np.exp(-2.0), rel=1e-9)   # both electrons: e^{−t/T1} each


def test_cirq_damping_matches_lindblad_direction_and_temperature():
    nm = SiliconNoiseModel(t1_us=0.1, t2_star_us=1e9, charge_noise_amp=0.0, overhauser_sigma=0.0,
                           temperature_mk=100.0, larmor_frequency_mhz=1000.0)
    from spin_optimal_control.cirq_backend import CirqSiliconSimulator
    q = cirq.LineQubit(0)
    noisy = CirqSiliconSimulator(nm).create_noisy_circuit(cirq.Circuit(cirq.I(q)), gate_duration_ns=5000.0)
    rho = cirq.DensityMatrixSimulator().simulate(noisy).final_density_matrix
    assert abs(np.real(rho[0, 0]) - nm.excited_state_population()) < 1e-4


# --------------------------------------------------------------------------- #
# compiled Clifford RB
# --------------------------------------------------------------------------- #
def test_native_clifford_tables():
    c1 = one_qubit_cliffords()
    assert len(c1) == 24 and np.mean([c.n_pulses for c in c1]) == 1.0
    c2 = two_qubit_cliffords()
    counts = np.bincount([c.n_cz for c in c2])
    assert counts.tolist() == [576, 5184, 5184, 576]               # 1.5 CZ per Clifford
    for i in range(0, 11520, 97):
        assert unitary_key(compiled_unitary(c2[i])) == unitary_key(c2[i].unitary)
        assert lookup_two_qubit_clifford(c2[i].unitary) == i


def test_compiled_circuit_is_native_and_exact():
    rb = CompiledCliffordRB()
    for i in (0, 700, 6000, 11519):
        c = rb.circuit(i)
        for op in c.all_operations():
            assert op.gate in (cirq.CZ, cirq.S) or isinstance(op.gate, (cirq.Rx, cirq.Ry))
        U = c.unitary(qubits_that_should_be_present=cirq.LineQubit.range(2))
        assert cirq.equal_up_to_global_phase(U, rb.c2[i].unitary, atol=1e-8)


def test_compiled_rb_matches_exact_depolarizing_decay():
    """CZ-only depolarising error: p_RB = Σ_k w_k f^k with the class weights w = (576, 5184, 5184, 576)/11520."""
    p = 0.02
    f = 1 - 16 * p / 15
    rb = CompiledCliffordRB(p_cz=p)
    res = rb.run([1, 4, 8, 16, 32, 48], n_sequences=30, seed=1)
    exact = (576 + 5184 * f + 5184 * f**2 + 576 * f**3) / 11520
    assert res["decay_p"] == pytest.approx(exact, abs=2e-3)
    ir = rb.run_interleaved_cz([1, 4, 8, 16, 32], n_sequences=30, seed=2)
    assert ir["cz_error"] == pytest.approx(0.75 * (1 - f), rel=0.25)


def test_compiled_rb_sees_coherent_leakage_of_a_square_cz():
    """
    The CZ's only error is coherent |↑↓⟩↔|↓↑⟩ leakage. Reference RB must price it at
    ≈ 1.5 CZ per Clifford, and the cosine window must beat the square pulse.
    (Interleaved RB is deliberately not asserted here: for coherent errors it is
    only bounded, not accurate — consecutive CZs add their error amplitudes.)
    """
    h = SiliconSpinHamiltonian(delta_bz=20.0)
    d = AdiabaticCZDesigner(h, n_steps=200)
    from spin_optimal_control import ExchangeDynamics
    dyn = ExchangeDynamics(h)
    epc = {}
    for shape in ("square", "cosine"):
        cz = d.calibrate(shape, 100.0)
        U = best_z_corrected_cz(dyn.propagate_unitary(cz.j_pulse, cz.dt_ns))
        res = CompiledCliffordRB(cz_unitary=U).run([1, 100, 300, 600, 1000], n_sequences=15, seed=4)
        epc[shape] = (res["clifford_error"], cz.infidelity)
    sq_epc, sq_inf = epc["square"]
    assert 0.5 * 1.5 * sq_inf < sq_epc < 3.0 * 1.5 * sq_inf
    assert epc["cosine"][0] < sq_epc / 3
