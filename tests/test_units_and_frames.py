"""
Tests for the physical unit convention (MHz / ns), the rotating frame, and
the fidelity measures used throughout spin_optimal_control.
"""

import numpy as np
import pytest
import scipy.linalg

from spin_optimal_control.units import MHZ_NS_TO_RAD, phase
from spin_optimal_control.hamiltonian import (
    SiliconSpinHamiltonian,
    ExchangeDynamics,
    HEISENBERG_EXCHANGE,
    I_4,
)


def _rz(theta: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def test_unit_factor_value():
    # 1 MHz for 1 ns accumulates 2*pi*1e-3 rad of phase.
    assert np.isclose(MHZ_NS_TO_RAD, 2.0 * np.pi * 1e-3)
    assert np.isclose(phase(10.0, 50.0), 2.0 * np.pi * 1e-3 * 500.0)


def test_constant_exchange_pulse_gives_exact_exchange_angle():
    # A constant J = 10 MHz pulse for 50 ns must implement the exchange gate
    # with angle 2*pi*10e-3*50 = pi rad, i.e. a SWAP (up to global phase).
    h = SiliconSpinHamiltonian(j_0=10.0, delta_bz=0.0, b_0=0.0)
    dyn = ExchangeDynamics(h)
    j_pulse = np.full(200, 10.0)
    dt = 50.0 / 200
    u = dyn.propagate_unitary(j_pulse, dt)

    angle = phase(10.0, 50.0)
    assert np.isclose(angle, np.pi)
    u_expected = ExchangeDynamics.exchange_gate(angle)
    assert ExchangeDynamics.gate_fidelity(u, u_expected) > 1.0 - 1e-10
    assert ExchangeDynamics.gate_fidelity(u, ExchangeDynamics.target_gate_swap()) > 1.0 - 1e-10


def test_sqrt_swap_at_quarter_cycle():
    # pi/2 exchange angle -> sqrt(SWAP): J*T = 0.25 cycles.
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=0.0, b_0=0.0)
    dyn = ExchangeDynamics(h)
    t_gate = 0.25 / 20.0 * 1e3  # ns
    j_pulse = np.full(100, 20.0)
    u = dyn.propagate_unitary(j_pulse, t_gate / 100)
    assert ExchangeDynamics.gate_fidelity(u, ExchangeDynamics.target_gate_sqrt_swap()) > 1.0 - 1e-10


def test_rotating_frame_removes_homogeneous_zeeman_term():
    h_a = SiliconSpinHamiltonian(j_0=20.0, delta_bz=5.0, b_0=0.0, rotating_frame=True)
    h_b = SiliconSpinHamiltonian(j_0=20.0, delta_bz=5.0, b_0=1234.5, rotating_frame=True)
    j_pulse = 15.0 * np.sin(np.pi * np.linspace(0, 1, 40))
    u_a = ExchangeDynamics(h_a).propagate_unitary(j_pulse, 0.5)
    u_b = ExchangeDynamics(h_b).propagate_unitary(j_pulse, 0.5)
    assert np.allclose(u_a, u_b, atol=1e-12)

    # In the lab frame the B0 term is present and changes the unitary.
    h_lab = SiliconSpinHamiltonian(j_0=20.0, delta_bz=5.0, b_0=1234.5, rotating_frame=False)
    u_lab = ExchangeDynamics(h_lab).propagate_unitary(j_pulse, 0.5)
    assert not np.allclose(u_a, u_lab, atol=1e-6)
    # ...but only by a Z_SUM rotation that commutes with everything, so the
    # local-Z-free fidelity between the two is still 1.
    f_z, _ = ExchangeDynamics.gate_fidelity_local_z_free(u_lab, u_a)
    assert f_z > 1.0 - 1e-8


def test_hamiltonian_matrix_frames():
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=5.0, b_0=300.0, rotating_frame=True)
    m_rot = h.get_hamiltonian_matrix(20.0)
    m_lab = h.get_hamiltonian_matrix(20.0, lab_frame=True)
    assert np.allclose(m_rot, m_rot.conj().T)
    assert np.allclose(m_lab, m_lab.conj().T)
    diff = m_lab - m_rot
    # difference is (B0/2)(Z1+Z2) = diag(B0, 0, 0, -B0)
    assert np.allclose(np.diag(diff), [300.0, 0.0, 0.0, -300.0])


def test_local_z_free_fidelity_recovers_cz_dressed_with_virtual_z():
    cz = ExchangeDynamics.target_gate_cz()
    pre = np.kron(_rz(1.1), _rz(0.2))
    post = np.kron(_rz(0.3), _rz(-0.7))
    dressed = post @ cz @ pre * np.exp(0.4j)
    f_raw = ExchangeDynamics.gate_fidelity(dressed, cz)
    f_free, angles = ExchangeDynamics.gate_fidelity_local_z_free(dressed, cz)
    assert f_raw < 0.9
    assert f_free > 1.0 - 1e-8
    assert angles.shape == (4,)


def test_local_z_free_fidelity_cannot_fix_non_z_error():
    # An X rotation is not a virtual-Z; the local-Z-free fidelity must stay < 1.
    cz = ExchangeDynamics.target_gate_cz()
    rx = scipy.linalg.expm(-0.5j * 0.8 * np.array([[0, 1], [1, 0]]))
    dressed = np.kron(rx, np.eye(2)) @ cz
    f_free, _ = ExchangeDynamics.gate_fidelity_local_z_free(dressed, cz)
    assert f_free < 0.95


def test_average_gate_fidelity_relation():
    # F_avg = (d * F_pro + 1) / (d + 1); identity gives 1, orthogonal gives 1/(d+1)... test bounds
    assert np.isclose(ExchangeDynamics.average_gate_fidelity(I_4, I_4), 1.0)
    swap = ExchangeDynamics.target_gate_swap()
    f_pro = ExchangeDynamics.gate_fidelity(I_4, swap)
    f_avg = ExchangeDynamics.average_gate_fidelity(I_4, swap)
    assert np.isclose(f_avg, (4.0 * f_pro + 1.0) / 5.0)


def test_exchange_gate_family():
    # exchange_gate(theta) must be unitary and reduce to the named targets.
    for theta, target in [
        (np.pi, ExchangeDynamics.target_gate_swap()),
        (np.pi / 2, ExchangeDynamics.target_gate_sqrt_swap()),
        (np.pi / 4, ExchangeDynamics.target_gate_fourth_swap()),
        (0.0, I_4),
    ]:
        u = ExchangeDynamics.exchange_gate(theta)
        assert np.allclose(u.conj().T @ u, I_4, atol=1e-12)
        assert ExchangeDynamics.gate_fidelity(u, target) > 1.0 - 1e-10
