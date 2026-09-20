"""
Tests for the GRAPE / smooth-Fourier pulse optimiser.

All numbers use the MHz / ns convention. Exchange angles are
2π·1e-3·∫J dt, so √SWAP needs ∫J dt = 250 MHz·ns.
"""

import numpy as np
import pytest

from spin_optimal_control.hamiltonian import SiliconSpinHamiltonian, ExchangeDynamics
from spin_optimal_control.grape import GRAPEOptimizer, SmoothFourierPulse, PulseOptimizationResult
from spin_optimal_control import grape as grape_module


def test_smooth_pulse_boundary_conditions_and_clip():
    basis = SmoothFourierPulse(n_harmonics=4, t_gate_ns=20.0, j_max=25.0)
    t = np.linspace(0.0, 20.0, 101)
    params = np.array([40.0, 5.0, -2.0, 1.0, 4.0, -1.0, 2.0, 0.5])
    pulse = basis.evaluate(params, t)
    assert np.isclose(pulse[0], 0.0, atol=1e-9)
    assert np.isclose(pulse[-1], 0.0, atol=1e-9)
    assert np.all(pulse >= 0.0)
    assert np.all(pulse <= 25.0 + 1e-12)
    assert pulse.max() == pytest.approx(25.0)  # a1=40 saturates j_max

    # JAX and NumPy pulse maps must be numerically identical.
    if grape_module.HAS_JAX:
        import jax.numpy as jnp
        pulse_jax = np.asarray(basis.evaluate_jax(jnp.asarray(params), jnp.asarray(t)))
        assert np.allclose(pulse, pulse_jax, atol=1e-12)


def test_midpoint_time_grid():
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=0.0)
    opt = GRAPEOptimizer(h, t_gate_ns=30.0, n_steps=60, n_harmonics=4)
    assert opt.dt == pytest.approx(0.5)
    assert opt.time_grid[0] == pytest.approx(0.25)
    assert opt.time_grid[-1] == pytest.approx(29.75)
    assert len(opt.time_grid) == 60


def test_grape_sqrt_swap_without_gradient_reaches_machine_precision():
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=0.0, b_0=0.0)
    opt = GRAPEOptimizer(h, t_gate_ns=20.0, n_steps=50, n_harmonics=4, j_max=40.0)
    res = opt.optimize_pulse(ExchangeDynamics.target_gate_sqrt_swap(), max_iter=300)
    assert isinstance(res, PulseOptimizationResult)
    assert res.gate_fidelity > 1.0 - 1e-6
    assert res.is_converged
    # Physically: the exchange angle must be pi/2 -> ∫J dt = 250 MHz·ns (mod 1000).
    area = float(np.sum(res.j_pulse) * res.dt)
    assert np.isclose((area - 250.0) % 1000.0, 0.0, atol=2.0) or np.isclose((area - 250.0) % 1000.0, 1000.0, atol=2.0)
    assert res.j_pulse.shape == (50,)
    assert res.detuning_pulse.shape == (50,)
    assert len(res.loss_history) == res.iterations


def test_grape_cz_under_zeeman_gradient_with_virtual_z():
    # CZ under a 30 MHz gradient: the raw unitary carries single-qubit Z phases
    # which are absorbed into the two virtual-Z parameters.
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=30.0, b_0=0.0)
    opt = GRAPEOptimizer(h, t_gate_ns=40.0, n_steps=80, n_harmonics=6, j_max=40.0, local_z_free=True)
    res = opt.optimize_pulse(ExchangeDynamics.target_gate_cz(), max_iter=400)
    assert res.virtual_z is not None and res.virtual_z.shape == (4,)
    assert res.gate_fidelity > 0.999
    # The reported fidelity must agree with the independent local-Z-free metric.
    f_check, _ = ExchangeDynamics.gate_fidelity_local_z_free(res.synthesized_unitary, res.target_unitary)
    assert f_check >= res.gate_fidelity - 1e-6


@pytest.mark.skipif(not grape_module.HAS_JAX, reason="JAX not installed")
def test_jax_gradient_matches_finite_difference():
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=10.0, b_0=0.0)
    opt = GRAPEOptimizer(h, t_gate_ns=25.0, n_steps=30, n_harmonics=3, slew_penalty=1e-5)
    u_t = ExchangeDynamics.target_gate_cz()
    rng = np.random.default_rng(1)
    p = rng.uniform(-5, 15, opt.num_params())
    val, grad = opt.loss_and_grad(p, u_t)
    fd = np.zeros_like(p)
    eps = 1e-6
    for i in range(len(p)):
        pp = p.copy(); pp[i] += eps
        pm = p.copy(); pm[i] -= eps
        fd[i] = (opt.loss_and_grad(pp, u_t)[0] - opt.loss_and_grad(pm, u_t)[0]) / (2 * eps)
    assert np.allclose(grad, fd, rtol=1e-4, atol=1e-7)


def test_numpy_fallback_matches_jax_loss(monkeypatch):
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=10.0, b_0=0.0)
    u_t = ExchangeDynamics.target_gate_sqrt_swap()
    opt_jax = GRAPEOptimizer(h, t_gate_ns=25.0, n_steps=20, n_harmonics=3)
    p = np.array([12.0, -3.0, 1.0, 4.0, 0.5, -0.5])
    val_ref, grad_ref = opt_jax.loss_and_grad(p, u_t)

    monkeypatch.setattr(grape_module, "HAS_JAX", False)
    opt_np = GRAPEOptimizer(h, t_gate_ns=25.0, n_steps=20, n_harmonics=3)
    val_np, grad_np = opt_np.loss_and_grad(p, u_t)
    assert np.isclose(val_np, val_ref, atol=1e-10)
    assert np.allclose(grad_np, grad_ref, rtol=1e-4, atol=1e-6)


def test_robust_ensemble_improves_noisy_fidelity():
    # Optimise the same gate nominally and robustly; evaluate both under the
    # same quasi-static (ε-scale, ΔBz-shift) perturbations. The robust pulse
    # must not be worse on average. Exchange-only control is single-axis, so
    # gains are modest; the point is the ensemble machinery and the metric.
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=0.5, b_0=0.0)
    u_t = ExchangeDynamics.target_gate_sqrt_swap()
    nominal = GRAPEOptimizer(h, t_gate_ns=30.0, n_steps=40, n_harmonics=4, j_max=40.0)
    robust = GRAPEOptimizer(
        h, t_gate_ns=30.0, n_steps=40, n_harmonics=4, j_max=40.0,
        robust=True, robust_eps_scales=(0.95, 1.0, 1.05), robust_db_shifts=(-0.3, 0.0, 0.3),
    )
    r_nom = nominal.optimize_pulse(u_t, max_iter=200)
    r_rob = robust.optimize_pulse(u_t, max_iter=200)
    assert r_rob.robust_fidelity_mean is not None
    assert r_rob.robust_fidelity_mean > 0.99
    assert r_rob.j_pulse[0] == pytest.approx(0.0, abs=1e-9) or r_rob.j_pulse[0] < 1.0

    dyn = ExchangeDynamics(h)
    def mean_fid(res):
        fids = []
        for s in (0.95, 1.0, 1.05):
            for db in (-0.3, 0.0, 0.3):
                u = dyn.propagate_unitary(res.j_pulse * s, res.dt, np.full(len(res.j_pulse), h.delta_bz + db))
                fids.append(dyn.gate_fidelity(u, u_t))
        return float(np.mean(fids))
    assert mean_fid(r_rob) >= mean_fid(r_nom) - 1e-4


def test_slew_penalty_reduces_pulse_roughness():
    h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=0.0, b_0=0.0)
    u_t = ExchangeDynamics.target_gate_sqrt_swap()
    smooth = GRAPEOptimizer(h, t_gate_ns=20.0, n_steps=40, n_harmonics=6, slew_penalty=1e-3)
    rough = GRAPEOptimizer(h, t_gate_ns=20.0, n_steps=40, n_harmonics=6, slew_penalty=0.0)
    p0 = np.zeros(smooth.num_params()); p0[0] = 15.0; p0[3] = 12.0; p0[5] = -9.0
    r_s = smooth.optimize_pulse(u_t, initial_params=p0, max_iter=200)
    r_r = rough.optimize_pulse(u_t, initial_params=p0, max_iter=200)
    def roughness(res):
        return float(np.mean(np.diff(res.j_pulse) ** 2))
    assert roughness(r_s) <= roughness(r_r) + 1e-9
    assert r_s.gate_fidelity > 0.999
