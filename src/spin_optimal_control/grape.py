"""
Differentiable optimal control (GRAPE in a smooth Fourier basis) for silicon
exchange gates.

* Pulse parameterisation: band-limited sine/cosine series with J(0) = J(T) = 0,
  clipped to [0, J_max] (identical map in NumPy and JAX).
* Loss: 1 − F_pro + λ_slew · mean((dJ/dt)²), optionally averaged over a
  quasi-static noise ensemble (ε lever-arm scale × ΔB_z shift) with ``jax.vmap``.
* Optional virtual-Z co-optimisation: four free Z-phases (before/after the
  gate, one per qubit) are optimised jointly with the pulse, matching how
  CZ-type gates are calibrated on hardware.
* Solver: L-BFGS-B with analytic JAX gradients; NumPy central-difference
  fallback when JAX is unavailable.

Units: MHz and ns (``exchange angle = 2π·1e-3·∫J dt``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize

from . import hamiltonian as _ham
from .hamiltonian import SiliconSpinHamiltonian, ExchangeDynamics
from .units import MHZ_NS_TO_RAD

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:  # pragma: no cover
    HAS_JAX = False
    jax = None
    jnp = np


@dataclass
class PulseOptimizationResult:
    """Outcome of a pulse optimisation run."""

    optimal_params: np.ndarray
    j_pulse: np.ndarray               # MHz, shape (n_steps,)
    detuning_pulse: np.ndarray        # mV, shape (n_steps,)
    time_grid: np.ndarray             # ns, midpoints
    dt: float                         # ns
    gate_fidelity: float              # F_pro (with virtual-Z applied if enabled)
    infidelity: float
    iterations: int
    loss_history: List[float]
    synthesized_unitary: np.ndarray   # raw propagator (no virtual-Z)
    target_unitary: np.ndarray
    is_converged: bool
    virtual_z: Optional[np.ndarray] = None   # [a_pre, b_pre, c_post, d_post] rad
    robust_fidelity_mean: Optional[float] = None
    robust_fidelity_min: Optional[float] = None
    exchange_area_mhz_ns: float = 0.0        # ∫J dt
    n_starts: int = 1


class SmoothFourierPulse:
    """
    Band-limited pulse envelope
        u(t) = Σ_k a_k sin(kπt/T) + b_k (1 − cos(2kπt/T)),  k = 1..K
    clipped to [0, J_max]. Both boundary values vanish identically.
    """

    def __init__(self, n_harmonics: int = 6, t_gate_ns: float = 40.0, j_max: float = 50.0):
        self.n_harmonics = int(n_harmonics)
        self.t_gate = float(t_gate_ns)
        self.j_max = float(j_max)

    def num_params(self) -> int:
        return 2 * self.n_harmonics

    def _raw(self, xp, params, t):
        a = params[: self.n_harmonics]
        b = params[self.n_harmonics : 2 * self.n_harmonics]
        k = xp.arange(1, self.n_harmonics + 1, dtype=float)
        s = xp.sin(xp.outer(t, k) * (np.pi / self.t_gate))          # (n_t, K)
        c = 1.0 - xp.cos(xp.outer(t, k) * (2.0 * np.pi / self.t_gate))
        return s @ a + c @ b

    def evaluate(self, params: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        """J(t) on ``time_grid`` (NumPy)."""
        raw = self._raw(np, np.asarray(params, dtype=float), np.asarray(time_grid, dtype=float))
        return np.clip(raw, 0.0, self.j_max)

    def evaluate_jax(self, params, time_grid):
        """J(t) on ``time_grid`` (JAX, differentiable)."""
        raw = self._raw(jnp, params, time_grid)
        return jnp.clip(raw, 0.0, self.j_max)

    # Backwards-compatible aliases (v0.2 API)
    evaluate_pulse_np = evaluate
    evaluate_pulse_jax = evaluate_jax


def _rz_np(theta: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


class GRAPEOptimizer:
    """
    Gradient-based pulse optimiser for silicon exchange gates.

    Parameters
    ----------
    hamiltonian : SiliconSpinHamiltonian
    t_gate_ns : gate duration (ns)
    n_steps : number of piecewise-constant slices; ``time_grid`` holds midpoints
    n_harmonics : Fourier harmonics K (2K pulse parameters)
    j_max : hard amplitude cap (MHz)
    slew_penalty : weight λ on mean((ΔJ/Δt)²) in (MHz/ns)²
    robust : average the fidelity over ``robust_eps_scales × robust_db_shifts``
    local_z_free : co-optimise four virtual-Z phases
    target_infidelity : convergence threshold reported in ``is_converged``
    """

    def __init__(
        self,
        hamiltonian: SiliconSpinHamiltonian,
        t_gate_ns: float = 40.0,
        n_steps: int = 80,
        n_harmonics: int = 6,
        j_max: float = 60.0,
        slew_penalty: float = 1e-6,
        robust: bool = False,
        robust_eps_scales: Sequence[float] = (0.95, 1.0, 1.05),
        robust_db_shifts: Sequence[float] = (-0.5, 0.0, 0.5),
        local_z_free: bool = False,
        target_infidelity: float = 1e-4,
        max_amplitude: Optional[float] = None,   # v0.2 alias for j_max
        slew_rate_penalty: Optional[float] = None,  # v0.2 alias
    ):
        if max_amplitude is not None:
            j_max = max_amplitude
        if slew_rate_penalty is not None:
            slew_penalty = slew_rate_penalty
        self.h = hamiltonian
        self.t_gate = float(t_gate_ns)
        self.n_steps = int(n_steps)
        self.dt = self.t_gate / self.n_steps
        self.time_grid = (np.arange(self.n_steps) + 0.5) * self.dt
        self.j_max = float(j_max)
        self.slew_penalty = float(slew_penalty)
        self.robust = bool(robust)
        self.eps_scales = tuple(float(x) for x in robust_eps_scales)
        self.db_shifts = tuple(float(x) for x in robust_db_shifts)
        self.local_z_free = bool(local_z_free)
        self.target_infidelity = float(target_infidelity)
        self.pulse_basis = SmoothFourierPulse(n_harmonics=n_harmonics, t_gate_ns=self.t_gate, j_max=self.j_max)
        self.max_amp = self.j_max  # v0.2 attribute name

        self._use_jax = HAS_JAX
        if self._use_jax:
            self._build_jax()

    # ----------------------------------------------------------------- #
    def num_params(self) -> int:
        return self.pulse_basis.num_params() + (4 if self.local_z_free else 0)

    def _split(self, params):
        n_p = self.pulse_basis.num_params()
        return params[:n_p], (params[n_p:n_p + 4] if self.local_z_free else None)

    # ----------------------------------------------------------------- #
    # JAX engine
    # ----------------------------------------------------------------- #
    def _build_jax(self):
        t_grid = jnp.asarray(self.time_grid)
        dt = self.dt
        dB = float(self.h.delta_bz)
        basis = self.pulse_basis
        slew_w = self.slew_penalty
        local_z = self.local_z_free
        n_p = basis.num_params()

        scales = jnp.asarray(self.eps_scales) if self.robust else jnp.asarray([1.0])
        shifts = jnp.asarray(self.db_shifts) if self.robust else jnp.asarray([0.0])
        grid_s, grid_d = jnp.meshgrid(scales, shifts, indexing="ij")
        grid_s, grid_d = grid_s.ravel(), grid_d.ravel()

        def rz(theta):
            return jnp.diag(jnp.array([jnp.exp(-0.5j * theta), jnp.exp(0.5j * theta)]))

        def fidelity_one(pulse, scale, db, zs, U_t):
            U = _ham.propagate_unitary_jax(pulse * scale, dt, dB + db)
            if local_z:
                pre = jnp.kron(rz(zs[0]), rz(zs[1]))
                post = jnp.kron(rz(zs[2]), rz(zs[3]))
                U = post @ U @ pre
            ov = jnp.trace(jnp.conj(U_t).T @ U)
            return jnp.real(ov * jnp.conj(ov)) / 16.0

        def loss_fn(params, U_t):
            pulse = basis.evaluate_jax(params[:n_p], t_grid)
            zs = params[n_p:n_p + 4] if local_z else jnp.zeros(4)
            fids = jax.vmap(lambda s, d: fidelity_one(pulse, s, d, zs, U_t))(grid_s, grid_d)
            F = jnp.mean(fids)
            slew = slew_w * jnp.mean((jnp.diff(pulse) / dt) ** 2)
            return 1.0 - F + slew

        self._jax_value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # ----------------------------------------------------------------- #
    # NumPy engine
    # ----------------------------------------------------------------- #
    def _loss_numpy(self, params: np.ndarray, U_t: np.ndarray) -> float:
        p_pulse, zs = self._split(params)
        pulse = self.pulse_basis.evaluate(p_pulse, self.time_grid)
        dyn = ExchangeDynamics(self.h)
        scales = self.eps_scales if self.robust else (1.0,)
        shifts = self.db_shifts if self.robust else (0.0,)
        fids = []
        for s in scales:
            for d in shifts:
                U = dyn.propagate_unitary(pulse * s, self.dt, np.full(self.n_steps, self.h.delta_bz + d))
                if zs is not None:
                    U = np.kron(_rz_np(zs[2]), _rz_np(zs[3])) @ U @ np.kron(_rz_np(zs[0]), _rz_np(zs[1]))
                fids.append(dyn.gate_fidelity(U, U_t))
        slew = self.slew_penalty * float(np.mean((np.diff(pulse) / self.dt) ** 2))
        return 1.0 - float(np.mean(fids)) + slew

    # ----------------------------------------------------------------- #
    def loss_and_grad(self, params: np.ndarray, target_unitary: np.ndarray) -> Tuple[float, np.ndarray]:
        """Loss and gradient at ``params`` (JAX autodiff, or central differences)."""
        params = np.asarray(params, dtype=float)
        if self._use_jax and HAS_JAX:
            val, g = self._jax_value_and_grad(jnp.asarray(params), jnp.asarray(target_unitary, dtype=jnp.complex128))
            return float(val), np.asarray(g, dtype=float)
        base = self._loss_numpy(params, target_unitary)
        grad = np.zeros_like(params)
        eps = 1e-6
        for i in range(len(params)):
            pp = params.copy(); pp[i] += eps
            pm = params.copy(); pm[i] -= eps
            grad[i] = (self._loss_numpy(pp, target_unitary) - self._loss_numpy(pm, target_unitary)) / (2 * eps)
        return base, grad

    # ----------------------------------------------------------------- #
    def _default_seeds(self, seed: int) -> List[np.ndarray]:
        """Half-sine envelopes with exchange areas of 0.25/0.5/0.75 cycles (+ noise)."""
        rng = np.random.default_rng(seed)
        seeds = []
        for cycles in (0.25, 0.5, 0.75):
            p = np.zeros(self.num_params())
            a1 = min(0.9 * self.j_max, cycles * 1e3 * np.pi / (2.0 * self.t_gate))
            p[0] = a1
            if self.pulse_basis.n_harmonics > 1:
                p[1] = 0.05 * a1 * rng.standard_normal()
            seeds.append(p)
        return seeds

    def optimize_pulse(
        self,
        target_unitary: np.ndarray,
        initial_params: Optional[np.ndarray] = None,
        max_iter: int = 200,
        tolerance: float = 1e-12,
        seed: int = 0,
        # v0.2 compatibility (robustness is now configured in the constructor)
        robust_ensemble: Optional[bool] = None,
        n_ensemble_samples: Optional[int] = None,
    ) -> PulseOptimizationResult:
        """Synthesise ``target_unitary``; multi-start when no initial guess is given."""
        if robust_ensemble is not None and robust_ensemble != self.robust:
            self.robust = bool(robust_ensemble)
            if self._use_jax:
                self._build_jax()

        U_t = np.asarray(target_unitary, dtype=np.complex128)
        starts = [np.asarray(initial_params, dtype=float)] if initial_params is not None else self._default_seeds(seed)

        n_pulse = self.pulse_basis.num_params()
        bounds = [(-self.j_max, self.j_max)] * n_pulse + ([(-2 * np.pi, 2 * np.pi)] * 4 if self.local_z_free else [])

        best = None
        for p0 in starts:
            history: List[float] = []

            def objective(p):
                v, g = self.loss_and_grad(p, U_t)
                history.append(v)
                return v, g

            res = scipy.optimize.minimize(
                objective, p0, method="L-BFGS-B", jac=True, bounds=bounds,
                options={"maxiter": int(max_iter), "ftol": float(tolerance), "gtol": 1e-12},
            )
            if best is None or res.fun < best[0].fun:
                best = (res, history)
            if res.fun < self.target_infidelity * 1e-2:
                break

        res, history = best
        return self._package(res.x, history, U_t, len(starts))

    # ----------------------------------------------------------------- #
    def _package(self, params, history, U_t, n_starts) -> PulseOptimizationResult:
        p_pulse, zs = self._split(params)
        j_pulse = self.pulse_basis.evaluate(p_pulse, self.time_grid)
        dyn = ExchangeDynamics(self.h)
        U = dyn.propagate_unitary(j_pulse, self.dt)
        U_eval = U
        if zs is not None:
            U_eval = np.kron(_rz_np(zs[2]), _rz_np(zs[3])) @ U @ np.kron(_rz_np(zs[0]), _rz_np(zs[1]))
        fid = dyn.gate_fidelity(U_eval, U_t)

        rob_mean = rob_min = None
        if self.robust:
            fids = []
            for s in self.eps_scales:
                for d in self.db_shifts:
                    Us = dyn.propagate_unitary(j_pulse * s, self.dt, np.full(self.n_steps, self.h.delta_bz + d))
                    if zs is not None:
                        Us = np.kron(_rz_np(zs[2]), _rz_np(zs[3])) @ Us @ np.kron(_rz_np(zs[0]), _rz_np(zs[1]))
                    fids.append(dyn.gate_fidelity(Us, U_t))
            rob_mean, rob_min = float(np.mean(fids)), float(np.min(fids))

        return PulseOptimizationResult(
            optimal_params=np.asarray(params, dtype=float),
            j_pulse=j_pulse,
            detuning_pulse=self.h.detuning_from_exchange(j_pulse),
            time_grid=self.time_grid.copy(),
            dt=self.dt,
            gate_fidelity=fid,
            infidelity=1.0 - fid,
            iterations=len(history),
            loss_history=list(history),
            synthesized_unitary=U,
            target_unitary=U_t,
            is_converged=bool((1.0 - fid) <= self.target_infidelity),
            virtual_z=None if zs is None else np.asarray(zs, dtype=float),
            robust_fidelity_mean=rob_mean,
            robust_fidelity_min=rob_min,
            exchange_area_mhz_ns=float(np.sum(j_pulse) * self.dt),
            n_starts=n_starts,
        )
