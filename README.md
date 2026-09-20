# spin-optimal-control

**Differentiable optimal control for silicon spin exchange gates under 1/f charge noise and Overhauser drift.**

[![CI](https://github.com/Jaspersands/spin-optimal-control/actions/workflows/ci.yml/badge.svg)](https://github.com/Jaspersands/spin-optimal-control/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/autodiff-JAX-red.svg)](https://github.com/google/jax)
[![Cirq](https://img.shields.io/badge/benchmarking-Cirq-teal.svg)](https://quantumai.google/cirq)

JAX-differentiable GRAPE pulse engineering in a band-limited Fourier basis, exact Lindblad open-system simulation, first-order filter functions, a genuine two-qubit Clifford randomized-benchmarking suite in Cirq, valley-leakage modelling, Bayesian drift tracking and AWG waveform export — all in one consistent unit system (MHz / ns).

**[▶ Interactive demo](web/index.html)** — the browser page runs a real 4×4 propagator, gradient optimiser, 1/f noise Monte-Carlo and RB simulation in JavaScript; every number on it is computed, not illustrative.

---

## Physics

Two electrons in a silicon double quantum dot (DQD), in the frame rotating at the mean Zeeman frequency:

$$
\frac{H(t)}{h} \;=\; \frac{J(t)}{4}\,\big(X_1X_2 + Y_1Y_2 + Z_1Z_2\big) \;+\; \frac{\Delta B_z}{2}\,\big(Z_1 - Z_2\big)
$$

* $J(\varepsilon) = J_0\,e^{\varepsilon/\varepsilon_0}$ is the Heisenberg exchange controlled by the detuning voltage $\varepsilon(t)$.
* $\Delta B_z$ is the Zeeman gradient (micromagnet or g-factor difference); the homogeneous field $B_0$ commutes with everything and is dropped in the rotating frame (`rotating_frame=False` keeps it).
* **Units:** all frequencies are linear frequencies in MHz, times in ns; the propagator is $U=\exp(-i\,2\pi\cdot10^{-3}\,H\,dt)$. A constant $J=10$ MHz for 50 ns therefore accumulates an exchange angle of $\pi$ (a SWAP); $\sqrt{\mathrm{SWAP}}$ needs $\int J\,dt = 250$ MHz·ns.
* With $J \gg \Delta B_z$ the exchange pulse gives SWAP-family gates; with $\Delta B_z \gg J$ the same $\int J\,dt = 500$ MHz·ns produces a CZ up to single-qubit Z phases, which are free *virtual-Z* frame updates on hardware. `gate_fidelity_local_z_free` and the GRAPE `local_z_free=True` option account for this.

## What the package does

| Module | Capability |
|---|---|
| `hamiltonian` | `SiliconSpinHamiltonian`, `ExchangeDynamics` — rotating/lab frame, NumPy + JAX (`lax.scan`) propagation, singlet–triplet populations, gate targets (√SWAP, SWAP, SWAP¼, CZ, `exchange_gate(θ)`), process / average / local-Z-free fidelities |
| `grape` | `GRAPEOptimizer` — band-limited Fourier pulses with $J(0)=J(T)=0$ and hard amplitude cap, analytic gradients through matrix exponentials, slew-rate penalty on $dJ/dt$, quasi-static noise ensemble via `jax.vmap`, virtual-Z co-optimisation, multi-start seeds sized by the physical exchange area, NumPy fallback |
| `noise` | 1/f$^\alpha$ charge noise (spectral synthesis or OU two-level fluctuators), Overhauser shifts, exact 16×16 Liouvillian Lindblad solver with $T_1$ / $T_2^*$, channel process fidelity, filter functions |
| `cirq_backend` | `SiliconExchangeGate`, silicon $T_1$/$T_\phi$ channels, the full **11 520-element two-qubit Clifford group**, standard and interleaved RB with exponential fits |
| `valley` | 8-dimensional spin ⊗ effective-valley model; leakage out of the computational valley during fast pulses |
| `drag` | Derivative (DRAG-style) adiabatic corrections for detuning ramps |
| `calibration` | 2-D Kalman filter tracking $(J_0,\Delta B_z)$ drift from Ramsey / exchange oscillation experiments |
| `awg_export` | Resampled waveform export: JSON, CSV, Qblox, Zurich Instruments |
| `cli` | `spin-control optimize | valley | rb | noise-psd | benchmark` (all with `--json`) |

## Quickstart

```python
import numpy as np
from spin_optimal_control import (
    SiliconSpinHamiltonian, ExchangeDynamics, GRAPEOptimizer,
    SiliconNoiseModel, run_interleaved_rb, export_awg_waveforms,
)

# 1. √SWAP with a 0.5 MHz residual gradient (no micromagnet)
h = SiliconSpinHamiltonian(j_0=20.0, epsilon_0=1.0, delta_bz=0.5)
opt = GRAPEOptimizer(h, t_gate_ns=30.0, n_steps=60, n_harmonics=5, j_max=40.0)
res = opt.optimize_pulse(ExchangeDynamics.target_gate_sqrt_swap())
print(f"√SWAP fidelity {res.gate_fidelity:.5f}, ∫J dt = {res.exchange_area_mhz_ns:.1f} MHz·ns")  # 0.9975, 250.3

# 2. CZ under a 30 MHz micromagnet gradient, virtual-Z phases co-optimised
h_cz = SiliconSpinHamiltonian(j_0=20.0, delta_bz=30.0)
res_cz = GRAPEOptimizer(h_cz, t_gate_ns=40.0, n_steps=80, n_harmonics=6, j_max=40.0,
                        local_z_free=True).optimize_pulse(ExchangeDynamics.target_gate_cz())
print(f"CZ fidelity {res_cz.gate_fidelity:.6f}, virtual-Z = {np.round(res_cz.virtual_z, 3)}")   # 1.000000

# 3. Open-system fidelity and two-qubit Clifford interleaved RB in Cirq
noise = SiliconNoiseModel(t1_us=800.0, t2_star_us=25.0, charge_noise_amp=0.04, overhauser_sigma=0.3)
f_lindblad = noise.channel_process_fidelity(res.j_pulse, res.dt, res.target_unitary, h.get_hamiltonian_matrix)
rb = run_interleaved_rb(res.synthesized_unitary, [1, 2, 4, 8, 16, 32, 64], n_sequences_per_length=10, noise_model=noise)
print(f"Lindblad F_pro {f_lindblad:.4f}, IRB gate error {rb['gate_error']:.2e}")

# 4. Ship it to the AWG
export_awg_waveforms(res.time_grid, res.j_pulse, res.detuning_pulse, sample_rate_gsps=1.0,
                     export_format="qblox", file_path="sqrt_swap_qblox.json")
```

Command line:

```bash
spin-control optimize --target cz --dbz 30 --duration 40 --local-z --format qblox --output cz.json
spin-control rb --lengths 1,2,4,8,16,32,64 --t2 25 --json
spin-control benchmark
```

## Robustness: what exchange-only control can and cannot do

The `robust=True` option averages the fidelity over a grid of amplitude scalings and $\Delta B_z$ shifts with analytic gradients. Because the exchange term is a *single-axis* control, amplitude (lever-arm) errors are not correctable by pulse shaping alone — every $J(t)$ commutes with every other — and gains against $\Delta B_z$ shifts are limited by the amplitude cap. The machinery is there and correct; the README makes no claim beyond what the tests show (`test_robust_ensemble_improves_noisy_fidelity`).

## Install & test

```bash
pip install -e ".[dev]"
pytest -v tests/                     # 65 tests
python benchmarks/run_benchmarks.py  # GRAPE → noise MC → Lindblad → IRB → valley
```

JAX is optional (NumPy finite-difference fallback); Cirq is needed only for `cirq_backend`.

## Tutorial

[`notebooks/01_silicon_grape_drag_valley_simulation.ipynb`](notebooks/01_silicon_grape_drag_valley_simulation.ipynb) — spectrum vs $J$, √SWAP and CZ synthesis, noise Monte-Carlo, Lindblad fidelity vs $T_2^*$, two-qubit RB, valley leakage, AWG export (executed outputs included).

## Changelog

See [CHANGELOG.md](CHANGELOG.md). v0.3.0 is a correctness release: v0.2 had no MHz→rad conversion in the propagator (so the documented defaults could not reach √SWAP), Lindblad rates in the wrong unit, a factor-two error in the Cirq phase-damping parameter and an "RB" that sampled only 1Q⊗1Q Cliffords.

## License

Apache-2.0 — Jasper Sands.
