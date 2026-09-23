# spin-optimal-control

Pulse design and noise analysis for exchange gates in silicon double quantum dots.

[![CI](https://github.com/Jaspersands/spin-optimal-control/actions/workflows/ci.yml/badge.svg)](https://github.com/Jaspersands/spin-optimal-control/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

The package optimises exchange pulses with JAX, then asks how those pulses behave under the noise that matters in silicon: 1/f charge noise, nuclear-field fluctuations, relaxation and dephasing at finite temperature, and valley leakage. It also covers smooth pulse windows that make a CZ adiabatic, filter functions computed in the frame of the gate, and two-qubit randomized benchmarking with every Clifford compiled to native pulses.

[Interactive page](https://spin.jaspersands.com/): the same calculations in the browser, with optimisation in a background worker.

## Model

Two electrons, in the frame rotating at the mean Zeeman frequency:

$$
\frac{H(t)}{h} = \frac{J(t)}{4}\,(X_1X_2 + Y_1Y_2 + Z_1Z_2) + \frac{\Delta B_z}{2}\,(Z_1 - Z_2)
$$

- $J(\varepsilon) = J_0 e^{\varepsilon/\varepsilon_0}$ is set by the detuning voltage $\varepsilon(t)$, and $\Delta B_z$ is the Zeeman gradient from a micromagnet or a g-factor difference.
- Frequencies are in MHz and times in ns, and the propagator is $\exp(-i\,2\pi\cdot 10^{-3} H\,dt)$. A constant $J = 10$ MHz for 50 ns is a SWAP, and $\sqrt{\mathrm{SWAP}}$ needs $\int J\,dt = 250$ MHz·ns.
- When $\Delta B_z \gg J$ the same kind of pulse makes a CZ. Its conditional phase is exactly $-2\pi\cdot10^{-3}\int J\,dt$ for any pulse shape, so a CZ needs an area of 500 MHz·ns; the leftover single-qubit Z phases are free virtual-Z frame updates.

## Modules

| Module | Contents |
|---|---|
| `hamiltonian` | Two-electron Hamiltonian, propagation in NumPy and JAX, gate targets, process, average and virtual-Z-free fidelities |
| `grape` | Pulse optimisation on a band-limited Fourier basis with an amplitude cap and slew penalty; optional error-ensemble averaging and virtual-Z co-optimisation |
| `pulse_shaping` | Square, cosine, Tukey and Blackman exchange windows scaled to a CZ; leakage and fidelity from the exact propagator |
| `noise` | 1/f charge noise, nuclear-field shifts, a Lindblad solver with thermal relaxation (detailed balance at the electron temperature), and toggle-frame filter functions with the first-order infidelity they predict |
| `clifford_compiler` | The 24 single-qubit Cliffords as shortest ±X/2, ±Y/2 sequences with virtual Z, and the 11 520 two-qubit Cliffords with minimum-CZ decompositions (576 / 5184 / 5184 / 576 need 0 / 1 / 2 / 3 CZ) |
| `cirq_backend` | Native-gate randomized benchmarking with per-gate noise (`CompiledCliffordRB`), a Clifford-level model for interleaving arbitrary gates, and Cirq decoherence channels |
| `valley` | An 8-level spin ⊗ valley model for leakage during fast pulses |
| `calibration` | A Kalman filter tracking $J_0$ and $\Delta B_z$ drift |
| `awg_export` | Waveforms resampled for JSON, CSV, Qblox and Zurich Instruments |
| `cli` | `spin-control optimize | shape | valley | rb | noise-psd | benchmark` |

## Examples

```python
import numpy as np
from spin_optimal_control import (
    SiliconSpinHamiltonian, ExchangeDynamics, GRAPEOptimizer, AdiabaticCZDesigner,
    SiliconNoiseModel, CompiledCliffordRB, best_z_corrected_cz, export_awg_waveforms,
)

# A √SWAP with a 0.5 MHz residual gradient
h = SiliconSpinHamiltonian(j_0=20.0, delta_bz=0.5)
res = GRAPEOptimizer(h, t_gate_ns=30.0, n_steps=60, n_harmonics=5, j_max=40.0).optimize_pulse(
    ExchangeDynamics.target_gate_sqrt_swap())
print(res.gate_fidelity, res.exchange_area_mhz_ns)          # 0.9975, 250.3

# A CZ under a 20 MHz gradient: pulse shape against leakage
designer = AdiabaticCZDesigner(SiliconSpinHamiltonian(delta_bz=20.0))
for shape in ("square", "cosine"):
    print(shape, designer.calibrate(shape, 100.0).infidelity)  # 5.9e-05, 5.3e-07

# Randomized benchmarking with every Clifford compiled to native pulses
cz = designer.calibrate("cosine", 100.0)
U = best_z_corrected_cz(ExchangeDynamics(designer.h).propagate_unitary(cz.j_pulse, cz.dt_ns))
rb = CompiledCliffordRB(SiliconNoiseModel(t1_us=1000, t2_star_us=20), t_pulse_ns=50, t_cz_ns=100, cz_unitary=U)
print(rb.run([1, 4, 8, 16, 32, 64], n_sequences=30)["clifford_error"])

# Waveforms for the AWG
export_awg_waveforms(res.time_grid, res.j_pulse, res.detuning_pulse, export_format="qblox", file_path="sqrt_swap.json")
```

```bash
spin-control optimize --target cz --dbz 30 --duration 40 --local-z --format qblox --output cz.json
spin-control shape --dbz 20 --durations 50,100,200
spin-control rb --lengths 1,4,8,16,32,64 --t2 20
spin-control benchmark
```

## Checks

The tests assert physics rather than shapes. Some examples:

- Filter-function predictions of gate error under coloured noise agree with Monte-Carlo simulation within its statistical error, and reduce to the scalar formula when the noise commutes with the control.
- At long times the thermal bath relaxes to Boltzmann populations, and T₁ remains the measured 1/e time.
- With depolarising CZ errors, compiled randomized benchmarking reproduces the exact decay $\sum_k w_k f^k$ over the four Clifford classes.
- The conditional phase of any exchange pulse equals $-2\pi\cdot10^{-3}\int J\,dt$ for any gradient.

## Limitations

- Exchange is a single control axis, so averaging the loss over amplitude errors cannot cancel them. The ensemble option mostly helps against gradient shifts.
- Charge noise is quasi-static within one gate in the Monte-Carlo; the filter functions handle its spectrum properly.
- Interleaved benchmarking of a CZ whose error is coherent is only bounded, not accurate, because consecutive CZs add their error amplitudes. The tests check the reference decay instead.

## Install

```bash
pip install -e ".[dev]"          # the core needs NumPy, SciPy and JAX; Cirq and matplotlib are extras
pytest tests/                    # 79 tests
python benchmarks/run_benchmarks.py
```

The notebook [`notebooks/01_exchange_gates_noise_and_benchmarking.ipynb`](notebooks/01_exchange_gates_noise_and_benchmarking.ipynb) goes through each part with executed outputs. Changes between versions are in [CHANGELOG.md](CHANGELOG.md).

Apache-2.0 · Jasper Sands
