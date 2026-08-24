# Differentiable Optimal Control for Silicon Spin Exchange Gates

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX-Accelerated](https://img.shields.io/badge/Autodiff-JAX-red.svg)](https://github.com/google/jax)
[![Cirq Integration](https://img.shields.io/badge/Framework-Cirq%20%7C%20PennyLane-teal.svg)](https://quantumai.google/cirq)

> **JAX-differentiable pulse engineering (GRAPE / smooth Fourier basis), Valley Splitting dynamics, analytical DRAG corrections, Bayesian active feedback calibration, and Cirq Randomized Benchmarking for silicon spin qubits under $1/f^\alpha$ pink charge noise and Overhauser drift.**

---

## 🌟 Key Features (v0.2.0)

- **Realistic Silicon Quantum Dot Physics**:
  - Full two-electron spin exchange Hamiltonian $H(t) = \frac{J(t)}{4}(\vec{\sigma}_1 \cdot \vec{\sigma}_2) + \frac{\Delta B_z(t)}{2}(\sigma_1^z - \sigma_2^z) + \frac{B_0}{2}(\sigma_1^z + \sigma_2^z)$.
  - Detuning exponential voltage relation $J(\epsilon) = J_0 \exp(\epsilon / \epsilon_0)$.
  - **Valley Splitting & Multi-Valley Leakage (`SiliconValleyModel`)**: 8x8 valley-spin Hamiltonian modeling conduction band valley splitting $E_v$ and inter-valley spin-orbit coupling.
- **Physical Noise Modeling**:
  - $1/f^\alpha$ pink charge noise via multi-trap Lorentzian Two-Level Fluctuators (TLF) and spectral synthesis.
  - Quasistatic Overhauser field fluctuations ($\delta B_n \sim \mathcal{N}(0, \sigma_N^2)$) from residual $^{29}\text{Si}$ nuclear spins.
  - Open-system Lindblad master equation with finite $T_1$ relaxation and $T_2^*$ dephasing.
- **Differentiable Optimal Control & Analytical Corrections**:
  - Smooth Fourier / Slepian basis pulse envelopes enforcing physical AWG slew-rate and bandwidth bounds ($J(0)=J(T)=0$).
  - **Derivative-Removal via Adiabatic Gate (DRAG)**: Analytical phase error suppression $\epsilon_{\text{DRAG}}(t) = \epsilon(t) - \frac{\dot{\epsilon}(t)}{\Delta B_z}$.
  - JAX automatic differentiation through time-sliced matrix exponentials.
- **Closed-Loop Calibration & Hardware Exporters**:
  - **Bayesian Active Calibration (`BayesianActiveCalibrator`)**: Real-time Kalman tracking of drifting $J_0(t)$ and $\Delta B_z(t)$.
  - **AWG Exporter (`export_awg_waveforms`)**: Direct JSON/CSV export compatible with Qblox, Zurich Instruments, and Keysight hardware.
  - Custom `SiliconExchangeGate(unitary)` and 2-qubit Clifford Randomized Benchmarking in Cirq ($>99.8\%$ fidelity).

---

## 🚀 Quickstart

```python
import numpy as np
from spin_optimal_control import (
    SiliconSpinHamiltonian,
    ExchangeDynamics,
    GRAPEOptimizer,
    SiliconValleyModel,
    DRAGPulseSynthesizer,
    BayesianActiveCalibrator,
    export_awg_waveforms,
)

# 1. Initialize Device Hamiltonian & Valley Model
h = SiliconSpinHamiltonian(j_0=20.0, epsilon_0=1.0, delta_bz=15.0)
valley_model = SiliconValleyModel(valley_splitting_uev=120.0)

# 2. Optimize Pulse via JAX GRAPE
opt = GRAPEOptimizer(h, t_gate_ns=30.0, n_steps=80, n_harmonics=6)
result = opt.optimize_pulse(ExchangeDynamics.target_gate_sqrt_swap())
print(f"Gate Fidelity: {result.gate_fidelity * 100:.4f}%")

# 3. Apply Analytical DRAG Correction
drag = DRAGPulseSynthesizer(delta_bz_mhz=15.0)
in_phase, quad = drag.apply_drag_correction(result.j_pulse, result.dt)

# 4. Export AWG Waveforms
export_awg_waveforms(result.time_grid, in_phase, result.detuning_pulse, quad, file_path="awg_pulse.json")
```

---

## 🧪 Testing & Benchmarks

```bash
pytest -v tests/
python benchmarks/run_benchmarks.py
```

---

## 📄 Citation & License

Developed by **Jasper Sands** under the **Apache-2.0 License**.
