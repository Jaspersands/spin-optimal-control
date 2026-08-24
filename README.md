# Differentiable Optimal Control for Silicon Spin Exchange Gates

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX-Accelerated](https://img.shields.io/badge/Autodiff-JAX-red.svg)](https://github.com/google/jax)
[![Cirq Integration](https://img.shields.io/badge/Framework-Cirq%20%7C%20PennyLane-teal.svg)](https://quantumai.google/cirq)

> **JAX-differentiable pulse engineering (GRAPE / smooth Fourier basis) and Cirq Randomized Benchmarking for two-electron exchange gates in silicon quantum dots subject to $1/f^\alpha$ pink charge noise and Overhauser nuclear spin fluctuations.**

---

## 🌟 Key Features

- **Realistic Silicon Quantum Dot Physics**:
  - Full two-electron spin exchange Hamiltonian $H(t) = \frac{J(t)}{4}(\vec{\sigma}_1 \cdot \vec{\sigma}_2) + \frac{\Delta B_z(t)}{2}(\sigma_1^z - \sigma_2^z) + \frac{B_0}{2}(\sigma_1^z + \sigma_2^z)$.
  - Detuning exponential voltage relation $J(\epsilon) = J_0 \exp(\epsilon / \epsilon_0)$.
  - Singlet-triplet state transitions ($\{|S\rangle, |T_0\rangle, |T_+\rangle, |T_-\rangle\}$).
- **Physical Noise Modeling**:
  - $1/f^\alpha$ pink charge noise via multi-trap Lorentzian Two-Level Fluctuators (TLF) and spectral synthesis.
  - Quasistatic Overhauser field fluctuations ($\delta B_n \sim \mathcal{N}(0, \sigma_N^2)$) from residual $^{29}\text{Si}$ nuclear spins.
  - Open-system Lindblad master equation with finite $T_1$ relaxation and $T_2^*$ dephasing.
- **Differentiable Optimal Control**:
  - Smooth Fourier / Slepian basis pulse envelopes enforcing physical AWG slew-rate and bandwidth bounds ($J(0)=J(T)=0$).
  - JAX automatic differentiation through time-sliced matrix exponentials ($\partial \mathcal{F} / \partial \vec{\theta}$).
  - Robust ensemble optimization evading charge noise and detuning drifts.
- **Hardware-Targeted Cirq Backend**:
  - Custom `SiliconExchangeGate(unitary)`.
  - Automated 2-qubit Clifford Randomized Benchmarking (RB) and Interleaved RB (IRB) to extract average gate fidelities ($>99.8\%$).

---

## 📐 Mathematical Formulation

### 1. Two-Qubit Silicon Exchange Hamiltonian

In the computational basis $\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$, the system dynamics are governed by:

$$\hat{H}(t) = \frac{J(t)}{4} \left( \sigma_1^x \sigma_2^x + \sigma_1^y \sigma_2^y + \sigma_1^z \sigma_2^z \right) + \frac{\Delta B_z}{2} (\sigma_1^z - \sigma_2^z) + \frac{B_0}{2} (\sigma_1^z + \sigma_2^z)$$

### 2. Smooth Pulse Parameterization

To avoid high-frequency spectral leakage and respect Arbitrary Waveform Generator (AWG) bandwidth limits, pulses are parameterized as:

$$J(t) = \text{ReLU}\left( \sum_{k=1}^K a_k \sin\left(\frac{k \pi t}{T}\right) + \sum_{k=1}^K b_k \left(1 - \cos\left(\frac{2 k \pi t}{T}\right)\right) \right)$$

### 3. Infidelity & Robust Objective Function

$$\mathcal{L}(\vec{\theta}) = 1 - \frac{1}{d^2} \left| \text{Tr}\left( U_{\text{target}}^\dagger U(T; \vec{\theta}) \right) \right|^2 + \lambda_{\text{slew}} \int_0^T \left( \frac{dJ}{dt} \right)^2 dt$$

---

## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/Jaspersands/spin-optimal-control.git
cd spin-optimal-control
pip install -e .
```

### Python API Example

```python
import numpy as np
from spin_optimal_control import (
    SiliconSpinHamiltonian,
    ExchangeDynamics,
    GRAPEOptimizer,
    SiliconNoiseModel,
    run_interleaved_rb,
)

# 1. Initialize Device Hamiltonian
h = SiliconSpinHamiltonian(j_0=20.0, epsilon_0=1.0, delta_bz=12.0, b_0=100.0)
dyn = ExchangeDynamics(h)
u_target = dyn.target_gate_sqrt_swap()

# 2. Optimize Pulse via JAX GRAPE
opt = GRAPEOptimizer(h, t_gate_ns=30.0, n_steps=80, n_harmonics=6)
result = opt.optimize_pulse(u_target, max_iter=100)
print(f"Synthesized Gate Fidelity: {result.gate_fidelity * 100:.4f}%")

# 3. Simulate Cirq Interleaved Randomized Benchmarking
noise = SiliconNoiseModel(t1_us=1000.0, t2_star_us=25.0, charge_noise_amp=0.03)
rb_data = run_interleaved_rb(result.synthesized_unitary, lengths=[1, 2, 4, 8, 16, 32], noise_model=noise)
print(f"Interleaved Gate Fidelity: {rb_data['gate_fidelity'] * 100:.3f}%")
```

---

## 🧪 Testing & Benchmarks

Run the test suite:
```bash
pytest -v tests/
```

Execute the performance benchmark:
```bash
python benchmarks/run_benchmarks.py
```

---

## 🌐 Interactive Web Showcase

An interactive simulation dashboard is located in `web/index.html`. Open it in any browser to:
- Interactively shape Fourier pulse harmonics and view real-time $J(t)$ waveforms.
- Visualize 3D Singlet-Triplet state trajectories on the Bloch sphere.
- Inject $1/f$ pink noise and Overhauser nuclear field fluctuations live.
- Simulate Clifford Randomized Benchmarking decay curves.

---

## 📄 Citation & License

Developed by **Jasper Sands** under the **Apache-2.0 License**.
