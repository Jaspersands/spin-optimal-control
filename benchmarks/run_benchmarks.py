"""
Benchmark runner for pulse optimization under pink charge noise and Overhauser drift.
"""

import time
import numpy as np
from spin_optimal_control.hamiltonian import SiliconSpinHamiltonian, ExchangeDynamics
from spin_optimal_control.noise import SiliconNoiseModel
from spin_optimal_control.grape import GRAPEOptimizer
from spin_optimal_control.cirq_backend import run_interleaved_rb


def run_full_benchmark():
    print("=" * 65)
    print("SILICON SPIN EXCHANGE OPTIMAL CONTROL BENCHMARK")
    print("=" * 65)

    h = SiliconSpinHamiltonian(j_0=20.0, epsilon_0=1.0, delta_bz=12.0, b_0=100.0)
    dyn = ExchangeDynamics(h)
    u_target = dyn.target_gate_sqrt_swap()

    print("\n1. Running JAX Differentiable GRAPE Pulse Optimization...")
    t0 = time.time()
    opt = GRAPEOptimizer(h, t_gate_ns=30.0, n_steps=60, n_harmonics=5)
    res = opt.optimize_pulse(u_target, max_iter=80, tolerance=1e-5)
    t1 = time.time()

    print(f"   -> Completed in {t1 - t0:.3f} s")
    print(f"   -> Optimized Gate Fidelity: {res.gate_fidelity * 100:.4f}%")
    print(f"   -> Gate Infidelity: {res.infidelity:.2e}")
    print(f"   -> Optimization Iterations: {res.iterations}")

    print("\n2. Evaluating under 1/f Charge Noise & Overhauser Fluctuations...")
    noise = SiliconNoiseModel(
        t1_us=800.0, t2_star_us=25.0, charge_noise_amp=0.04, overhauser_sigma=0.6, seed=42
    )

    n_trials = 50
    fids_noisy = []
    for _ in range(n_trials):
        j_noisy, db_noisy = noise.apply_noisy_pulses(res.j_pulse, h.epsilon_0, res.dt, h.delta_bz)
        u_noisy = dyn.propagate_unitary(j_noisy, res.dt, db_noisy)
        fids_noisy.append(dyn.gate_fidelity(u_noisy, u_target))

    print(f"   -> Mean Fidelity under 1/f Pink Noise: {np.mean(fids_noisy) * 100:.3f}% +/- {np.std(fids_noisy)*100:.3f}%")

    print("\n3. Executing Cirq Interleaved Randomized Benchmarking...")
    rb_lengths = [1, 2, 4, 8, 16, 32]
    rb_res = run_interleaved_rb(
        res.synthesized_unitary,
        rb_lengths,
        n_sequences_per_length=10,
        noise_model=noise,
        seed=42,
    )

    print(f"   -> Reference Clifford Decay p: {rb_res['decay_p_ref']:.4f}")
    print(f"   -> Interleaved Decay p:        {rb_res['decay_p_interleaved']:.4f}")
    print(f"   -> Isolated Gate Fidelity:     {rb_res['gate_fidelity'] * 100:.3f}%")
    print("=" * 65)


if __name__ == "__main__":
    run_full_benchmark()
