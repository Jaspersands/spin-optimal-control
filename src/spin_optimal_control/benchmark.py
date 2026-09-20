"""
End-to-end benchmark: GRAPE synthesis → noise Monte-Carlo → Lindblad channel
fidelity → two-qubit Clifford IRB → valley leakage. Also exposed as
``spin-control benchmark``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

import numpy as np

from .hamiltonian import SiliconSpinHamiltonian, ExchangeDynamics
from .noise import SiliconNoiseModel
from .grape import GRAPEOptimizer
from .valley import SiliconValleyModel


def run_full_benchmark(as_json: bool = False, quick: bool = False) -> int:
    out: Dict[str, Any] = {}
    log = [] if as_json else None

    def say(msg: str = "") -> None:
        if log is None:
            print(msg)
        else:
            log.append(msg)

    say("=" * 70)
    say("SILICON SPIN EXCHANGE OPTIMAL CONTROL BENCHMARK  (v0.3, MHz / ns units)")
    say("=" * 70)

    # 1. √SWAP with a small residual gradient (no micromagnet)
    h = SiliconSpinHamiltonian(j_0=20.0, epsilon_0=1.0, delta_bz=0.5)
    dyn = ExchangeDynamics(h)
    u_sqrt = ExchangeDynamics.target_gate_sqrt_swap()
    say("\n1. JAX GRAPE: √SWAP, T = 30 ns, ΔBz = 0.5 MHz")
    t0 = time.time()
    opt = GRAPEOptimizer(h, t_gate_ns=30.0, n_steps=60, n_harmonics=5, j_max=40.0)
    res = opt.optimize_pulse(u_sqrt, max_iter=200)
    say(f"   -> {time.time() - t0:.2f} s | F = {res.gate_fidelity*100:.5f}% | ∫J dt = {res.exchange_area_mhz_ns:.1f} MHz·ns | peak J = {res.j_pulse.max():.1f} MHz")
    out["sqrt_swap"] = {"fidelity": res.gate_fidelity, "area": res.exchange_area_mhz_ns, "peak_j": float(res.j_pulse.max())}

    # 2. CZ under a micromagnet gradient with virtual-Z co-optimisation
    say("\n2. JAX GRAPE: CZ, T = 40 ns, ΔBz = 30 MHz, virtual-Z co-optimised")
    h_cz = SiliconSpinHamiltonian(j_0=20.0, delta_bz=30.0)
    t0 = time.time()
    opt_cz = GRAPEOptimizer(h_cz, t_gate_ns=40.0, n_steps=80, n_harmonics=6, j_max=40.0, local_z_free=True)
    res_cz = opt_cz.optimize_pulse(ExchangeDynamics.target_gate_cz(), max_iter=300)
    say(f"   -> {time.time() - t0:.2f} s | F = {res_cz.gate_fidelity*100:.5f}% | virtual-Z = {np.round(res_cz.virtual_z, 3).tolist()}")
    out["cz"] = {"fidelity": res_cz.gate_fidelity, "virtual_z": res_cz.virtual_z.tolist()}

    # 3. Noise Monte-Carlo on the √SWAP pulse
    say("\n3. Quasi-static 1/f charge noise + Overhauser Monte-Carlo (√SWAP pulse)")
    noise = SiliconNoiseModel(t1_us=800.0, t2_star_us=25.0, charge_noise_amp=0.04, overhauser_sigma=0.3, seed=42)
    n_trials = 30 if quick else 100
    fids = []
    for _ in range(n_trials):
        j_n, db_n = noise.apply_noisy_pulses(res.j_pulse, h.epsilon_0, res.dt, h.delta_bz)
        fids.append(dyn.gate_fidelity(dyn.propagate_unitary(j_n, res.dt, db_n), u_sqrt))
    say(f"   -> mean F = {np.mean(fids)*100:.3f}% ± {np.std(fids)*100:.3f}%  ({n_trials} trials)")
    out["noise_mc"] = {"mean": float(np.mean(fids)), "std": float(np.std(fids))}

    # 4. Lindblad channel fidelity
    say("\n4. Lindblad channel process fidelity (T1 = 800 µs, T2* = 25 µs)")
    f_ch = noise.channel_process_fidelity(res.j_pulse, res.dt, u_sqrt, h.get_hamiltonian_matrix)
    say(f"   -> F_pro = {f_ch*100:.4f}%  (coherent limit {res.gate_fidelity*100:.4f}%)")
    out["lindblad"] = {"process_fidelity": f_ch}

    # 5. Two-qubit Clifford IRB
    try:
        from .cirq_backend import run_interleaved_rb
        say("\n5. Cirq two-qubit Clifford interleaved RB (11 520-element group)")
        t0 = time.time()
        rb = run_interleaved_rb(res.synthesized_unitary, [1, 2, 4, 8, 16, 32, 64],
                                n_sequences_per_length=6 if quick else 12, noise_model=noise, seed=42)
        say(f"   -> {time.time() - t0:.1f} s | p_ref = {rb['decay_p_ref']:.5f} | p_int = {rb['decay_p_interleaved']:.5f} | gate error = {rb['gate_error']:.2e}")
        out["irb"] = {k: rb[k] for k in ("decay_p_ref", "decay_p_interleaved", "gate_error", "gate_fidelity")}
    except ImportError:
        say("\n5. Cirq not installed — skipping RB")

    # 6. Valley leakage
    say("\n6. Valley leakage of the √SWAP pulse vs valley splitting (SOC = 2.5 MHz)")
    leak = {}
    for ev in (50.0, 100.0, 200.0):
        vm = SiliconValleyModel(valley_splitting_uev=ev)
        leak[ev] = vm.compute_valley_leakage(res.j_pulse, dt_ns=res.dt, delta_bz=h.delta_bz)["max_valley_leakage"]
        say(f"   -> E_v = {ev:5.0f} µeV : max leakage = {leak[ev]:.2e}")
    out["valley"] = {str(k): v for k, v in leak.items()}
    say("=" * 70)

    if as_json:
        print(json.dumps(out, indent=2))
    return 0
