"""
End-to-end benchmark: GRAPE synthesis → window-shaped adiabatic CZ → noise
Monte-Carlo → Lindblad channel fidelity → toggle-frame filter function vs
Monte-Carlo → native-gate Clifford RB → valley leakage. Also exposed as
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
    say("SILICON SPIN EXCHANGE OPTIMAL CONTROL BENCHMARK  (v0.4, MHz / ns units)")
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

    # 2b. Window-shaped adiabatic CZ
    from .pulse_shaping import AdiabaticCZDesigner, WINDOWS
    say("\n2b. Window-shaped CZ, ΔBz = 20 MHz (pulse area 500 MHz·ns; infidelity up to local Z)")
    designer = AdiabaticCZDesigner(SiliconSpinHamiltonian(delta_bz=20.0), n_steps=400)
    durations = (50.0, 100.0, 200.0)
    say("   window    " + "".join(f"{T:>10.0f} ns" for T in durations))
    shaped = {}
    for w in WINDOWS:
        row = [designer.calibrate(w, T).infidelity for T in durations]
        shaped[w] = row
        say(f"   {w:9s} " + "".join(f"{v:>13.1e}" for v in row))
    out["shaped_cz"] = shaped

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

    # 4b. Toggle-frame filter function vs Monte-Carlo
    import scipy.linalg
    from .noise import exchange_gate_noise_operators, gate_filter_functions, infidelity_from_filter_function
    from .units import MHZ_NS_TO_RAD
    say("\n4b. Filter function in the toggle frame: OU frequency noise (σ = 0.3 MHz, τc = 5 ns) on one electron, cosine CZ 100 ns")
    hz = SiliconSpinHamiltonian(delta_bz=20.0)
    czp = AdiabaticCZDesigner(hz, n_steps=160).calibrate("cosine", 100.0)
    Hs = [hz.get_hamiltonian_matrix(float(x)) for x in czp.j_pulse]
    ops = exchange_gate_noise_operators(czp.j_pulse, "zeeman1")
    om = np.concatenate([[0.0], np.logspace(-4, np.log10(np.pi / czp.dt_ns), 2500)])
    sig, tc = 0.3, 5.0
    pred = infidelity_from_filter_function(om, gate_filter_functions(Hs, czp.dt_ns, ops, om)[0], 2 * sig**2 * 2 * tc / (1 + (om * tc) ** 2))
    steps = [scipy.linalg.expm(-1j * MHZ_NS_TO_RAD * H * czp.dt_ns) for H in Hs]
    U0 = np.linalg.multi_dot(steps[::-1])
    rng = np.random.default_rng(7); decay = np.exp(-czp.dt_ns / tc); vals = []
    for _ in range(60 if quick else 200):
        x = np.empty(len(Hs)); x[0] = rng.normal(0, sig)
        for k in range(1, len(Hs)):
            x[k] = decay * x[k - 1] + np.sqrt(1 - decay**2) * sig * rng.normal()
        U = np.eye(4, dtype=complex)
        for k in range(len(Hs)):
            U = scipy.linalg.expm(-1j * MHZ_NS_TO_RAD * (Hs[k] + x[k] * ops[0][0]) * czp.dt_ns) @ U
        vals.append(1 - abs(np.trace(U0.conj().T @ U)) ** 2 / 16)
    say(f"   -> first-order prediction {pred:.3e} | Monte-Carlo {np.mean(vals):.3e} ± {np.std(vals)/np.sqrt(len(vals)):.1e}")
    out["filter_function"] = {"prediction": pred, "monte_carlo": float(np.mean(vals))}

    # 5. Two-qubit Clifford IRB
    try:
        from .cirq_backend import run_interleaved_rb
        say("\n5. Clifford-level interleaved RB of the GRAPE √SWAP (11 520-element group, one ideal layer per Clifford)")
        t0 = time.time()
        rb = run_interleaved_rb(res.synthesized_unitary, [1, 2, 4, 8, 16, 32, 64],
                                n_sequences_per_length=6 if quick else 12, noise_model=noise, seed=42)
        say(f"   -> {time.time() - t0:.1f} s | p_ref = {rb['decay_p_ref']:.5f} | p_int = {rb['decay_p_interleaved']:.5f} | gate error = {rb['gate_error']:.2e}")
        out["irb"] = {k: rb[k] for k in ("decay_p_ref", "decay_p_interleaved", "gate_error", "gate_fidelity")}
        from .cirq_backend import CompiledCliffordRB, best_z_corrected_cz
        say("\n5b. Native-gate Clifford RB: ±X/2, ±Y/2 (50 ns, p = 1e-3), virtual Z, cosine CZ (100 ns, ΔBz = 20 MHz), T1 = 1 ms, T2* = 20 µs")
        U_cz = best_z_corrected_cz(ExchangeDynamics(hz).propagate_unitary(czp.j_pulse, czp.dt_ns))
        nm = SiliconNoiseModel(t1_us=1000.0, t2_star_us=20.0, charge_noise_amp=0.0, overhauser_sigma=0.0)
        crb = CompiledCliffordRB(nm, t_pulse_ns=50.0, t_cz_ns=100.0, p_pulse=1e-3, cz_unitary=U_cz)
        t0 = time.time()
        ref = crb.run([1, 4, 8, 16, 32, 64], n_sequences=15 if quick else 40, seed=5)
        irb = crb.run_interleaved_cz([1, 4, 8, 16, 32, 64], n_sequences=15 if quick else 40, seed=6)
        say(f"   -> {time.time() - t0:.1f} s | {ref['mean_cz_per_clifford']:.2f} CZ and {ref['mean_pulse_layers_per_clifford']:.2f} pulse layers per Clifford "
            f"({ref['mean_clifford_duration_ns']:.0f} ns) | error per Clifford {ref['clifford_error']:.2e} | IRB CZ error {irb['cz_error']:.2e}")
        out["compiled_rb"] = {"clifford_error": ref["clifford_error"], "cz_error": irb["cz_error"], **{k: ref[k] for k in ("mean_cz_per_clifford", "mean_clifford_duration_ns")}}
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
