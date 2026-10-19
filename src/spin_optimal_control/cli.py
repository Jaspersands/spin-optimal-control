"""
Command-line interface: ``spin-control <command>``.

Commands
--------
optimize   GRAPE synthesis of an exchange gate (optionally robust / virtual-Z) with AWG export
shape      window-shaped adiabatic CZ: leakage and fidelity per window and duration
valley     valley leakage of a half-sine pulse versus valley splitting
rb         two-qubit Clifford randomized benchmarking on native gates under T1/T2*
noise-psd  generate a 1/f trace and report its fitted spectral slope
devices    measured two-qubit gate fidelity of published devices against the modelled CZ error budget
benchmark  the full benchmark suite (same as ``python benchmarks/run_benchmarks.py``)

Add ``--json`` to any command for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from .hamiltonian import SiliconSpinHamiltonian, ExchangeDynamics
from .grape import GRAPEOptimizer
from .valley import SiliconValleyModel
from .pulse_shaping import AdiabaticCZDesigner, WINDOWS
from .awg_export import export_awg_waveforms
from .noise import PinkNoiseGenerator, SiliconNoiseModel

TARGETS = {
    "sqrt_swap": ExchangeDynamics.target_gate_sqrt_swap,
    "swap": ExchangeDynamics.target_gate_swap,
    "fourth_swap": ExchangeDynamics.target_gate_fourth_swap,
    "cz": ExchangeDynamics.target_gate_cz,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spin-control",
        description="Silicon spin exchange-gate optimal control (GRAPE / valley / RB / noise / device comparison / AWG)",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("optimize", help="Optimise an exchange pulse with JAX GRAPE")
    p.add_argument("--target", choices=sorted(TARGETS), default="sqrt_swap")
    p.add_argument("--duration", type=float, default=30.0, help="gate duration (ns)")
    p.add_argument("--j0", type=float, default=20.0, help="baseline exchange J0 (MHz)")
    p.add_argument("--dbz", type=float, default=0.0, help="Zeeman gradient ΔBz (MHz)")
    p.add_argument("--jmax", type=float, default=40.0, help="amplitude cap (MHz)")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--harmonics", type=int, default=6)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--robust", action="store_true", help="average over a quasi-static noise ensemble")
    p.add_argument("--local-z", action="store_true", help="co-optimise virtual-Z phases")
    p.add_argument("--format", choices=["json", "csv", "qblox", "zi"], default="json")
    p.add_argument("--output", type=str, default=None, help="AWG waveform file")
    p.add_argument("--json", action="store_true")

    w = sub.add_parser("shape", help="Window-shaped adiabatic CZ versus duration")
    w.add_argument("--dbz", type=float, default=20.0, help="Zeeman gradient ΔBz (MHz)")
    w.add_argument("--durations", type=str, default="50,100,200", help="ns, comma-separated")
    w.add_argument("--windows", type=str, default=",".join(WINDOWS))
    w.add_argument("--json", action="store_true")

    v = sub.add_parser("valley", help="Valley leakage for a half-sine pulse")
    v.add_argument("--ev", type=float, default=120.0, help="valley splitting (µeV)")
    v.add_argument("--soc", type=float, default=2.5, help="inter-valley SOC (MHz)")
    v.add_argument("--j-max", type=float, default=30.0)
    v.add_argument("--duration", type=float, default=30.0)
    v.add_argument("--dbz", type=float, default=15.0)
    v.add_argument("--json", action="store_true")

    r = sub.add_parser("rb", help="Two-qubit Clifford randomized benchmarking")
    r.add_argument("--lengths", type=str, default="1,2,4,8,16")
    r.add_argument("--sequences", type=int, default=6)
    r.add_argument("--t1", type=float, default=1000.0, help="µs")
    r.add_argument("--t2", type=float, default=20.0, help="µs")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--abstract", action="store_true", help="Clifford-level model (one ideal layer per Clifford) instead of native gates")
    r.add_argument("--t-pulse", type=float, default=50.0, help="π/2 pulse duration (ns), native model")
    r.add_argument("--t-cz", type=float, default=100.0, help="CZ duration (ns), native model")
    r.add_argument("--json", action="store_true")

    n = sub.add_parser("noise-psd", help="Generate a 1/f^α trace and fit its spectral slope")
    n.add_argument("--alpha", type=float, default=1.0)
    n.add_argument("--steps", type=int, default=4096)
    n.add_argument("--dt", type=float, default=0.1, help="ns")
    n.add_argument("--seed", type=int, default=0)
    n.add_argument("--json", action="store_true")

    d = sub.add_parser("devices", help="Published devices: measured gate fidelity against the CZ error budget")
    d.add_argument("--budget", action="store_true", help="also print each contribution at the published gate time")
    d.add_argument("--optimum", action="store_true", help="also find the best gate time for square and cosine pulses (slower)")
    d.add_argument("--gap", action="store_true",
                   help="also show how much of the measured-minus-modelled gap each candidate mechanism closes "
                        "(closed-form parts only; benchmarks/make_figures.py adds the sweeps)")
    d.add_argument("--json", action="store_true")

    b = sub.add_parser("benchmark", help="Run the benchmark suite")
    b.add_argument("--json", action="store_true")
    return parser


def _emit(payload: Dict[str, Any], as_json: bool, lines: List[str]) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    else:
        print("\n".join(lines))


def cmd_optimize(args) -> int:
    h = SiliconSpinHamiltonian(j_0=args.j0, delta_bz=args.dbz)
    opt = GRAPEOptimizer(
        h, t_gate_ns=args.duration, n_steps=args.steps, n_harmonics=args.harmonics,
        j_max=args.jmax, robust=args.robust, local_z_free=args.local_z,
    )
    res = opt.optimize_pulse(TARGETS[args.target](), max_iter=args.max_iter)
    if args.output:
        export_awg_waveforms(res.time_grid, res.j_pulse, res.detuning_pulse,
                             export_format=args.format, file_path=args.output)
    payload = {
        "target": args.target, "duration_ns": args.duration, "j0_mhz": args.j0, "dbz_mhz": args.dbz,
        "gate_fidelity": res.gate_fidelity, "infidelity": res.infidelity, "iterations": res.iterations,
        "converged": res.is_converged, "exchange_area_mhz_ns": res.exchange_area_mhz_ns,
        "peak_j_mhz": float(res.j_pulse.max()), "virtual_z": res.virtual_z,
        "robust_fidelity_mean": res.robust_fidelity_mean, "output": args.output,
        "j_pulse": res.j_pulse, "time_grid": res.time_grid,
    }
    lines = [
        f"[*] {args.target} | T = {args.duration} ns | J0 = {args.j0} MHz | ΔBz = {args.dbz} MHz",
        f"[+] Gate fidelity {res.gate_fidelity*100:.5f}% (infidelity {res.infidelity:.2e}) in {res.iterations} evaluations",
        f"[+] Exchange area ∫J dt = {res.exchange_area_mhz_ns:.1f} MHz·ns, peak J = {res.j_pulse.max():.2f} MHz",
    ]
    if res.virtual_z is not None:
        lines.append(f"[+] Virtual-Z phases (rad): {np.round(res.virtual_z, 4).tolist()}")
    if res.robust_fidelity_mean is not None:
        lines.append(f"[+] Ensemble fidelity mean/min: {res.robust_fidelity_mean:.5f} / {res.robust_fidelity_min:.5f}")
    if args.output:
        lines.append(f"[+] Wrote {args.format} waveforms to {args.output}")
    _emit(payload, args.json, lines)
    return 0


def cmd_shape(args) -> int:
    designer = AdiabaticCZDesigner(SiliconSpinHamiltonian(delta_bz=args.dbz))
    durations = [float(x) for x in args.durations.split(",")]
    shapes = [x.strip() for x in args.windows.split(",")]
    rows = {s: [designer.calibrate(s, T) for T in durations] for s in shapes}
    payload = {"delta_bz_mhz": args.dbz, "durations_ns": durations,
               "windows": {s: [{"j_max_mhz": r.j_max_mhz, "swap_leakage": r.swap_leakage, "infidelity": r.infidelity} for r in rs]
                           for s, rs in rows.items()}}
    lines = [f"CZ from a window-shaped exchange pulse, ΔBz = {args.dbz} MHz (area 500 MHz·ns; infidelity up to local Z)",
             "  window    " + "".join(f"{T:>12.0f} ns" for T in durations)]
    for s, rs in rows.items():
        lines.append(f"  {s:9s} " + "".join(f"{r.infidelity:>15.2e}" for r in rs))
    _emit(payload, args.json, lines)
    return 0


def cmd_valley(args) -> int:
    vm = SiliconValleyModel(valley_splitting_uev=args.ev, inter_valley_soc_mhz=args.soc)
    n = 120
    dt = args.duration / n
    t = (np.arange(n) + 0.5) * dt
    pulse = args.j_max * np.sin(np.pi * t / args.duration)
    res = vm.compute_valley_leakage(pulse, dt_ns=dt, delta_bz=args.dbz)
    payload = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    lines = [
        f"[+] Valley splitting: {res['valley_splitting_mhz']:.1f} MHz ({args.ev} µeV)",
        f"[+] Max / final valley leakage: {res['max_valley_leakage']:.3e} / {res['final_valley_leakage']:.3e}",
    ]
    _emit(payload, args.json, lines)
    return 0


def cmd_rb(args) -> int:
    from .cirq_backend import CompiledCliffordRB, run_randomized_benchmarking
    lengths = [int(x) for x in args.lengths.split(",")]
    noise = SiliconNoiseModel(t1_us=args.t1, t2_star_us=args.t2)
    if args.abstract:
        res = run_randomized_benchmarking(lengths, n_sequences_per_length=args.sequences, noise_model=noise, seed=args.seed)
        lines = [f"[+] Clifford-level two-qubit RB (T1={args.t1} µs, T2*={args.t2} µs)"]
    else:
        res = CompiledCliffordRB(noise, t_pulse_ns=args.t_pulse, t_cz_ns=args.t_cz).run(lengths, n_sequences=args.sequences, seed=args.seed)
        lines = [f"[+] Native-gate two-qubit RB (T1={args.t1} µs, T2*={args.t2} µs, π/2 = {args.t_pulse} ns, CZ = {args.t_cz} ns; "
                 f"{res['mean_cz_per_clifford']:.2f} CZ / Clifford)"]
    for m, f in zip(res["lengths"], res["fidelities"]):
        lines.append(f"    m={m:3d}  P(00)={f:.4f}")
    lines.append(f"[+] decay p = {res['decay_p']:.5f}, error per Clifford = {res['clifford_error']:.3e}")
    _emit(res, args.json, lines)
    return 0


def cmd_noise_psd(args) -> int:
    gen = PinkNoiseGenerator(alpha=args.alpha, amplitude=1.0, seed=args.seed)
    acc = None
    for _ in range(20):
        f, psd = gen.psd_estimate(gen.generate_spectral_trace(args.steps, args.dt), args.dt)
        acc = psd if acc is None else acc + psd
    acc /= 20
    lo, hi = np.percentile(f, [10, 80])
    band = (f > lo) & (f < hi)
    slope = float(np.polyfit(np.log10(f[band]), np.log10(acc[band]), 1)[0])
    payload = {"alpha": args.alpha, "slope": slope, "n_points": int(band.sum())}
    _emit(payload, args.json, [f"[+] α = {args.alpha}: fitted PSD slope = {slope:.3f} (expected ≈ {-args.alpha:.1f})"])
    return 0


def cmd_devices(args) -> int:
    from .devices import best_gate_times, comparison_table, format_comparison_table, load_devices
    from .error_budget import LABELS
    devs = load_devices()
    rows = comparison_table(devs)
    lines = ["Two-qubit gate fidelity: published measurement against the modelled CZ at the published gate time",
             "(average gate fidelity). The model estimates the error from the noise each paper characterises, with",
             "assumed values where it doesn't; it is not a bound in either direction. It has no crosstalk, drift, SPAM",
             "or leakage.", "",
             format_comparison_table(rows)]
    if args.budget:
        lines += ["", "Modelled contributions at the published gate time (average gate infidelity):",
                  f"  {'device':42s}" + "".join(f"{LABELS[k]:>28s}" for k in rows[0]["contributions"])]
        for r in rows:
            lines.append(f"  {r['device']:42s}" + "".join(f"{v:>28.2e}" for v in r["contributions"].values()))
    if args.optimum:
        lines += ["", "Best modelled gate time per pulse shape:"]
        for r in rows:
            best = best_gate_times(devs[r["key"]])
            r["optimum"] = {s: {"duration_ns": b.duration_ns, "fidelity": b.fidelity, "dominant": b.dominant} for s, b in best.items()}
            lines.append(f"  {r['device']:42s} " + "   ".join(
                f"{s}: {b.duration_ns:6.1f} ns, {100 * b.fidelity:.3f}%" for s, b in best.items()))
    lines += ["", "Zeeman noise exponent alpha per qubit (fitted to T2* and the Hahn echo where one is reported):"]
    for r in rows:
        fits = devs[r["key"]].zeeman_fit()
        lines.append(f"  {r['device']:42s} " + ", ".join(f"Q{q + 1} {f['alpha']:.2f} ({f['status']})" for q, f in enumerate(fits)))
    notes = [f"  {r['device']}: measured gate is a {r['gate']}; the model is an exchange-pulsed CZ of the same duration"
             for r in rows if r["gate"] != "CZ"]
    notes += [f"  {r['device']}: modelled with the frequency difference at which the square pulse is synchronised; "
              f"at the idle value its coherent error would be {r['coherent_at_idle_dez']:.1e}"
              for r in rows if r.get("coherent_at_idle_dez") is not None]
    if notes:
        lines += [""] + notes
    if args.gap:
        from .devices import format_gap_table, gap_analysis

        gaps = [gap_analysis(devs[r["key"]], sweeps=False) for r in rows]
        for r, g in zip(rows, gaps):
            r["gap"] = g
        lines += ["", "Candidate mechanisms for the gap (measured minus modelled, average infidelity). A value is the",
                  "added error and its share of the gap, from published or derived inputs ('<=': an upper bound,",
                  "the quantity was fed back during the run). 'needs' is the unpublished input that would close",
                  "the gap on its own; terms that need a sweep are left out here.", "",
                  format_gap_table(gaps)]
        for g in gaps:
            irb = g["benchmark"].get("irb")
            if irb:
                lines.append(f"  {g['short']}: interleaved RB gives {irb['estimate']:.2e}; with the reference "
                             f"Clifford error the true gate error lies within +/- {irb['bound']:.2e} of it "
                             "(Magesan et al. 2012)")
    _emit({"devices": rows}, args.json, lines)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "benchmark":
        from . import benchmark as bench
        return bench.run_full_benchmark(as_json=args.json)
    return {
        "optimize": cmd_optimize,
        "shape": cmd_shape,
        "valley": cmd_valley,
        "rb": cmd_rb,
        "noise-psd": cmd_noise_psd,
        "devices": cmd_devices,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
