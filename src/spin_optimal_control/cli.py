"""
Command-Line Interface for spin_optimal_control.
"""

from __future__ import annotations
import argparse
import sys
import json
import numpy as np
from .hamiltonian import SiliconSpinHamiltonian, ExchangeDynamics
from .grape import GRAPEOptimizer
from .valley import SiliconValleyModel
from .drag import DRAGPulseSynthesizer
from .awg_export import export_awg_waveforms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spin-control",
        description="Silicon Spin Pulse Optimal Control CLI (GRAPE / Valley / DRAG / AWG)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: optimize
    opt_parser = subparsers.add_parser("optimize", help="Optimize spin exchange pulse via JAX GRAPE")
    opt_parser.add_argument(
        "--target",
        type=str,
        choices=["sqrt_swap", "swap", "fourth_swap", "cz"],
        default="sqrt_swap",
        help="Target two-qubit exchange operation",
    )
    opt_parser.add_argument("--duration", type=float, default=30.0, help="Gate duration in ns")
    opt_parser.add_argument("--j0", type=float, default=20.0, help="Baseline exchange J0 in MHz")
    opt_parser.add_argument("--steps", type=int, default=60, help="Time discretization steps")
    opt_parser.add_argument("--drag", action="store_true", help="Apply analytical DRAG correction")
    opt_parser.add_argument("--ev", type=float, default=120.0, help="Valley splitting Ev in ueV")
    opt_parser.add_argument("--output", type=str, default=None, help="Output file path for AWG waveform JSON")

    # Command: valley
    val_parser = subparsers.add_parser("valley", help="Simulate valley leakage for a given pulse")
    val_parser.add_argument("--ev", type=float, default=120.0, help="Valley splitting Ev in ueV")
    val_parser.add_argument("--j-max", type=float, default=30.0, help="Peak exchange amplitude in MHz")
    val_parser.add_argument("--duration", type=float, default=30.0, help="Pulse duration in ns")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "optimize":
        print(f"[*] Initializing Silicon DQD Hamiltonian (J0={args.j0} MHz, T={args.duration} ns)...")
        h = SiliconSpinHamiltonian(j_0=args.j0, delta_bz=15.0)
        opt = GRAPEOptimizer(h, t_gate_ns=args.duration, n_steps=args.steps, n_harmonics=6)

        target_map = {
            "sqrt_swap": ExchangeDynamics.target_gate_sqrt_swap(),
            "swap": ExchangeDynamics.target_gate_swap(),
            "fourth_swap": ExchangeDynamics.target_gate_fourth_swap(),
            "cz": ExchangeDynamics.target_gate_cz(),
        }
        target_u = target_map[args.target]

        print(f"[*] Running JAX GRAPE optimization for {args.target}...")
        res = opt.optimize_pulse(target_u)
        print(f"[+] Optimization Converged! Gate Fidelity: {res.gate_fidelity * 100:.4f}% (Infidelity: {res.infidelity:.2e})")

        in_phase = res.j_pulse
        quad = None
        if args.drag:
            print("[*] Applying analytical DRAG correction...")
            drag = DRAGPulseSynthesizer(delta_bz_mhz=15.0)
            in_phase, quad = drag.apply_drag_correction(res.j_pulse, res.dt)

        if args.output:
            export_awg_waveforms(res.time_grid, in_phase, res.detuning_pulse, quad, file_path=args.output)
            print(f"[+] Saved AWG waveforms to {args.output}")

        return 0

    elif args.command == "valley":
        vm = SiliconValleyModel(valley_splitting_uev=args.ev)
        t_grid = np.linspace(0, args.duration, 60)
        dt = args.duration / 60.0
        pulse = args.j_max * np.sin(np.pi * t_grid / args.duration)
        res = vm.compute_valley_leakage(pulse, dt=dt)
        print(f"[+] Valley Splitting: {res['valley_splitting_mhz']:.1f} MHz ({args.ev} ueV)")
        print(f"[+] Final Valley Leakage: {res['final_valley_leakage'] * 100:.4f}%")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
