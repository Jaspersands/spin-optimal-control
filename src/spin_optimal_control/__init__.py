"""
spin_optimal_control
====================
Differentiable Optimal Control for Silicon Spin Exchange Gates
under 1/f Charge Noise and Overhauser Nuclear Spin Drift.
"""

from .hamiltonian import SiliconSpinHamiltonian, ExchangeDynamics
from .noise import PinkNoiseGenerator, OverhauserNoise, SiliconNoiseModel
from .grape import GRAPEOptimizer, SmoothFourierPulse, PulseOptimizationResult
from .cirq_backend import (
    SiliconExchangeGate,
    CirqSiliconSimulator,
    run_randomized_benchmarking,
    run_interleaved_rb,
)

__version__ = "0.1.0"
__all__ = [
    "SiliconSpinHamiltonian",
    "ExchangeDynamics",
    "PinkNoiseGenerator",
    "OverhauserNoise",
    "SiliconNoiseModel",
    "GRAPEOptimizer",
    "SmoothFourierPulse",
    "PulseOptimizationResult",
    "SiliconExchangeGate",
    "CirqSiliconSimulator",
    "run_randomized_benchmarking",
    "run_interleaved_rb",
]
