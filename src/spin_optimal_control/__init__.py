"""
spin_optimal_control
====================
Differentiable optimal control for silicon spin exchange gates under 1/f
charge noise and Overhauser drift — JAX GRAPE, Lindblad open-system
simulation, filter functions, two-qubit Clifford randomized benchmarking in
Cirq, valley leakage, Bayesian drift tracking and AWG export.

Units: frequencies in MHz, times in ns (see :mod:`spin_optimal_control.units`).
"""

from .units import MHZ_NS_TO_RAD, UEV_TO_MHZ, phase
from .hamiltonian import SiliconSpinHamiltonian, ExchangeDynamics
from .noise import (
    PinkNoiseGenerator,
    OverhauserNoise,
    SiliconNoiseModel,
    filter_function,
    filter_function_infidelity,
)
from .grape import GRAPEOptimizer, SmoothFourierPulse, PulseOptimizationResult
from .cirq_backend import (
    SiliconExchangeGate,
    CirqSiliconSimulator,
    generate_single_qubit_cliffords,
    generate_two_qubit_cliffords,
    run_randomized_benchmarking,
    run_interleaved_rb,
    rb_fit,
    clifford_error_from_p,
)
from .valley import SiliconValleyModel
from .drag import DRAGPulseSynthesizer
from .calibration import BayesianActiveCalibrator, CalibrationState, simulate_drift_tracking
from .awg_export import export_awg_waveforms

__version__ = "0.3.0"
__all__ = [
    "MHZ_NS_TO_RAD",
    "UEV_TO_MHZ",
    "phase",
    "SiliconSpinHamiltonian",
    "ExchangeDynamics",
    "PinkNoiseGenerator",
    "OverhauserNoise",
    "SiliconNoiseModel",
    "filter_function",
    "filter_function_infidelity",
    "GRAPEOptimizer",
    "SmoothFourierPulse",
    "PulseOptimizationResult",
    "SiliconExchangeGate",
    "CirqSiliconSimulator",
    "generate_single_qubit_cliffords",
    "generate_two_qubit_cliffords",
    "run_randomized_benchmarking",
    "run_interleaved_rb",
    "rb_fit",
    "clifford_error_from_p",
    "SiliconValleyModel",
    "DRAGPulseSynthesizer",
    "BayesianActiveCalibrator",
    "CalibrationState",
    "simulate_drift_tracking",
    "export_awg_waveforms",
]
