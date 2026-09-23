"""
spin_optimal_control
====================
Optimal control of exchange gates in silicon double quantum dots: JAX pulse
optimisation, adiabatic CZ pulse shapes, 1/f charge and nuclear-field noise,
thermal Lindblad decoherence, toggle-frame filter functions, native-gate
Clifford randomized benchmarking in Cirq, valley leakage, drift tracking and
AWG export.

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
    gate_filter_functions,
    control_matrix,
    infidelity_from_filter_function,
    exchange_gate_noise_operators,
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
    CompiledCliffordRB,
    best_z_corrected_cz,
)
from .valley import SiliconValleyModel
from .pulse_shaping import AdiabaticCZDesigner, ShapedCZ, window, conditional_phase, swap_leakage
from .calibration import BayesianActiveCalibrator, CalibrationState, simulate_drift_tracking
from .awg_export import export_awg_waveforms

__version__ = "0.4.0"
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
    "gate_filter_functions",
    "control_matrix",
    "infidelity_from_filter_function",
    "exchange_gate_noise_operators",
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
    "CompiledCliffordRB",
    "best_z_corrected_cz",
    "SiliconValleyModel",
    "AdiabaticCZDesigner",
    "ShapedCZ",
    "window",
    "conditional_phase",
    "swap_leakage",
    "BayesianActiveCalibrator",
    "CalibrationState",
    "simulate_drift_tracking",
    "export_awg_waveforms",
]
