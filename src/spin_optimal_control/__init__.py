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
from .valley import SiliconValleyModel
from .drag import DRAGPulseSynthesizer
from .calibration import BayesianActiveCalibrator, CalibrationState
from .awg_export import export_awg_waveforms

__version__ = "0.2.0"
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
    "SiliconValleyModel",
    "DRAGPulseSynthesizer",
    "BayesianActiveCalibrator",
    "CalibrationState",
    "export_awg_waveforms",
]
