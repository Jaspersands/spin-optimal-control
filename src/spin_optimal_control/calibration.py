"""
Closed-Loop Active Feedback Calibration and Bayesian Parameter Tracking.

Tracks slow 1/f charge drift and Overhauser frequency shifts in real time
using Bayesian update rules and simulated adaptive Ramsey/Rabi experiments.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any


@dataclass
class CalibrationState:
    """Estimated mean and variance of physical device parameters."""
    estimated_j0: float
    var_j0: float
    estimated_delta_bz: float
    var_delta_bz: float


class BayesianActiveCalibrator:
    """
    Recursive Bayesian filter tracking device parameters (J_0, Delta_Bz)
    under continuous 1/f drift.
    """

    def __init__(
        self,
        initial_j0: float = 20.0,
        initial_delta_bz: float = 15.0,
        prior_std_j0: float = 2.0,
        prior_std_db: float = 1.5,
        drift_rate_per_step: float = 0.05,
    ):
        self.state = CalibrationState(
            estimated_j0=initial_j0,
            var_j0=prior_std_j0**2,
            estimated_delta_bz=initial_delta_bz,
            var_delta_bz=prior_std_db**2,
        )
        self.drift_var = drift_rate_per_step**2

    def update_from_ramsey_measurement(
        self, observed_frequency_mhz: float, measurement_std_mhz: float = 0.2
    ) -> CalibrationState:
        """
        Updates the estimate of Delta_Bz from a single Ramsey experiment.
        """
        # 1. Prediction step: Add process drift variance
        p_var = self.state.var_delta_bz + self.drift_var

        # 2. Kalman / Bayesian update step
        meas_var = measurement_std_mhz**2
        kalman_gain = p_var / (p_var + meas_var)

        new_mean = self.state.estimated_delta_bz + kalman_gain * (
            observed_frequency_mhz - self.state.estimated_delta_bz
        )
        new_var = (1.0 - kalman_gain) * p_var

        self.state.estimated_delta_bz = float(new_mean)
        self.state.var_delta_bz = float(new_var)
        return self.state

    def update_from_exchange_oscillation(
        self, observed_j_mhz: float, measurement_std_mhz: float = 0.4
    ) -> CalibrationState:
        """
        Updates the estimate of J_0 from two-electron exchange oscillations.
        """
        p_var = self.state.var_j0 + self.drift_var
        meas_var = measurement_std_mhz**2
        kalman_gain = p_var / (p_var + meas_var)

        new_mean = self.state.estimated_j0 + kalman_gain * (
            observed_j_mhz - self.state.estimated_j0
        )
        new_var = (1.0 - kalman_gain) * p_var

        self.state.estimated_j0 = float(new_mean)
        self.state.var_j0 = float(new_var)
        return self.state
