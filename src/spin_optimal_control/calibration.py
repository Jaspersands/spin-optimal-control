"""
Closed-loop Bayesian (Kalman) tracking of drifting device parameters.

A two-dimensional linear Kalman filter tracks (J₀, ΔB_z) under random-walk
drift, updated by two kinds of calibration experiments:

* Ramsey oscillations → direct noisy observation of ΔB_z,
* exchange oscillations at a reference detuning → observation of J₀.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class CalibrationState:
    """Posterior mean and covariance of (J₀, ΔB_z)."""

    estimated_j0: float
    var_j0: float
    estimated_delta_bz: float
    var_delta_bz: float
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))

    @property
    def mean(self) -> np.ndarray:
        return np.array([self.estimated_j0, self.estimated_delta_bz])


class BayesianActiveCalibrator:
    """
    Kalman filter over x = [J₀, ΔB_z] with process noise Q = σ_drift² · I and
    scalar observations of either component.
    """

    def __init__(
        self,
        initial_j0: float = 20.0,
        initial_delta_bz: float = 15.0,
        prior_std_j0: float = 2.0,
        prior_std_db: float = 1.5,
        drift_rate_per_step: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.x = np.array([float(initial_j0), float(initial_delta_bz)])
        self.P = np.diag([float(prior_std_j0) ** 2, float(prior_std_db) ** 2])
        self.Q = np.eye(2) * float(drift_rate_per_step) ** 2
        self.rng = np.random.default_rng(seed)
        self.history = []

    # ----------------------------------------------------------------- #
    @property
    def state(self) -> CalibrationState:
        return CalibrationState(
            estimated_j0=float(self.x[0]),
            var_j0=float(self.P[0, 0]),
            estimated_delta_bz=float(self.x[1]),
            var_delta_bz=float(self.P[1, 1]),
            covariance=self.P.copy(),
        )

    def _update(self, idx: int, z: float, meas_std: float) -> CalibrationState:
        # Predict (random-walk drift)
        self.P = self.P + self.Q
        # Update
        h = np.zeros(2); h[idx] = 1.0
        S = float(h @ self.P @ h) + float(meas_std) ** 2
        K = self.P @ h / S
        self.x = self.x + K * (float(z) - float(h @ self.x))
        self.P = (np.eye(2) - np.outer(K, h)) @ self.P
        st = self.state
        self.history.append(st.mean)
        return st

    def update_from_ramsey_measurement(self, observed_frequency_mhz: float, measurement_std_mhz: float = 0.2) -> CalibrationState:
        """Incorporate a Ramsey-frequency observation of ΔB_z."""
        return self._update(1, observed_frequency_mhz, measurement_std_mhz)

    def update_from_exchange_oscillation(self, observed_j_mhz: float, measurement_std_mhz: float = 0.4) -> CalibrationState:
        """Incorporate an exchange-oscillation observation of J₀."""
        return self._update(0, observed_j_mhz, measurement_std_mhz)


def simulate_drift_tracking(
    n_steps: int = 200,
    drift_rate_per_step: float = 0.05,
    measurement_std_mhz: float = 0.3,
    initial_delta_bz: float = 15.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Simulate a randomly drifting ΔB_z, track it with Ramsey updates, and
    compare the tracked RMS error with a static (never re-calibrated) estimate.
    """
    rng = np.random.default_rng(seed)
    truth = initial_delta_bz + np.cumsum(rng.normal(0.0, drift_rate_per_step, size=n_steps))
    cal = BayesianActiveCalibrator(initial_delta_bz=initial_delta_bz, drift_rate_per_step=drift_rate_per_step, seed=seed)
    est = np.zeros(n_steps)
    for k in range(n_steps):
        z = truth[k] + rng.normal(0.0, measurement_std_mhz)
        est[k] = cal.update_from_ramsey_measurement(z, measurement_std_mhz).estimated_delta_bz
    return {
        "truth_delta_bz": truth,
        "tracked_delta_bz": est,
        "rms_error_tracked": float(np.sqrt(np.mean((est - truth) ** 2))),
        "rms_error_static": float(np.sqrt(np.mean((initial_delta_bz - truth) ** 2))),
    }
