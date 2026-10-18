"""
Closed-loop Bayesian (Kalman) tracking of drifting device parameters.

A two-dimensional linear Kalman filter tracks (J₀, ΔB_z) under random-walk
drift, updated by two kinds of calibration experiments:

* Ramsey oscillations: a direct noisy observation of ΔB_z,
* exchange oscillations at a reference detuning: an observation of J₀.

``recalibration_interval`` uses the same filter to estimate how often a CZ
has to be recalibrated against drift of J₀ and ΔE_z.
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

    def predict(self) -> None:
        """Random-walk prediction step: add the process noise Q once."""
        self.P = self.P + self.Q

    def _update(self, idx: int, z: float, meas_std: float, predict: bool = True) -> CalibrationState:
        if predict:
            self.predict()
        h = np.zeros(2); h[idx] = 1.0
        S = float(h @ self.P @ h) + float(meas_std) ** 2
        K = self.P @ h / S
        self.x = self.x + K * (float(z) - float(h @ self.x))
        self.P = (np.eye(2) - np.outer(K, h)) @ self.P
        st = self.state
        self.history.append(st.mean)
        return st

    def update_from_ramsey_measurement(self, observed_frequency_mhz: float, measurement_std_mhz: float = 0.2,
                                       predict: bool = True) -> CalibrationState:
        """Incorporate a Ramsey-frequency observation of ΔB_z (after a prediction step unless ``predict=False``)."""
        return self._update(1, observed_frequency_mhz, measurement_std_mhz, predict)

    def update_from_exchange_oscillation(self, observed_j_mhz: float, measurement_std_mhz: float = 0.4,
                                         predict: bool = True) -> CalibrationState:
        """Incorporate an exchange-oscillation observation of J₀ (after a prediction step unless ``predict=False``)."""
        return self._update(0, observed_j_mhz, measurement_std_mhz, predict)


def longest_interval(intervals_h: np.ndarray, infidelity: np.ndarray, target_fidelity: float) -> float:
    """
    Longest interval whose infidelity stays within ``1 − target_fidelity``,
    interpolated in log(interval) at the crossing; ``nan`` if none does.
    Assumes the infidelity grows with the interval.
    """
    intervals_h, curve = np.asarray(intervals_h, dtype=float), np.asarray(infidelity, dtype=float)
    budget = 1.0 - float(target_fidelity)
    ok = np.where(curve <= budget)[0]
    if len(ok) == 0:
        return float("nan")
    k = ok[-1]
    if k + 1 < len(curve):
        x0, x1 = np.log(intervals_h[k]), np.log(intervals_h[k + 1])
        y0, y1 = curve[k], curve[k + 1]
        return float(np.exp(x0 + (budget - y0) * (x1 - x0) / (y1 - y0)))
    return float(intervals_h[k])


def recalibration_interval(
    baseline_infidelity: float,
    gate_time_ns: float,
    j_on_mhz: float,
    target_fidelity: float = 0.999,
    drift_j_rel_per_sqrt_h: float = 0.01,
    drift_dez_mhz_per_sqrt_h: float = 0.02,
    meas_std_j_rel: float = 0.005,
    meas_std_dez_mhz: float = 0.005,
    independent_qubit_drift: bool = True,
    intervals_h: Optional[np.ndarray] = None,
    horizon_h: float = 48.0,
    n_runs: int = 100,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """
    How often must a CZ be recalibrated to stay above ``target_fidelity``?

    J₀ (relative) and ΔE_z random-walk with the given rates. Every ``interval``
    one exchange-oscillation and one Ramsey measurement are taken, with errors
    ``meas_std_*``. Two strategies use them:

    * plain recalibration: the gate is set to the latest measurement;
    * Kalman tracking: ``BayesianActiveCalibrator`` with one prediction step
      per interval whose process noise equals the drift over that interval,
      then both measurements.

    Between updates the calibration error grows by the drift. A relative J
    error e_J leaves a conditional-phase error π·e_J, and a ΔE_z error e_B
    rotates the two qubits by ±π·10⁻³·e_B·T, so

        1 − F = baseline + (4/5)[π² e_J²/16 + k (2π·10⁻³ T)² e_B²/8]

    (average gate infidelity, time-averaged over each interval). With
    ``independent_qubit_drift`` (k = 2) the two qubit frequencies drift
    independently, each by 1/√2 of the ΔE_z rate, and each is measured with
    1/√2 of the ΔE_z error; their common mode then shifts the Z phases as much
    as the difference does, which doubles the frequency term exactly. k = 1
    keeps only the ΔE_z drift.

    ``baseline_infidelity`` is the drift-free error of the gate. Returns the
    mean infidelity per interval for both strategies and the longest interval
    that meets the target (``nan`` if none does). Each run spans ``horizon_h``
    but at least 12 and at most 120 updates.
    """
    if intervals_h is None:
        intervals_h = np.geomspace(0.05, 24.0, 18)
    intervals_h = np.asarray(intervals_h, dtype=float)
    c_j = 0.8 * np.pi**2 / 16.0
    c_b = (2.0 if independent_qubit_drift else 1.0) * 0.8 * (2.0 * np.pi * 1e-3 * float(gate_time_ns)) ** 2 / 8.0
    rng = np.random.default_rng(seed)
    plain, kalman = [], []
    for dt in intervals_h:
        n_up = int(min(max(np.ceil(horizon_h / dt), 12), 120))   # the filter settles within a few updates
        sj = drift_j_rel_per_sqrt_h * np.sqrt(dt) * j_on_mhz          # MHz per interval
        sb = drift_dez_mhz_per_sqrt_h * np.sqrt(dt)
        rj, rb = meas_std_j_rel * j_on_mhz, meas_std_dez_mhz
        err_p, err_k = [], []
        for _ in range(int(n_runs)):
            truth = np.array([j_on_mhz, 0.0])
            cal = BayesianActiveCalibrator(initial_j0=j_on_mhz, initial_delta_bz=0.0, prior_std_j0=rj, prior_std_db=rb)
            cal.Q = np.diag([sj**2, sb**2])
            e2p = e2k = 0.0
            for k in range(n_up):
                zj = truth[0] + rng.normal(0.0, rj)
                zb = truth[1] + rng.normal(0.0, rb)
                if k > 0:
                    cal.predict()                                     # drift since the last interval, once
                cal.update_from_exchange_oscillation(zj, rj, predict=False)
                cal.update_from_ramsey_measurement(zb, rb, predict=False)
                step = rng.normal(0.0, [sj, sb])
                if k >= 3:                                            # skip the Kalman burn-in
                    for est, acc in (((zj, zb), "p"), ((cal.x[0], cal.x[1]), "k")):
                        e0 = truth - np.asarray(est)
                        # mean square error over the interval: error at its start plus half the drift variance
                        cost = c_j * (e0[0] ** 2 + sj**2 / 2) / j_on_mhz**2 + c_b * (e0[1] ** 2 + sb**2 / 2)
                        if acc == "p":
                            e2p += cost
                        else:
                            e2k += cost
                truth = truth + step
            err_p.append(e2p / (n_up - 3))
            err_k.append(e2k / (n_up - 3))
        plain.append(baseline_infidelity + float(np.mean(err_p)))
        kalman.append(baseline_infidelity + float(np.mean(err_k)))
    plain, kalman = np.array(plain), np.array(kalman)
    return {
        "intervals_h": intervals_h,
        "infidelity_plain": plain,
        "infidelity_kalman": kalman,
        "target_fidelity": float(target_fidelity),
        "interval_plain_h": longest_interval(intervals_h, plain, target_fidelity),
        "interval_kalman_h": longest_interval(intervals_h, kalman, target_fidelity),
        "floor_plain": float(baseline_infidelity + c_j * meas_std_j_rel**2 + c_b * meas_std_dez_mhz**2),
    }


def drift_rate_for_infidelity(excess_infidelity: float, interval_h: float, meas_std_j_rel: float = 0.0) -> float:
    """
    Random-walk rate of J (relative, per √h) that, with plain recalibration
    every ``interval_h``, adds ``excess_infidelity`` on average:
    (4/5)(π²/16)(e_meas² + D²·interval/2) = excess. ``nan`` if the measurement
    error alone exceeds it.
    """
    c_j = 0.8 * np.pi**2 / 16.0
    d2 = 2.0 * (float(excess_infidelity) / c_j - float(meas_std_j_rel) ** 2) / float(interval_h)
    return float(np.sqrt(d2)) if d2 > 0 else float("nan")


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
