"""
Arbitrary-waveform-generator exporters.

Waveforms are resampled to the instrument sample rate and written as

* ``json``  – generic channel table with metadata,
* ``csv``   – one row per sample (time, J, ε[, quadrature]),
* ``qblox`` – Qblox Q1ASM-style ``{"waveforms": {name: {"data": [...], "index": i}}}``,
* ``zi``    – Zurich Instruments CSV (header row + one column per channel).
"""

from __future__ import annotations

import csv
import json
from typing import Any, Dict, Optional

import numpy as np

SUPPORTED_FORMATS = ("json", "csv", "qblox", "zi")


def export_awg_waveforms(
    time_grid: np.ndarray,
    j_pulse: np.ndarray,
    detuning_pulse: np.ndarray,
    quadrature_drag: Optional[np.ndarray] = None,
    sample_rate_gsps: float = 1.0,
    export_format: str = "json",
    file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resample the waveforms to ``sample_rate_gsps`` (GSa/s) and optionally write
    them in ``export_format``. Returns the resampled data in all cases.
    """
    fmt = export_format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"export_format must be one of {SUPPORTED_FORMATS}, got {export_format!r}")

    t = np.asarray(time_grid, dtype=float)
    # The grid holds slice midpoints; the waveform window is [t0 - dt/2, tN + dt/2].
    dt_in = float(t[1] - t[0]) if len(t) > 1 else 1.0 / sample_rate_gsps
    t_start = float(t[0]) - 0.5 * dt_in
    duration = len(t) * dt_in
    n_samples = max(int(np.round(duration * sample_rate_gsps)), 1)
    t_res = t_start + (np.arange(n_samples) + 0.5) / sample_rate_gsps

    j_res = np.interp(t_res, t, np.asarray(j_pulse, dtype=float))
    eps_res = np.interp(t_res, t, np.asarray(detuning_pulse, dtype=float))
    q_res = None if quadrature_drag is None else np.interp(t_res, t, np.asarray(quadrature_drag, dtype=float))

    channels: Dict[str, list] = {
        "exchange_j_mhz": [float(x) for x in np.round(j_res, 6)],
        "detuning_eps_mv": [float(x) for x in np.round(eps_res, 6)],
    }
    if q_res is not None:
        channels["drag_quadrature"] = [float(x) for x in np.round(q_res, 6)]

    data: Dict[str, Any] = {
        "metadata": {
            "format": fmt,
            "sample_rate_gsps": float(sample_rate_gsps),
            "duration_ns": duration,
            "num_samples": n_samples,
            "time_ns": [float(x) for x in np.round(t_res, 6)],
            "instrument_target": {
                "json": "generic", "csv": "generic",
                "qblox": "Qblox QCM / Cluster", "zi": "Zurich Instruments HDAWG",
            }[fmt],
        },
        "channels": channels,
    }

    if file_path:
        if fmt == "json":
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        elif fmt == "qblox":
            payload = {
                "waveforms": {name: {"data": vals, "index": i} for i, (name, vals) in enumerate(channels.items())},
                "metadata": {k: v for k, v in data["metadata"].items() if k != "time_ns"},
            }
            with open(file_path, "w") as f:
                json.dump(payload, f, indent=2)
        else:  # csv / zi
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time_ns", *channels.keys()])
                cols = list(channels.values())
                for i in range(n_samples):
                    writer.writerow([float(np.round(t_res[i], 6))] + [c[i] for c in cols])
    return data
