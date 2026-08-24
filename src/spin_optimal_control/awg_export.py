"""
Hardware AWG Waveform Exporters.

Exports optimized pulse waveforms into standard formats:
1. JSON / CSV definition tables.
2. Qblox / Zurich Instruments / Keysight compatible sample arrays.
"""

from __future__ import annotations
import json
import numpy as np
from typing import Dict, Any, Optional


def export_awg_waveforms(
    time_grid: np.ndarray,
    j_pulse: np.ndarray,
    detuning_pulse: np.ndarray,
    quadrature_drag: Optional[np.ndarray] = None,
    sample_rate_gsps: float = 1.0, # 1 GSa/s
    export_format: str = "json",    # 'json', 'csv', 'qblox'
    file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Exports continuous waveforms resampled to target AWG sample rate.
    """
    t_max = time_grid[-1]
    n_samples = int(np.round(t_max * sample_rate_gsps))
    resampled_t = np.linspace(0.0, t_max, n_samples)

    # Linear interpolation
    j_resampled = np.interp(resampled_t, time_grid, j_pulse)
    eps_resampled = np.interp(resampled_t, time_grid, detuning_pulse)

    data = {
        "metadata": {
            "sample_rate_gsps": sample_rate_gsps,
            "duration_ns": float(t_max),
            "num_samples": n_samples,
            "instrument_target": "Qblox / Zurich Instruments / Keysight AWG",
        },
        "channels": {
            "ch1_exchange_j_mhz": list(np.round(j_resampled, 5)),
            "ch2_detuning_eps_mv": list(np.round(eps_resampled, 5)),
        }
    }

    if quadrature_drag is not None:
        q_resampled = np.interp(resampled_t, time_grid, quadrature_drag)
        data["channels"]["ch3_drag_quadrature"] = list(np.round(q_resampled, 5))

    if file_path:
        if export_format == "json":
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        elif export_format == "csv":
            import csv
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                headers = ["time_ns", "exchange_j_mhz", "detuning_eps_mv"]
                if quadrature_drag is not None:
                    headers.append("drag_quadrature")
                writer.writerow(headers)
                for i in range(n_samples):
                    row = [resampled_t[i], j_resampled[i], eps_resampled[i]]
                    if quadrature_drag is not None:
                        row.append(q_resampled[i])
                    writer.writerow(row)

    return data
