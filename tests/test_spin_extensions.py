"""
Tests for Project 1 Extensions: Valley splitting, DRAG, Bayesian calibration, and AWG export.
"""

import pytest
import numpy as np
from spin_optimal_control.valley import SiliconValleyModel
from spin_optimal_control.drag import DRAGPulseSynthesizer
from spin_optimal_control.calibration import BayesianActiveCalibrator
from spin_optimal_control.awg_export import export_awg_waveforms


def test_valley_hamiltonian_and_leakage():
    vm = SiliconValleyModel(valley_splitting_uev=100.0, inter_valley_soc_mhz=2.0)
    H_8 = vm.get_valley_hamiltonian(j_val=25.0)
    assert H_8.shape == (8, 8)
    assert np.allclose(H_8, H_8.conj().T), "8x8 Valley Hamiltonian must be Hermitian"

    j_pulse = np.array([5.0, 15.0, 30.0, 20.0, 5.0])
    res = vm.compute_valley_leakage(j_pulse, dt=0.2)
    assert "final_valley_leakage" in res
    assert 0.0 <= res["final_valley_leakage"] <= 1.0 + 1e-6


def test_drag_pulse_correction():
    drag = DRAGPulseSynthesizer(delta_bz_mhz=15.0, drag_coefficient=0.5)
    t = np.linspace(0, 20, 50)
    j_nom = 30.0 * np.sin(np.pi * t / 20.0)
    dt = 20.0 / 50.0

    in_phase, quad = drag.apply_drag_correction(j_nom, dt)
    assert len(in_phase) == 50
    assert len(quad) == 50
    assert np.all(in_phase >= 0.0)


def test_bayesian_active_calibration():
    cal = BayesianActiveCalibrator(initial_j0=20.0, initial_delta_bz=15.0)
    
    # Simulate series of noisy Ramsey measurements with true Delta_Bz = 16.2
    for _ in range(10):
        meas = 16.2 + np.random.normal(0, 0.15)
        st = cal.update_from_ramsey_measurement(meas, measurement_std_mhz=0.15)
    
    assert abs(st.estimated_delta_bz - 16.2) < 0.25
    assert st.var_delta_bz < 0.1 # Variance should decrease significantly


def test_awg_export_formats(tmp_path):
    t_grid = np.linspace(0, 30, 60)
    j_p = np.full(60, 25.0)
    eps_p = np.full(60, 1.2)
    
    json_file = str(tmp_path / "pulse.json")
    csv_file = str(tmp_path / "pulse.csv")
    
    data_json = export_awg_waveforms(t_grid, j_p, eps_p, export_format="json", file_path=json_file)
    assert "channels" in data_json
    assert data_json["metadata"]["num_samples"] == 30
    
    data_csv = export_awg_waveforms(t_grid, j_p, eps_p, export_format="csv", file_path=csv_file)
    assert "metadata" in data_csv
