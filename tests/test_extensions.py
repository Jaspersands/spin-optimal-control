"""
Tests for the v0.3 extensions: valley leakage, derivative (DRAG-style)
corrections, Bayesian drift tracking, AWG exporters, and the CLI.
"""

import json
import csv

import numpy as np
import pytest

from spin_optimal_control.valley import SiliconValleyModel
from spin_optimal_control.drag import DRAGPulseSynthesizer
from spin_optimal_control.calibration import BayesianActiveCalibrator, simulate_drift_tracking
from spin_optimal_control.awg_export import export_awg_waveforms
from spin_optimal_control import __version__
from spin_optimal_control.cli import main as cli_main


def test_version():
    assert __version__ == "0.3.0"


# ----------------------------------------------------------------------------- #
# Valley
# ----------------------------------------------------------------------------- #
def test_valley_hamiltonian_is_hermitian_and_dimension_8():
    vm = SiliconValleyModel(valley_splitting_uev=100.0, inter_valley_soc_mhz=2.0)
    H = vm.get_valley_hamiltonian(j_val=25.0)
    assert H.shape == (8, 8)
    assert np.allclose(H, H.conj().T)
    assert np.isclose(vm.e_valley_mhz, 100.0 * 241.798935, rtol=1e-6)


def test_valley_leakage_decreases_with_splitting():
    t = (np.arange(60) + 0.5) * 0.5
    j_pulse = 30.0 * np.sin(np.pi * t / 30.0)
    leaks = []
    for ev in (30.0, 60.0, 120.0, 300.0):
        vm = SiliconValleyModel(valley_splitting_uev=ev, inter_valley_soc_mhz=50.0)
        res = vm.compute_valley_leakage(j_pulse, dt_ns=0.5, delta_bz=15.0)
        assert 0.0 <= res["final_valley_leakage"] <= 1.0
        assert len(res["leakage_vs_time"]) == 60
        leaks.append(res["max_valley_leakage"])
    assert leaks[0] > leaks[-1]
    assert all(leaks[i] >= leaks[i + 1] * 0.5 for i in range(len(leaks) - 1))  # broadly monotone


def test_valley_initial_state_is_singlet_ground_valley():
    vm = SiliconValleyModel(valley_splitting_uev=120.0, inter_valley_soc_mhz=0.0)
    psi0 = vm.initial_state(j_val=0.0, delta_bz=0.0)
    s = np.array([0, 1, -1, 0]) / np.sqrt(2)
    expected = np.kron(s, np.array([0.0, 1.0]))  # ground valley is the −E_v/2 eigenstate of τ_z
    assert np.isclose(abs(np.vdot(expected, psi0)), 1.0, atol=1e-9)


# ----------------------------------------------------------------------------- #
# DRAG-style derivative corrections
# ----------------------------------------------------------------------------- #
def test_drag_correction_vanishes_for_constant_pulse():
    drag = DRAGPulseSynthesizer(delta_bz_mhz=15.0, drag_coefficient=0.5)
    j = np.full(50, 20.0)
    in_phase, quad = drag.apply_drag_correction(j, dt_ns=0.4)
    assert np.allclose(in_phase, j)
    assert np.allclose(quad, 0.0)


def test_drag_quadrature_is_scaled_derivative():
    drag = DRAGPulseSynthesizer(delta_bz_mhz=10.0, drag_coefficient=1.0)
    t = np.linspace(0, 20, 81)
    j = 30.0 * np.sin(np.pi * t / 20.0)
    in_phase, quad = drag.apply_drag_correction(j, dt_ns=0.25)
    expected = -np.gradient(j, 0.25) / 10.0
    assert np.allclose(quad, expected)
    assert np.all(in_phase >= 0.0)
    assert len(in_phase) == len(quad) == 81


# ----------------------------------------------------------------------------- #
# Calibration
# ----------------------------------------------------------------------------- #
def test_kalman_calibrator_converges():
    cal = BayesianActiveCalibrator(initial_j0=20.0, initial_delta_bz=15.0, seed=0)
    rng = np.random.default_rng(1)
    for _ in range(30):
        cal.update_from_ramsey_measurement(16.2 + rng.normal(0, 0.15), measurement_std_mhz=0.15)
        cal.update_from_exchange_oscillation(21.5 + rng.normal(0, 0.3), measurement_std_mhz=0.3)
    st = cal.state
    assert abs(st.estimated_delta_bz - 16.2) < 0.15
    assert abs(st.estimated_j0 - 21.5) < 0.3
    assert st.var_delta_bz < 0.05
    assert st.covariance.shape == (2, 2)


def test_drift_tracking_beats_static_estimate():
    res = simulate_drift_tracking(n_steps=200, drift_rate_per_step=0.05, measurement_std_mhz=0.3, seed=5)
    assert res["rms_error_tracked"] < res["rms_error_static"]
    assert res["truth_delta_bz"].shape == (200,)


# ----------------------------------------------------------------------------- #
# AWG export
# ----------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", ["json", "csv", "qblox", "zi"])
def test_awg_export_formats(tmp_path, fmt):
    t = (np.arange(60) + 0.5) * 0.5
    j = 25.0 * np.sin(np.pi * t / 30.0)
    eps = np.log(np.maximum(j / 20.0, 1e-4))
    quad = np.gradient(j, 0.5)
    ext = {"json": "json", "csv": "csv", "qblox": "json", "zi": "csv"}[fmt]
    path = tmp_path / f"pulse.{ext}"
    data = export_awg_waveforms(t, j, eps, quad, sample_rate_gsps=2.0, export_format=fmt, file_path=str(path))
    assert path.exists()
    assert data["metadata"]["num_samples"] == 60
    assert data["metadata"]["sample_rate_gsps"] == 2.0
    if fmt == "qblox":
        payload = json.loads(path.read_text())
        assert "waveforms" in payload
        assert set(payload["waveforms"]) >= {"exchange_j_mhz", "detuning_eps_mv", "drag_quadrature"}
        assert payload["waveforms"]["exchange_j_mhz"]["index"] == 0
    if fmt in ("csv", "zi"):
        rows = list(csv.reader(path.open()))
        assert len(rows) == 61


def test_awg_export_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        export_awg_waveforms(np.arange(3.0), np.ones(3), np.ones(3), export_format="tek", file_path=str(tmp_path / "x"))


# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #
def test_cli_optimize_json(capsys, tmp_path):
    out = tmp_path / "pulse.json"
    rc = cli_main(["optimize", "--target", "cz", "--dbz", "30", "--duration", "40", "--steps", "40",
                   "--harmonics", "4", "--local-z", "--json", "--output", str(out)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "cz"
    assert payload["gate_fidelity"] > 0.99
    assert out.exists()


def test_cli_valley_and_noise_psd(capsys):
    assert cli_main(["valley", "--ev", "120", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "final_valley_leakage" in payload
    assert cli_main(["noise-psd", "--steps", "512", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["slope"] < 0


def test_cli_rb_json(capsys):
    assert cli_main(["rb", "--lengths", "1,2,4", "--sequences", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "decay_p" in payload
