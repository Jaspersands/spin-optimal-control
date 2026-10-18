"""
Tests for the v0.3 extensions: valley leakage, window-shaped adiabatic CZ
Bayesian drift tracking, AWG exporters, and the CLI.
"""

import json
import csv

import numpy as np
import pytest

from spin_optimal_control.valley import SiliconValleyModel
from spin_optimal_control.pulse_shaping import AdiabaticCZDesigner, WINDOWS, conditional_phase, window
from spin_optimal_control import SiliconSpinHamiltonian
from spin_optimal_control.calibration import BayesianActiveCalibrator, recalibration_interval, simulate_drift_tracking
from spin_optimal_control.awg_export import export_awg_waveforms
from spin_optimal_control import __version__
from spin_optimal_control.cli import main as cli_main


def test_version():
    assert __version__ == "0.5.0"


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
# Window-shaped adiabatic CZ
# ----------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", WINDOWS)
def test_window_is_unit_peak_and_calibrated_to_a_pi_phase(shape):
    w = window(shape, 400)
    assert w.max() <= 1.0 + 1e-12 and w.min() >= 0.0
    cz = AdiabaticCZDesigner(SiliconSpinHamiltonian(delta_bz=20.0)).calibrate(shape, 100.0)
    assert abs(abs(cz.conditional_phase) - np.pi) < 1e-9
    assert abs(np.sum(cz.j_pulse) * cz.dt_ns - 500.0) < 1e-9


def test_conditional_phase_depends_only_on_pulse_area():
    """The (|↑↓⟩,|↓↑⟩) block is a phase times SU(2), so φ = −2π·1e-3·∫J dt for any shape and ΔBz."""
    rng = np.random.default_rng(0)
    from spin_optimal_control import ExchangeDynamics
    for dbz in (0.0, 7.0, 40.0):
        dyn = ExchangeDynamics(SiliconSpinHamiltonian(delta_bz=dbz))
        j = np.abs(rng.normal(5.0, 3.0, size=80))
        U = dyn.propagate_unitary(j, 0.7)
        expected = np.angle(np.exp(-2j * np.pi * 1e-3 * np.sum(j) * 0.7))
        assert abs(np.angle(np.exp(1j * (conditional_phase(U) - expected)))) < 1e-9


def test_smooth_windows_suppress_nonadiabatic_leakage():
    d = AdiabaticCZDesigner(SiliconSpinHamiltonian(delta_bz=20.0))
    for T in (100.0, 200.0):
        sq, cos = d.calibrate("square", T), d.calibrate("cosine", T)
        assert cos.swap_leakage < sq.swap_leakage / 30
        assert cos.infidelity < sq.infidelity / 30
    # longer (more adiabatic) pulses are better for every smooth window
    for s in ("cosine", "blackman"):
        assert d.calibrate(s, 200.0).infidelity < d.calibrate(s, 100.0).infidelity < d.calibrate(s, 50.0).infidelity
    # and a larger gap helps: adiabaticity is set by ΔBz·T
    d60 = AdiabaticCZDesigner(SiliconSpinHamiltonian(delta_bz=60.0))
    assert d60.calibrate("cosine", 100.0).infidelity < d.calibrate("cosine", 100.0).infidelity


def test_cz_fidelity_matches_numerical_local_z_optimum():
    from spin_optimal_control import ExchangeDynamics
    d = AdiabaticCZDesigner(SiliconSpinHamiltonian(delta_bz=20.0))
    cz = d.calibrate("square", 50.0)
    U = d._unitary("square", 50.0, cz.j_max_mhz, 0.5)
    f_pro, _ = ExchangeDynamics.gate_fidelity_local_z_free(U, ExchangeDynamics.target_gate_cz())
    assert abs(cz.fidelity - (4 * f_pro + 1) / 5) < 1e-6


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


def test_kalman_adds_the_drift_once_per_interval():
    """One prediction per interval: the posterior variance converges to the scalar Riccati fixed point."""
    Q, R = 0.04, 0.25
    cal = BayesianActiveCalibrator(initial_j0=10.0, prior_std_j0=1.0, seed=0)
    cal.Q = np.diag([Q, Q])
    for _ in range(200):
        cal.predict()
        cal.update_from_exchange_oscillation(10.0, np.sqrt(R), predict=False)
    assert cal.P[0, 0] == pytest.approx((-Q + np.sqrt(Q**2 + 4 * Q * R)) / 2, rel=1e-9)


def test_recalibration_interval_matches_the_closed_form():
    """
    In steady state the mean excess error over an interval is c·(e₀² + drift²/2), where e₀² is the
    measurement variance R for plain recalibration and the Riccati posterior for the Kalman tracker.
    Xue 2022 inputs; the frequency term counts twice for independent qubit drift.
    """
    base, t_ns, j_on = 1.3545e-4, 100.0, 10.06
    dj, db, mj, mb = 0.01, 0.02, 0.02, 0.005
    iv = np.array([0.05, 2.53])
    r = recalibration_interval(base, t_ns, j_on, drift_j_rel_per_sqrt_h=dj, drift_dez_mhz_per_sqrt_h=db,
                               meas_std_j_rel=mj, meas_std_dez_mhz=mb, intervals_h=iv, n_runs=200, seed=3)
    c_j = 0.8 * np.pi**2 / 16
    c_b = 2 * 0.8 * (2 * np.pi * 1e-3 * t_ns) ** 2 / 8

    def posterior(Q, R):
        return (-Q + np.sqrt(Q**2 + 4 * Q * R)) / 2

    for i, dt in enumerate(iv):
        Qj, Qb, Rj, Rb = dj**2 * dt, db**2 * dt, mj**2, mb**2
        plain = c_j * (Rj + Qj / 2) + c_b * (Rb + Qb / 2)
        kalman = c_j * (posterior(Qj, Rj) + Qj / 2) + c_b * (posterior(Qb, Rb) + Qb / 2)
        assert r["infidelity_plain"][i] - base == pytest.approx(plain, rel=0.05), dt
        assert r["infidelity_kalman"][i] - base == pytest.approx(kalman, rel=0.06), dt
    # the assumed 2 % J calibration error alone exceeds the drift-free gate error
    assert c_j * mj**2 > base


def test_recalibration_interval_limits():
    """Short intervals reach the measurement-noise floor, long ones the drift; tracking never hurts."""
    base = 1e-4
    r = recalibration_interval(base, 100.0, 10.0, target_fidelity=0.9995, drift_j_rel_per_sqrt_h=0.01,
                               drift_dez_mhz_per_sqrt_h=0.02, meas_std_j_rel=0.02, meas_std_dez_mhz=0.005,
                               intervals_h=np.geomspace(0.01, 48.0, 12), n_runs=30, seed=2)
    plain, kalman = r["infidelity_plain"], r["infidelity_kalman"]
    assert plain[0] == pytest.approx(r["floor_plain"], rel=0.25)
    assert np.all(kalman <= plain * 1.02) and kalman[0] < 0.5 * plain[0]
    assert plain[-1] > 1 - r["target_fidelity"] > base
    assert r["interval_kalman_h"] >= r["interval_plain_h"] > 0
    # without measurement noise there is nothing to filter
    q = recalibration_interval(base, 100.0, 10.0, meas_std_j_rel=1e-6, meas_std_dez_mhz=1e-6,
                               intervals_h=np.array([0.5, 4.0]), n_runs=20, seed=2)
    assert q["infidelity_kalman"] == pytest.approx(q["infidelity_plain"], rel=1e-3)


# ----------------------------------------------------------------------------- #
# AWG export
# ----------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", ["json", "csv", "qblox", "zi"])
def test_awg_export_formats(tmp_path, fmt):
    t = (np.arange(60) + 0.5) * 0.5
    j = 25.0 * np.sin(np.pi * t / 30.0)
    eps = np.log(np.maximum(j / 20.0, 1e-4))
    plunger = -0.3 * j
    ext = {"json": "json", "csv": "csv", "qblox": "json", "zi": "csv"}[fmt]
    path = tmp_path / f"pulse.{ext}"
    data = export_awg_waveforms(t, j, eps, {"plunger_comp_mv": plunger}, sample_rate_gsps=2.0, export_format=fmt, file_path=str(path))
    assert path.exists()
    assert data["metadata"]["num_samples"] == 60
    assert data["metadata"]["sample_rate_gsps"] == 2.0
    if fmt == "qblox":
        payload = json.loads(path.read_text())
        assert "waveforms" in payload
        assert set(payload["waveforms"]) >= {"exchange_j_mhz", "detuning_eps_mv", "plunger_comp_mv"}
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
