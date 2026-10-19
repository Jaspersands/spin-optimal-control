"""
Candidate mechanisms for the gap between the modelled and the measured CZ error:
each term against a closed form or a direct propagation, switched off by default,
and the per-device numbers the README quotes.
"""

import json

import numpy as np
import pytest
import scipy.linalg

from spin_optimal_control.cli import main as cli_main
from spin_optimal_control.devices import _crossing, gap_analysis, load_devices
from spin_optimal_control.error_budget import (
    E_TO_AVG,
    GAP_TERMS,
    CZNoiseParameters,
    ExchangeVoltageMap,
    GapMechanisms,
    GateZeemanCoupling,
    _band_integral,
    cz_error_budget,
    cz_error_budget_for_pulse,
    distorted_pulse,
    irb_error_bound,
    residual_exchange_error,
    shaped_cz_pulse,
    zeeman_sigma_from_t2_star,
)
from spin_optimal_control.hamiltonian import HEISENBERG_EXCHANGE, SIGMA_I, SIGMA_Z
from spin_optimal_control.pulse_shaping import CZ_AREA_MHZ_NS
from spin_optimal_control.units import MHZ_NS_TO_RAD

DEVICES = load_devices()

NOISY = CZNoiseParameters(
    delta_ez_mhz=30.0,
    zeeman_sigma_mhz=(zeeman_sigma_from_t2_star(3.0), zeeman_sigma_from_t2_star(5.0)),
    exchange_sigma_rel=0.03,
    t1_us=(200.0, 300.0),
    t2_markov_us=(60.0, 90.0),
    f_low_hz=1e-2,
    f_ramsey_hz=2.5e5,
    temperature_mk=100.0,
    larmor_mhz=(4000.0, 4000.0),
)


@pytest.fixture(scope="module")
def gaps():
    return {k: gap_analysis(d, sweeps=False) for k, d in DEVICES.items()}


# --------------------------------------------------------------------------- #
# Switching
# --------------------------------------------------------------------------- #
def test_no_mechanisms_leave_the_budget_unchanged():
    b0 = cz_error_budget(NOISY, "cosine", 100.0)
    b1 = cz_error_budget(NOISY, "cosine", 100.0, mechanisms=GapMechanisms())
    assert b1.gap == {} and GapMechanisms().active() == ()
    assert b1.total == pytest.approx(b0.total, rel=1e-12) and b1.base_total == pytest.approx(b0.total, rel=1e-12)


def test_each_term_switches_on_alone_and_adds_to_the_total():
    m = GapMechanisms(gate_zeeman=GateZeemanCoupling.linear((0.2, 0.5)), zeeman_f_run_hz=1e-4,
                      exchange_f_run_hz=1e-4, bandwidth_mhz=50.0, j_off_mhz=0.05, idle_ns=200.0,
                      frequency_offset_mhz=(0.01, 0.02), exchange_offset_rel=0.01, calibration_residual=1e-4)
    assert m.active() == GAP_TERMS
    for name in GAP_TERMS:
        assert m.only(name).active() == (name,)
    with pytest.raises(ValueError):
        m.only("leakage")
    b = cz_error_budget(NOISY, "cosine", 100.0, mechanisms=m)
    assert set(b.gap) == set(GAP_TERMS)
    assert b.total == pytest.approx(b.base_total + sum(b.gap.values()), rel=1e-12)
    assert b.as_dict()["slow_zeeman"] == b.gap["slow_zeeman"]


# --------------------------------------------------------------------------- #
# Terms against closed forms
# --------------------------------------------------------------------------- #
def test_slow_noise_is_the_quasistatic_term_of_the_extended_band():
    """Extending a 1/f^α spectrum from f_low down to f_run adds A·∫f^-α df to the quasi-static variance only."""
    f_run = 1.0 / (6 * 3600.0)
    nm = NOISY.replace(zeeman_alpha=(1.0, 1.3), exchange_alpha=1.1)
    b = cz_error_budget(nm, "cosine", 100.0, mechanisms=GapMechanisms(zeeman_f_run_hz=f_run, exchange_f_run_hz=f_run))
    A = nm.psd_amplitudes()
    extra = [A[q] * _band_integral(f_run, nm.f_low_hz, nm.alphas[q]) for q in range(3)]
    wider = nm.replace(f_low_hz=f_run,
                       zeeman_sigma_mhz=tuple(np.sqrt(nm.zeeman_sigma_mhz[q] ** 2 + extra[q]) for q in range(2)),
                       exchange_sigma_rel=float(np.sqrt(nm.exchange_sigma_rel**2 + extra[2])))
    assert np.allclose(wider.psd_amplitudes(), A, rtol=1e-12)          # same spectrum, wider band
    bw = cz_error_budget(wider, "cosine", 100.0)
    assert b.gap["slow_zeeman"] == pytest.approx(bw.zeeman_quasistatic - b.zeeman_quasistatic, rel=1e-9)
    assert b.gap["slow_exchange"] == pytest.approx(bw.exchange_quasistatic - b.exchange_quasistatic, rel=1e-9)
    assert bw.zeeman_fast == pytest.approx(b.zeeman_fast, rel=1e-9)
    # nothing to add when the run is no longer than the Ramsey average
    b2 = cz_error_budget(nm, "cosine", 100.0, mechanisms=GapMechanisms(zeeman_f_run_hz=nm.f_low_hz))
    assert b2.gap["slow_zeeman"] == 0.0


def test_offsets_of_one_sigma_equal_the_quasistatic_terms():
    """A static offset of rms x gives the same first-order error as a quasi-static Gaussian of width x."""
    s1, s2 = NOISY.zeeman_sigma_mhz
    m = GapMechanisms(frequency_offset_mhz=(s1, s2), exchange_offset_rel=NOISY.exchange_sigma_rel)
    b = cz_error_budget(NOISY, "cosine", 100.0, mechanisms=m)
    assert b.gap["frequency_offset"] == pytest.approx(b.zeeman_quasistatic, rel=1e-12)
    assert b.gap["exchange_offset"] == pytest.approx(b.exchange_quasistatic, rel=1e-12)


def test_residual_exchange_is_a_conditional_phase():
    for j, t, dez in ((0.0588, 100.0, 103.0), (0.01, 2000.0, 396.0), (0.5, 200.0, 100.0)):
        phi = MHZ_NS_TO_RAD * j * t
        assert residual_exchange_error(dez, j, t) == pytest.approx(E_TO_AVG * np.sin(phi / 4) ** 2, rel=2e-3)
    assert residual_exchange_error(100.0, 0.0, 100.0) == 0.0
    b = cz_error_budget(NOISY, "cosine", 100.0, mechanisms=GapMechanisms(j_off_mhz=0.05))
    assert "residual_exchange" not in b.gap                              # no idle time, no term


def test_measured_calibration_residual_is_added_as_given():
    b = cz_error_budget(NOISY, "cosine", 100.0, mechanisms=GapMechanisms(calibration_residual=2.05e-4))
    assert b.gap == {"calibration_residual": 2.05e-4}


# --------------------------------------------------------------------------- #
# Exchange-gate voltage moving the qubit frequencies
# --------------------------------------------------------------------------- #
def _quasistatic_error(nm, j, dt, k, sigma, n_q=9):
    """Gauss–Hermite average of 1 − F(U(η), U(0)) for J(1+η) with the qubits shifted by k_j(t)·η."""
    Z1, Z2 = np.kron(SIGMA_Z, SIGMA_I), np.kron(SIGMA_I, SIGMA_Z)
    h = nm.hamiltonian()

    def U(eta):
        out = np.eye(4, dtype=complex)
        for i, jj in enumerate(j):
            H = h.get_hamiltonian_matrix(float(jj)) + eta * float(jj) * HEISENBERG_EXCHANGE / 4.0
            H = H + eta * (k[0][i] * Z1 + k[1][i] * Z2) / 2.0
            out = scipy.linalg.expm(-1j * MHZ_NS_TO_RAD * H * dt) @ out
        return out

    U0 = U(0.0)
    x, w = np.polynomial.hermite_e.hermegauss(n_q)
    err = [1.0 - (abs(np.trace(U0.conj().T @ U(sigma * xi))) ** 2 + 4.0) / 20.0 for xi in x]
    return float(np.dot(w, err) / np.sqrt(2 * np.pi))


def test_gate_zeeman_term_matches_direct_propagation():
    """Quasi-static part, cross term with the exchange noise included: steep spectrum, so the fast part is negligible."""
    nm = CZNoiseParameters(delta_ez_mhz=30.0, exchange_sigma_rel=0.01, exchange_alpha=3.0, f_low_hz=1e-2, f_ramsey_hz=1e5)
    j, dt = shaped_cz_pulse("cosine", 100.0, nm.delta_ez_mhz, n_steps=120)
    vm = ExchangeVoltageMap(kappa_per_v=24.2, j_ref_mhz=0.0588)
    gz = GateZeemanCoupling(beta_mhz=(-2.91, 67.2), gamma=1.2, voltage_map=vm)
    term = cz_error_budget_for_pulse(nm, j, dt, mechanisms=GapMechanisms(gate_zeeman=gz)).gap["gate_zeeman"]
    k = gz.sensitivity_mhz(j)
    direct = _quasistatic_error(nm, j, dt, k, 0.01) - _quasistatic_error(nm, j, dt, np.zeros_like(k), 0.01)
    assert term > 0
    assert term == pytest.approx(direct, rel=0.03)


def test_gate_zeeman_sensitivity_follows_the_fit():
    vm = ExchangeVoltageMap(kappa_per_v=24.2, j_ref_mhz=0.0588)
    assert vm.exchange(vm.voltage(10.0)) == pytest.approx(10.0, rel=1e-12)
    gz = GateZeemanCoupling(beta_mhz=(-2.91, 67.2), gamma=1.2, voltage_map=vm)
    v = vm.voltage(10.0)
    k = gz.sensitivity_mhz(np.array([10.0, 0.01]))
    assert k[1, 0] == pytest.approx(67.2 * 1.2 * v**0.2 / 24.2, rel=1e-12)
    assert k[0, 1] == 0.0 and k[1, 1] == 0.0                              # below J_ref the voltage is not raised
    lin = GateZeemanCoupling.linear((0.0, 2.0)).sensitivity_mhz(np.array([0.5, 5.0]))
    assert np.allclose(lin, [[0.0, 0.0], [2.0, 2.0]])
    off = cz_error_budget(NOISY, "cosine", 100.0, mechanisms=GapMechanisms(gate_zeeman=GateZeemanCoupling.linear((0.0, 0.0))))
    assert off.gap["gate_zeeman"] == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------------------- #
# Pulse distortion
# --------------------------------------------------------------------------- #
def test_low_pass_slice_averages_are_exact_and_keep_the_area():
    tau = 1e3 / (2 * np.pi * 20.0)
    j = np.full(40, 5.0)
    filt, ref, dt = distorted_pulse(j, 2.5, 20.0, max_dt_tau=10.0)          # coarse slices: still exact
    t0 = np.arange(len(filt)) * dt
    t1 = t0 + dt
    T = 40 * 2.5
    # response of a unit step that switches off at T, averaged over each slice
    def integral(t):
        on = np.minimum(t, T)
        rise = on - tau * (1 - np.exp(-on / tau))
        fall = np.where(t > T, tau * (1 - np.exp(-(T) / tau)) * (1 - np.exp(-(t - T) / tau)), 0.0)
        return rise + fall
    exact = (integral(t1) - integral(t0)) / dt
    exact *= CZ_AREA_MHZ_NS / (np.sum(exact) * dt)
    assert np.allclose(filt, exact, rtol=1e-9, atol=1e-12)
    assert np.sum(filt) * dt == pytest.approx(CZ_AREA_MHZ_NS) and np.sum(ref) * dt == pytest.approx(CZ_AREA_MHZ_NS)


def test_distortion_vanishes_at_high_bandwidth_and_breaks_a_synchronised_square_pulse():
    tn = DEVICES["tanttu2024"]
    nm = tn.noise_parameters()
    vm = tn.voltage_map()
    lo = cz_error_budget(nm, "square", tn.cz_time_ns, mechanisms=GapMechanisms(bandwidth_mhz=5.0, voltage_map=vm))
    hi = cz_error_budget(nm, "square", tn.cz_time_ns, mechanisms=GapMechanisms(bandwidth_mhz=300.0, voltage_map=vm))
    assert abs(hi.gap["distortion"]) < 1e-4
    assert lo.gap["distortion"] > 1e-3            # the filtered square pulse no longer closes the SWAP-like rotation


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
def test_irb_bound_follows_magesan_2012():
    r = irb_error_bound(0.9867, 0.9951)                                   # Noiri 2022
    p = 1 - 4 * (1 - 0.9867) / 3
    ratio = 1 - 4 * 0.0049 / 3
    e1 = 3 * (abs(p - ratio) + 1 - p) / 4
    e2 = 2 * 15 * (1 - p) / (16 * p) + 4 * np.sqrt(1 - p) * np.sqrt(15) / p
    assert r["p_ref"] == pytest.approx(p) and r["bound"] == pytest.approx(min(e1, e2))
    assert r["estimate"] == pytest.approx(0.0049) and r["lower"] == 0.0
    # with a reference error larger than the gate's, the first form is 2 r_ref − r_est: the reference sets the range
    assert r["bound"] == pytest.approx(2 * (1 - 0.9867) - 0.0049, rel=1e-9)
    assert irb_error_bound(0.9999, 0.999)["bound"] < irb_error_bound(0.99, 0.999)["bound"]


def test_crossing_interpolates_and_reports_misses():
    x = np.geomspace(1, 100, 5)
    assert _crossing(x, x**2, 50.0) == pytest.approx(np.sqrt(50.0), rel=1e-9)
    assert _crossing(x, -x, 1.0) is None


# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #
def test_gap_inputs_are_sourced():
    x, t = DEVICES["xue2022"], DEVICES["tanttu2024"]
    assert not x.assumed("barrier_beta_mhz_per_v_gamma") and not x.assumed("barrier_gamma")
    assert x.entry("benchmark_duration_h")["basis"] == "derived" and x.entry("gst_cz_coherent_infidelity")["basis"] == "derived"
    assert t.entry("benchmark_duration_h")["basis"] == "derived" and not t.assumed("j_gate_swing_mv")
    assert DEVICES["noiri2022"].entry("noise_record_h")["basis"] == "derived"
    assert DEVICES["mills2022"].entry("reference_clifford_fidelity")["basis"] == "derived"
    # Tanttu's J(v) through the 330.4 mV swing lands on the printed exchange-off value
    assert t.voltage_map().j_ref_mhz * 1e6 == pytest.approx(float(t.value("j_off_hz")), rel=0.03)


def test_gap_terms_per_device(gaps):
    xue, noiri, mills, tanttu = gaps["xue2022"], gaps["noiri2022"], gaps["mills2022"], gaps["tanttu2024"]
    assert xue["gap"] == pytest.approx(3.5e-3 - 1.3545e-4, rel=0.01)
    assert xue["terms"]["gate_zeeman"]["value"] == pytest.approx(3.65e-5, rel=0.03)
    assert xue["terms"]["slow_zeeman"]["value"] == pytest.approx(8.6e-6, rel=0.03)
    assert xue["terms"]["slow_exchange"]["value"] == pytest.approx(1.50e-5, rel=0.03)
    assert xue["terms"]["calibration_residual"]["status"] == "measured"
    assert xue["closed_share"] == pytest.approx(0.079, abs=0.003)
    assert xue["terms"]["exchange_offset"]["required"]["value"] == pytest.approx(0.082, rel=0.02)
    # with the barrier coupling the model lands near the authors' own simulation (2.2e-4)
    assert xue["model"] + xue["terms"]["gate_zeeman"]["value"] == pytest.approx(1.72e-4, rel=0.03)

    for k in ("gate_zeeman", "distortion", "residual_exchange"):
        assert noiri["terms"][k]["status"] == "not applicable"
    assert noiri["closed"] == 0.0 and noiri["bounds"]["drive"]["value"] == pytest.approx(1.6e-3, rel=1e-3)
    assert noiri["benchmark"]["irb"]["bound"] == pytest.approx(0.0217, rel=0.01)

    assert mills["closed"] == 0.0
    assert mills["benchmark"]["irb"]["bound"] == pytest.approx(0.134, rel=0.01)
    assert mills["terms"]["residual_exchange"]["time_required_ns"] == pytest.approx(2760.0, rel=0.02)

    assert tanttu["terms"]["slow_zeeman"]["status"] == "upper bound"
    assert tanttu["terms"]["slow_zeeman"]["value"] == pytest.approx(1.75e-3, rel=0.03)
    assert tanttu["closed_share"] == pytest.approx(0.127, abs=0.005)
    assert tanttu["benchmark"]["irb"]["bound"] == pytest.approx(0.169, rel=0.01)
    for g in gaps.values():
        assert g["remaining"] > 0.85 * g["gap"]                             # nothing published closes most of it
        assert g["closed"] == pytest.approx(sum(t["value"] for t in g["terms"].values()
                                                if t["status"] in ("modelled", "upper bound", "measured")))


def test_gate_zeeman_sweep_where_unpublished():
    g = gap_analysis(DEVICES["tanttu2024"], n_points=5)
    t = g["terms"]["gate_zeeman"]
    assert t["status"] == "not published" and len(t["sweep"]["x"]) == 5
    y = np.asarray(t["sweep"]["y"])
    assert y[-1] > y[-2] > 0                                               # grows with the coupling
    assert g["terms"]["distortion"]["required"]["reached"]


def test_cli_devices_gap_table(capsys):
    assert cli_main(["devices", "--gap", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    rows = {r["key"]: r for r in payload["devices"]}
    assert rows["xue2022"]["gap"]["terms"]["gate_zeeman"]["status"] == "modelled"
    assert cli_main(["devices", "--gap"]) == 0
    out = capsys.readouterr().out
    assert "closed by published values" in out and "Magesan" in out
