"""
Published-device data: every number has a source, assumed values say why,
and the headline comparison numbers stay where the README puts them.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from spin_optimal_control.cli import main as cli_main
from spin_optimal_control.devices import (
    DATA_FILE,
    best_gate_times,
    comparison_table,
    iter_entries,
    load_devices,
)
from spin_optimal_control.error_budget import (
    CZ_AREA_MHZ_NS,
    combined_simulation,
    echo_time_from_1f,
    psd_amplitude,
    shortest_gate_time,
    square_sync_times,
    zeeman_sigma_from_t2_star,
)
from spin_optimal_control.pulse_shaping import window

DEVICES = load_devices()
REQUIRED = ["delta_ez_mhz", "gate_time_ns", "model_shape", "exchange_sigma_rel", "t1_us",
            "electron_temperature_mk", "measured_fidelity", "ramsey_averaging_s", "j_max_shown_mhz"]

# README table: total modelled infidelity at the published gate time (fitted 1/f^α Zeeman spectra)
HEADLINE = {
    "xue2022": 1.3545e-4,
    "noiri2022": 2.3256e-4,
    "mills2022": 4.0049e-4,
    "tanttu2024": 1.6649e-3,
}


def test_data_file_ships_with_the_package():
    assert DATA_FILE.is_file()
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert 'spin_optimal_control = ["data/*.json"]' in pyproject


def test_four_devices_load_in_file_order():
    raw = json.loads(DATA_FILE.read_text())
    assert list(DEVICES) == list(raw["devices"])
    assert list(DEVICES) == ["xue2022", "noiri2022", "mills2022", "tanttu2024"]


@pytest.mark.parametrize("key", list(DEVICES))
def test_every_value_has_a_source_and_assumptions_say_why(key):
    dev = DEVICES[key]
    for name, e in iter_entries(dev):
        assert "value" in e and e["value"] is not None, (key, name)
        assert isinstance(e.get("source"), str) and len(e["source"]) > 8, (key, name)
        assert isinstance(e.get("assumed"), bool), (key, name)
        if e["assumed"]:
            assert e.get("basis") in ("derived", "assumed", "lower bound", "upper bound"), (key, name)
            assert isinstance(e.get("reason"), str) and len(e["reason"]) > 10, (key, name)
        else:
            assert "basis" not in e and "reason" not in e, (key, name)
    for name in REQUIRED:
        dev.entry(name)
    assert dev.has("zeeman_sigma_khz") or dev.has("t2_star_us")
    assert dev.method and 0 < dev.measured_uncertainty < 0.01
    assert dev.citation and dev.arxiv and dev.doi
    assert "arXiv" in dev.values_from or "published" in dev.values_from     # which version the numbers come from
    assert dev.arxiv_title


def test_tanttu_values_name_the_arxiv_version_they_were_read_from():
    d = DEVICES["tanttu2024"]
    assert "v3" in d.values_from
    for name, e in d.params.items():
        if "Extended Data Table 2" in e["source"]:
            assert "arXiv v3" in e["source"] and "published: Extended Data Table 1" in e["source"], name


@pytest.mark.parametrize("key", [k for k, d in DEVICES.items() if d.has("f_q1_ghz")])
def test_frequency_difference_matches_the_qubit_frequencies(key):
    d = DEVICES[key]
    assert d.value("delta_ez_mhz") == pytest.approx(1e3 * abs(d.value("f_q1_ghz") - d.value("f_q2_ghz")), abs=1.0)


@pytest.mark.parametrize("key", ["xue2022", "tanttu2024"])
def test_published_exchange_and_time_give_a_cz_area(key):
    """J_on · T · mean(window) = 500 MHz·ns: the published numbers give a π conditional phase with the model window."""
    d = DEVICES[key]
    area = d.value("j_on_mhz") * d.cz_time_ns * np.mean(window(d.model_shape, 1000, d.tukey_alpha))
    assert area == pytest.approx(CZ_AREA_MHZ_NS, rel=0.02)
    assert d.model_peak_j_mhz == pytest.approx(d.value("j_on_mhz"), rel=0.02)


def test_mills_tukey_window_is_squeezed_into_the_published_time():
    """12.5 MHz × 40 ns is 500 MHz·ns for a flat pulse; the assumed Tukey α = 0.2 (mean 0.9) gives 450, so the model peaks at 13.9 MHz."""
    d = DEVICES["mills2022"]
    area = d.value("j_on_mhz") * d.cz_time_ns * np.mean(window(d.model_shape, 1000, d.tukey_alpha))
    assert area == pytest.approx(450.0, rel=0.01)
    assert d.model_peak_j_mhz == pytest.approx(13.9, abs=0.05)
    assert d.j_cap_mhz == pytest.approx(d.model_peak_j_mhz, rel=1e-6)     # the ceiling is the model's own peak


def test_exchange_ceiling_never_excludes_the_published_gate():
    for d in DEVICES.values():
        assert d.j_cap_mhz >= d.value("j_max_shown_mhz")
        assert shortest_gate_time(d.model_shape, d.j_cap_mhz, d.tukey_alpha) <= d.cz_time_ns * (1 + 1e-9)


# README table: measured / modelled infidelity. Pinned data, not an invariant: nothing in the model forbids exceeding a measurement.
GAP = {"xue2022": 26, "noiri2022": 21, "mills2022": 4.7, "tanttu2024": 9.5}


def test_gap_ratios_are_the_readme_numbers():
    for row in comparison_table(DEVICES):
        assert row["gap_ratio"] == pytest.approx(GAP[row["key"]], rel=0.03), row["device"]
        inside = row["measured_infidelity"] - row["measured_uncertainty"] <= row["model_infidelity"]
        assert inside == (row["key"] == "mills2022"), row["device"]      # only Mills's error bar reaches the model


@pytest.mark.parametrize("key", list(HEADLINE))
def test_headline_numbers(key):
    b = DEVICES[key].budget()
    assert b.total == pytest.approx(HEADLINE[key], rel=0.01)
    c = sorted(b.contributions.values(), reverse=True)
    if c[0] > 2 * c[1]:                                          # only where the dominant term is clear
        assert b.dominant == "zeeman"


def test_alpha_fit_converges_for_every_reported_echo():
    """The root-finder inverted the closed-form echo time exactly (see the next test for an independent check)."""
    for d in DEVICES.values():
        nm = d.noise_parameters()
        for q, (fit, echo) in enumerate(zip(d.zeeman_fit(), d.echo_times_us())):
            if echo is None:
                assert fit["status"] == "no echo" and nm.zeeman_alpha[q] == 1.0
                continue
            assert fit["status"] == "fitted", (d.key, q)
            assert 0.5 < nm.zeeman_alpha[q] < 2.5
            t2 = echo_time_from_1f(nm.zeeman_sigma_mhz[q], nm.f_low_hz, nm.f_ramsey_hz, nm.zeeman_alpha[q])
            assert t2 == pytest.approx(echo, rel=1e-4), (d.key, q)
            assert not np.isfinite(nm.t2_markov_us[q]) or nm.t2_markov_us[q] == pytest.approx(2 * nm.t1_us[q])


def _phase_variance(A_hz2, alpha, f_low_hz, t_s, kind):
    """⟨φ²⟩ = ∫ A f^-α |2π Y(f)|² df for a Ramsey or Hahn-echo sequence of length t, on a dense grid plus the averaged tail."""
    f = np.concatenate([np.geomspace(f_low_hz, 1.0 / t_s, 4000, endpoint=False), np.linspace(1.0 / t_s, 2000.0 / t_s, 400001)])
    if kind == "ramsey":
        w, tail = 4.0 * np.sin(np.pi * f * t_s) ** 2 / f**2, 2.0           # mean of sin² is 1/2
    else:
        w, tail = 16.0 * np.sin(np.pi * f * t_s / 2) ** 4 / f**2, 6.0      # mean of sin⁴ is 3/8
    trap = getattr(np, "trapezoid", None) or np.trapz
    return trap(A_hz2 * f**-alpha * w, f) + tail * A_hz2 * f[-1] ** (-alpha - 1) / (alpha + 1)


def test_device_spectra_through_the_ramsey_and_echo_filters():
    """
    Each fitted spectrum A/f^α, from 1/(averaging time) upwards, integrated numerically through the Ramsey
    and Hahn-echo filter functions: the 1/e times (⟨φ²⟩ = 2) give the reported echo times, and T2* within
    6 % (the spectrum is pinned to a band ending at 1/mean(T2*), and the Ramsey filter is not a sharp cut).
    """
    from scipy.optimize import brentq
    for d in DEVICES.values():
        nm = d.noise_parameters()
        for q in range(2):
            s, a = nm.zeeman_sigma_mhz[q], nm.zeeman_alpha[q]
            A = 1e12 * psd_amplitude(s, nm.f_low_hz, nm.f_ramsey_hz, a)
            t2s = 1e-6 * zeeman_sigma_from_t2_star(1.0) / s
            ramsey = np.exp(brentq(lambda lt: _phase_variance(A, a, nm.f_low_hz, np.exp(lt), "ramsey") - 2.0,
                                   np.log(0.2 * t2s), np.log(5 * t2s)))
            assert ramsey == pytest.approx(t2s, rel=0.06), (d.key, q)
            echo = d.echo_times_us()[q]
            if echo is not None:
                t = np.exp(brentq(lambda lt: _phase_variance(A, a, nm.f_low_hz, np.exp(lt), "echo") - 2.0,
                                  np.log(1e-7), np.log(1e-2)))
                assert 1e6 * t == pytest.approx(echo, rel=2e-3), (d.key, q)


def test_echo_data_set_the_direction_of_alpha():
    """α = 1 predicts echoes too short for Mills and Tanttu (α > 1) and slightly too long for Noiri (α < 1)."""
    for key, sign in (("mills2022", 1), ("tanttu2024", 1), ("noiri2022", -1)):
        for fit, echo in zip(DEVICES[key].zeeman_fit(), DEVICES[key].echo_times_us()):
            assert np.sign(echo - fit["echo_us_at_alpha1"]) == sign
            assert np.sign(fit["alpha"] - 1.0) == sign


def test_noiri_total_depends_on_how_the_echo_is_read():
    d = DEVICES["noiri2022"]
    fitted = d.budget()
    forced = d.budget(zeeman_alpha=1.0)                   # echo shortfall read as white noise
    assert forced.markovian > 50 * fitted.markovian
    assert forced.total > 3 * fitted.total
    assert d.value("exchange_sigma_rel") == pytest.approx(0.0034) and d.entry("exchange_sigma_rel")["basis"] == "upper bound"
    sigma = zeeman_sigma_from_t2_star(min(d.value("t2_star_us")))                 # widest conditional Ramsey width
    assert d.value("exchange_sigma_rel") == pytest.approx(2 * sigma / d.value("j_on_mhz"), rel=0.02)   # ΔJ/2 < Δf ≤ σ


def test_tanttu_gate_detuning_synchronises_the_square_pulse():
    """At the idle ΔE_z the 256 ns square pulse leaves P = (J/Ω)² sin²(πΩT) in |↑↓⟩↔|↓↑⟩, 1 − F = (4/5)·P/2."""
    d = DEVICES["tanttu2024"]
    J, T, dez = d.model_peak_j_mhz, d.cz_time_ns, d.value("delta_ez_mhz")
    om = np.hypot(J, dez)
    P = (J / om) ** 2 * np.sin(np.pi * om * 1e-3 * T) ** 2
    assert d.coherent_at_idle() == pytest.approx(0.8 * P / 2, rel=0.02)
    assert d.coherent_at_idle() == pytest.approx(4.0e-3, rel=0.02)
    assert np.min(np.abs(square_sync_times(d.delta_ez_model_mhz, 200.0, 300.0) - T)) < 0.05
    assert d.budget().coherent < 1e-8


def test_xue_model_is_below_the_authors_own_simulation():
    """Xue et al. simulate 1 − F = 0.22e-3 (quasi-static δf, 1/f δv_B, barrier-dependent qubit frequencies); the model omits the last."""
    d = DEVICES["xue2022"]
    b = d.budget()
    ratio = b.total / d.value("authors_simulated_infidelity")
    assert 0.58 < ratio < 0.66
    assert b.coherent < 1e-6                             # authors: < 1e-6 without noise


@pytest.mark.parametrize("key", list(DEVICES))
def test_device_budget_adds_up_to_the_combined_simulation(key):
    d = DEVICES[key]
    b = d.budget()
    sim = combined_simulation(d.noise_parameters(), b.j_pulse, b.dt_ns, n_quadrature=3, n_fast=8, seed=4)
    assert abs(sim["infidelity"] - b.total) < 3 * sim["stderr"] + 0.05 * b.total


def test_cli_devices_prints_the_table(capsys):
    assert cli_main(["devices", "--budget"]) == 0
    out = capsys.readouterr().out
    for d in DEVICES.values():
        assert d.label in out
    assert "upper bound" not in out and "not a bound" in out
    assert cli_main(["devices", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["devices"]) == len(DEVICES)
    assert all(0.99 < r["model_fidelity"] < 1.0 for r in payload["devices"])


def test_recalibration_inputs_are_flagged():
    d = DEVICES["xue2022"]
    assert not d.assumed("recalibration_interval_h")
    for name in ("drift_j_rel_per_sqrt_h", "drift_dez_mhz_per_sqrt_h", "calibration_std_j_rel", "calibration_std_dez_mhz"):
        assert d.assumed(name) and d.entry(name)["basis"] == "assumed"


@pytest.mark.slow
def test_best_square_gate_is_the_first_synchronised_time_under_the_ceiling():
    d = DEVICES["xue2022"]
    best = best_gate_times(d, ("square", "cosine"))
    t_min = shortest_gate_time("square", d.j_cap_mhz)
    first = square_sync_times(d.delta_ez_model_mhz, t_min, 1000.0)[0]
    assert best["square"].duration_ns == pytest.approx(first, rel=0.01)          # 28.7 ns, n = 3
    assert best["square"].j_max_mhz <= d.j_cap_mhz * 1.001
    assert best["cosine"].duration_ns >= shortest_gate_time("cosine", d.j_cap_mhz) - 1e-6
    assert best["square"].total < best["cosine"].total < d.budget().total
