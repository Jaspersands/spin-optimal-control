# Changelog

## Unreleased

Candidate mechanisms for the gap between the modelled and the measured two-qubit error.

### Added
- `error_budget.GapMechanisms`: terms outside the four-part budget, each off unless its input is given, reported in `CZErrorBudget.gap` and included in `total` (`base_total` is the four-part budget). The terms are qubit frequencies moved by the exchange-gate voltage (`GateZeemanCoupling`, correlated with the exchange noise), Zeeman and exchange noise slower than the Ramsey average, pulse distortion through a single-pole low-pass with recalibrated area (`distorted_pulse`, `ExchangeVoltageMap`), residual exchange outside the pulse (`residual_exchange_error`), uncalibrated frequency and exchange offsets, and a measured coherent residual. `GAP_TERMS` lists them and `GapMechanisms.only` selects some.
- `error_budget.irb_error_bound`: the range interleaved RB allows for the true gate error (Magesan et al. 2012).
- `devices.gap_analysis` and `format_gap_table`: per device, what each term adds and its share of the gap, with a status (modelled, upper bound, measured, not published, not applicable). Unpublished inputs are swept, with the value that would close the gap. `Device.gap_mechanisms`, `voltage_map`, `gate_zeeman_coupling` and `benchmark_hours` build the inputs, and `Device.budget` takes `mechanisms`.
- `devices.json`: Xue's fit of the qubit frequencies against the barrier voltage (β₁, β₂, γ), the GST error generators of the CZ and the coherent part derived from them, and the GST run length derived from the recalibration schedule; Noiri's reference Clifford and single-qubit primitive fidelities, the authors' simulated dephasing error and the length of their noise record; Mills's reference Clifford fidelity and the 10 kHz echo floor for residual exchange; Tanttu's J-gate swing, reference Clifford fidelity, IRB run length (derived) and feedback parameters. The `not_reported` lists now name the line bandwidth, valley and tunnel energies and benchmark durations.
- `spin-control devices --gap` prints the mechanism table and the interleaved-RB ranges.
- `make_figures.py` writes `figures/gap_mechanisms.png` and adds the gap analysis to `figures/comparison.json`.
- README: a table of candidate mechanisms against the gap each closes, per device, and what interleaved RB and GST capture.

### Changed
- Tests: 132 → 148.

## 0.5.0 — 2026-10-19

A comparison of the modelled CZ with four published silicon devices.

### Added
- `data/devices.json`, shipped with the package: parameters of Xue 2022, Noiri 2022, Mills 2022 and Tanttu 2024 (device A). Every value has a source, and each device records which version of the paper its values were read from (`values_from`) and the title of its arXiv version (`arxiv_title`). Derived values, bounds and assumed values are flagged with a reason. For Tanttu 2024 the square pulse is modelled at the frequency difference where it is synchronised (11.555 MHz, assumed), and the coherent error it would have at the idle 10.8 MHz is reported separately. `devices` loads them (`load_devices`, `Device.noise_parameters`, `Device.budget`, `comparison_table`, `best_gate_times`).
- `error_budget`: CZ infidelity split into the coherent error after virtual-Z correction, quasi-static and 1/f^α Zeeman noise, quasi-static and 1/f exchange noise (toggle-frame filter functions), and Markovian T₁ and white dephasing from the Lindblad equation. `combined_simulation` checks that the parts add up.
- The Zeeman exponent α is fitted per qubit to T₂* and the Hahn-echo time (`fit_zeeman_alpha`, `echo_time_from_1f`, `psd_amplitude`, `white_dephasing_t2`), with α = 1 where no echo is reported. `CZNoiseParameters` takes `zeeman_alpha` (per qubit) and `exchange_alpha`.
- `optimal_gate_time` enumerates the synchronised times T_n = √(n² − ¼)/ΔE_z of a square pulse and searches the other windows down to the shortest time the exchange ceiling allows; `coherent_error` and `square_sync_times` expose the pieces. `grape_cz_pulse` gives a GRAPE CZ for comparison.
- `calibration.recalibration_interval` and `longest_interval`: how often a CZ must be recalibrated against random-walk drift of J₀ and the qubit frequencies (independent by default), with plain recalibration and with the Kalman tracker. `drift_rate_for_infidelity` sizes a J drift to a given excess error. The drift rates and calibration errors for Xue 2022 are assumed and flagged in `devices.json`.
- `BayesianActiveCalibrator.predict()`, and a `predict` flag on its update methods, so that a caller can advance the drift once per interval and then apply several measurements.
- `spin-control devices` (with `--budget`, `--optimum`, `--json`) prints the comparison table, the fitted exponents and per-device notes.
- `benchmarks/make_figures.py` (`--out`, `--quick`) writes `figures/error_budget_vs_gate_time.png`, `figures/summary.png`, `figures/calibration_interval.png` and `figures/comparison.json`.
- Notebook sections 11 and 12: the device comparison and the recalibration interval.

### Changed
- `pytest` registers a `slow` marker; deselect with `-m "not slow"`.
- Tests: 79 → 132.

## 0.4.0 — 2026-09-23

Changes from an adversarial review of 0.3.

### Removed
- `drag` and the `--drag` CLI flag. Exchange is a scalar baseband control with no quadrature, and no propagator ever used the computed "quadrature". The AWG exporter now takes arbitrary extra channels instead.

### Added
- `pulse_shaping`: square, cosine, Tukey and Blackman exchange windows calibrated to a CZ (the conditional phase depends only on pulse area), with leakage and fidelity from the exact propagator; `spin-control shape`.
- Toggle-frame filter functions (`control_matrix`, `gate_filter_functions`, `infidelity_from_filter_function`) with exact per-slice integrals, validated against Monte-Carlo with coloured noise. `filter_function_infidelity` now uses them.
- Thermal relaxation: `temperature_mk` and `larmor_frequency_mhz` split 1/T₁ into decay and excitation by detailed balance; `idle_superoperator`.
- `clifford_compiler` and `CompiledCliffordRB`: native-gate two-qubit RB (±X/2, ±Y/2, virtual Z, CZ; 1.5 CZ per Clifford) with per-gate noise and an optional non-ideal CZ; `spin-control rb` uses it by default.

### Fixed
- Cirq amplitude damping relaxed towards |0⟩ = |↑⟩, the opposite of the Lindblad model; it now uses generalised amplitude damping towards the thermal state.

### Changed
- PennyLane removed (never imported); Cirq and matplotlib are optional extras.
- Notebook renamed to `01_exchange_gates_noise_and_benchmarking.ipynb` and extended.
- Tests: 65 → 79.

## 0.3.0 — 2026-09-21

### Fixed (correctness)
- **Units.** The propagator now applies `2π·1e-3` so that J in MHz and t in ns give the correct phase. In 0.2 `expm(-iH dt)` was used directly, i.e. "J = 20 MHz" behaved like 3.2 GHz; the README quickstart reached 73 % fidelity for √SWAP.
- **Rotating frame.** The homogeneous Zeeman term `B0` is dropped by default (it commutes with everything); the lab frame remains available.
- **Pulse map.** One `clip(·, 0, J_max)` map in NumPy and JAX. 0.2 optimised a `relu` pulse, evaluated a `softplus` pulse and reported a `clip` pulse.
- **Slew penalty** acts on `dJ/dt` and honours the constructor argument (0.2 penalised adjacent Fourier coefficients with a hard-coded weight).
- **Time grid** uses slice midpoints consistent with `dt`.
- **Lindblad rates** converted from 1/µs to 1/ns; coherent part in correct units; solver is now the exact per-slice Liouvillian exponential (RK4 removed). Verified against analytic `T1` / `T2*` decay.
- **Cirq phase damping** parameter is `1 − exp(−2t/Tφ)` (coherences shrink by `√(1−γ)`), with `1/Tφ = 1/T2* − 1/(2T1)`.
- **Pink noise** cut-offs are in Hz and compared with Hz; multi-trap OU rates use seconds.
- **Valley model** docstring/initial state: singlet ⊗ ground valley, honest 8-dimensional description.
- **AWG export** covers the full gate window and rejects unknown formats.

### Added
- `units` module and documented convention.
- `gate_fidelity_local_z_free`, `average_gate_fidelity`, `exchange_gate(angle)`.
- GRAPE: `robust=True` ensemble via `jax.vmap`, `local_z_free=True` virtual-Z co-optimisation, multi-start seeds sized by exchange area, `loss_and_grad` API, NumPy fallback reproducing the JAX loss.
- `filter_function`, `filter_function_infidelity`, `channel_process_fidelity`, `channel_superoperator`.
- Full two-qubit Clifford group (11 520 elements) with closure tests; RB / IRB draw from it; `rb_fit`, `clifford_error_from_p`, `interleaved_depolarizing` validation hook.
- 2-D Kalman calibrator with covariance; `simulate_drift_tracking`.
- Qblox and Zurich Instruments export formats.
- CLI: `rb`, `noise-psd`, `benchmark`, `--json`, `--robust`, `--local-z`, `--format`.
- Test suite grown from 17 to 65 tests with physics assertions.
- Interactive web demo computes everything in the browser.

### Changed
- Default `b_0` is 1000 MHz (lab frame only); README defaults chosen so the quickstart reaches ≥ 99.75 % (√SWAP, 0.5 MHz gradient) and 100.000 % (CZ, 30 MHz gradient with virtual-Z).

## 0.2.0
- Valley physics, DRAG corrections, Bayesian calibration, AWG export, CLI, CI, notebook.

## 0.1.0
- Initial release.
