# Changelog

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
