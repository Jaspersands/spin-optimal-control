"""
Published silicon two-qubit devices and the comparison of their measured
two-qubit gate fidelity with the CZ error budget of :mod:`error_budget`.

The parameters are stored in ``data/devices.json``. Every entry has a ``value``, a
``source`` (paper and table, figure or section) and an ``assumed`` flag.
Entries with ``assumed: true`` also carry a ``basis`` (``"derived"`` from
reported numbers, ``"assumed"`` typical value, ``"lower bound"`` or
``"upper bound"``) and a one-line ``reason``. Each device records which version
of the paper (arXiv vN or published) its values were read from.

Mapping onto the model (``Device.noise_parameters``):

* ΔE_z is the qubit frequency difference; the package Hamiltonian uses
  ΔB_z = ΔE_z/2 (see :mod:`error_budget`). ``delta_ez_gate_mhz``, where
  present, replaces it during the gate.
* Zeeman noise: the reported quasi-static widths δf if given, otherwise
  σ = √2/(2π T2*). The spectrum is 1/f^α, with α fitted per qubit to T2*
  and the Hahn-echo time where one is reported (``error_budget.fit_zeeman_alpha``),
  and α = 1 otherwise.
* Exchange noise: the quasi-static width of δJ/J with a 1/f spectrum.
* Lindblad term: T1, and the part of the echo decay the fitted spectrum
  cannot explain (``error_budget.white_dephasing_t2``).
* The spectra are pinned to the Ramsey band [1/averaging time, 1/T2*].
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .error_budget import (
    CZ_AREA_MHZ_NS,
    E_TO_AVG,
    GAP_TERMS,
    CZErrorBudget,
    CZNoiseParameters,
    ExchangeVoltageMap,
    GapMechanisms,
    GateZeemanCoupling,
    LABELS,
    coherent_error,
    cz_error_budget,
    fit_zeeman_alpha,
    irb_error_bound,
    optimal_gate_time,
    residual_exchange_error,
    white_dephasing_t2,
    zeeman_sigma_from_t2_star,
)
from .pulse_shaping import window

DATA_FILE = Path(__file__).resolve().parent / "data" / "devices.json"


@dataclass
class Device:
    """One published device: metadata and parameter entries from ``devices.json``."""

    key: str
    label: str
    short: str
    citation: str
    arxiv: str
    doi: str
    platform: str
    gate: str
    params: Dict[str, Dict[str, Any]] = field(repr=False)
    values_from: str = ""
    arxiv_title: str = ""
    defaults: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)
    not_reported: List[str] = field(default_factory=list, repr=False)

    # ----------------------------------------------------------------- #
    def has(self, name: str) -> bool:
        return name in self.params

    def entry(self, name: str) -> Dict[str, Any]:
        if name in self.params:
            return self.params[name]
        if name in self.defaults:
            return self.defaults[name]
        raise KeyError(f"{self.key}: no parameter {name!r}")

    def value(self, name: str, default: Any = None) -> Any:
        try:
            return self.entry(name)["value"]
        except KeyError:
            return default

    def assumed(self, name: str) -> bool:
        return bool(self.entry(name).get("assumed", False))

    # ----------------------------------------------------------------- #
    @property
    def cz_time_ns(self) -> float:
        return float(self.value("gate_time_ns"))

    @property
    def model_shape(self) -> str:
        return str(self.value("model_shape"))

    @property
    def tukey_alpha(self) -> float:
        return float(self.value("tukey_alpha", 0.5))

    @property
    def j_cap_mhz(self) -> Optional[float]:
        """
        Exchange ceiling (MHz) for the gate-time optimisation: the largest
        exchange the paper shows, or the peak the model pulse needs to reach
        the CZ area at the published time if that is higher (Mills 2022: a
        Tukey window at 40 ns peaks at 13.9 MHz against 12.5 MHz shown).
        """
        v = self.value("j_max_shown_mhz")
        if v is None:
            return None
        needed = CZ_AREA_MHZ_NS / (self.cz_time_ns * float(np.mean(window(self.model_shape, 2000, self.tukey_alpha))))
        return max(float(v), float(needed))

    @property
    def measured_fidelity(self) -> float:
        return float(self.value("measured_fidelity"))

    @property
    def measured_uncertainty(self) -> float:
        return float(self.entry("measured_fidelity").get("uncertainty", float("nan")))

    @property
    def method(self) -> str:
        return str(self.entry("measured_fidelity").get("method", ""))

    def zeeman_sigma_mhz(self) -> List[float]:
        if self.has("zeeman_sigma_khz"):
            return [1e-3 * float(x) for x in self.value("zeeman_sigma_khz")]
        return [zeeman_sigma_from_t2_star(float(x)) for x in self.value("t2_star_us")]

    @property
    def delta_ez_model_mhz(self) -> float:
        """ΔE_z used during the gate: ``delta_ez_gate_mhz`` if the data give one, else the idle ``delta_ez_mhz``."""
        return float(self.value("delta_ez_gate_mhz", self.value("delta_ez_mhz")))

    @property
    def model_peak_j_mhz(self) -> float:
        """Peak exchange of the model window at the published time (area 500 MHz·ns)."""
        return float(CZ_AREA_MHZ_NS / (self.cz_time_ns * np.mean(window(self.model_shape, 2000, self.tukey_alpha))))

    def _bands(self):
        sig = self.zeeman_sigma_mhz()
        t2s = [zeeman_sigma_from_t2_star(1.0) / x for x in sig]          # T2* (µs) implied by σ
        f_low = 1.0 / float(self.value("ramsey_averaging_s"))
        f_ramsey = 1.0 / (1e-6 * float(np.mean(t2s)))
        return sig, f_low, f_ramsey

    def echo_times_us(self) -> List[Optional[float]]:
        key = "t2_cpmg_us" if self.has("t2_cpmg_us") else "t2_hahn_us"
        v = self.value(key)
        return [None, None] if v is None else [float(x) for x in v]

    def zeeman_fit(self) -> List[Dict[str, float]]:
        """Per-qubit 1/f^α fit to T2* and the echo time (``error_budget.fit_zeeman_alpha``)."""
        sig, f_low, f_ramsey = self._bands()
        echo = self.echo_times_us()
        a0 = float(self.value("zeeman_alpha", 1.0))
        return [fit_zeeman_alpha(sig[q], echo[q], f_low, f_ramsey, default_alpha=a0) for q in range(2)]

    def noise_parameters(self, zeeman_alpha: Optional[float] = None) -> CZNoiseParameters:
        """
        Model parameters for :func:`error_budget.cz_error_budget`. ``zeeman_alpha``
        forces one exponent for both qubits instead of the fit (the part of the
        echo decay it cannot explain then becomes white dephasing).
        """
        sig, f_low, f_ramsey = self._bands()
        t1 = [float(x) for x in self.value("t1_us")]
        echo = [np.inf if e is None else e for e in self.echo_times_us()]
        alphas = [zeeman_alpha] * 2 if zeeman_alpha is not None else [f["alpha"] for f in self.zeeman_fit()]
        t2w = [white_dephasing_t2(echo[q], sig[q], f_low, f_ramsey, t1[q], alphas[q]) for q in range(2)]
        if self.has("f_q1_ghz"):
            larmor = (1e3 * float(self.value("f_q1_ghz")), 1e3 * float(self.value("f_q2_ghz")))
        else:
            larmor = (1e3 * float(self.value("mean_zeeman_ghz")),) * 2
        return CZNoiseParameters(
            delta_ez_mhz=self.delta_ez_model_mhz,
            zeeman_sigma_mhz=(sig[0], sig[1]),
            exchange_sigma_rel=float(self.value("exchange_sigma_rel")),
            t1_us=(t1[0], t1[1]),
            t2_markov_us=(t2w[0], t2w[1]),
            zeeman_alpha=(float(alphas[0]), float(alphas[1])),
            exchange_alpha=float(self.value("exchange_alpha", 1.0)),
            f_low_hz=f_low,
            f_ramsey_hz=f_ramsey,
            temperature_mk=float(self.value("electron_temperature_mk")),
            larmor_mhz=larmor,
        )

    def coherent_at_idle(self) -> Optional[float]:
        """Coherent error at the published time with the idle ΔE_z, when the data set a different ΔE_z for the gate."""
        if not self.has("delta_ez_gate_mhz"):
            return None
        nm = self.noise_parameters().noiseless().replace(delta_ez_mhz=float(self.value("delta_ez_mhz")))
        return coherent_error(nm, self.model_shape, self.cz_time_ns, self.tukey_alpha)

    def budget(self, duration_ns: Optional[float] = None, shape: Optional[str] = None,
               zeeman_alpha: Optional[float] = None, mechanisms: Optional[GapMechanisms] = None) -> CZErrorBudget:
        """
        CZ error budget at ``duration_ns`` (default: the published gate time) with ``shape`` (default: the model
        shape). ``mechanisms`` adds gap terms (``gap_mechanisms`` gives the ones the paper's numbers allow).
        """
        T = self.cz_time_ns if duration_ns is None else float(duration_ns)
        s = self.model_shape if shape is None else shape
        return cz_error_budget(self.noise_parameters(zeeman_alpha), s, T, tukey_alpha=self.tukey_alpha,
                               mechanisms=mechanisms)

    # ----------------------------------------------------------------- #
    # Candidate mechanisms for the gap
    # ----------------------------------------------------------------- #
    @property
    def exchange_is_pulsed(self) -> bool:
        """False for a gate under always-on exchange (Noiri 2022), where the pulse terms do not apply."""
        return "no exchange pulse" not in str(self.value("pulse", ""))

    def voltage_map(self) -> Optional[ExchangeVoltageMap]:
        """
        J(v) on the exchange gate where the paper gives it: Xue 2022, J = J_res e^{2αv_B}; Tanttu 2024,
        a slope in decades per volt over the J-gate swing, which puts J = J_on·10^(−slope·swing) at v = 0.
        """
        if self.has("exchange_alpha_per_v") and self.has("j_residual_khz"):
            return ExchangeVoltageMap(2.0 * float(self.value("exchange_alpha_per_v")),
                                      1e-3 * float(self.value("j_residual_khz")))
        if self.has("j_gate_slope_dec_per_v") and self.has("j_gate_swing_mv"):
            k = float(np.log(10.0)) * float(self.value("j_gate_slope_dec_per_v"))
            j0 = float(self.value("j_on_mhz")) * float(np.exp(-k * 1e-3 * float(self.value("j_gate_swing_mv"))))
            return ExchangeVoltageMap(k, j0)
        return None

    def gate_zeeman_coupling(self) -> Optional[GateZeemanCoupling]:
        """Qubit frequencies against the exchange-gate voltage, where the paper fits them (Xue 2022)."""
        if not self.has("barrier_beta_mhz_per_v_gamma"):
            return None
        b = self.value("barrier_beta_mhz_per_v_gamma")
        return GateZeemanCoupling((float(b[0]), float(b[1])), float(self.value("barrier_gamma")), self.voltage_map())

    def benchmark_hours(self) -> Dict[str, Optional[float]]:
        """
        Longest stretch of lab time a benchmark averages over without recalibrating, per noise source (h): the
        qubit-frequency recalibration interval where one is reported, otherwise the run length; None if neither is.
        """
        run = self.value("benchmark_duration_h")
        zeeman = self.value("recalibration_interval_h", run)
        return {"zeeman": None if zeeman is None else float(zeeman), "exchange": None if run is None else float(run)}

    def residual_exchange_mhz(self) -> float:
        if self.has("j_residual_khz"):
            return 1e-3 * float(self.value("j_residual_khz"))
        if self.has("j_off_hz"):
            return 1e-6 * float(self.value("j_off_hz"))
        return 0.0

    def gap_mechanisms(self) -> GapMechanisms:
        """
        The gap terms that the paper's own numbers (or values derived from them) allow. Terms that need an
        unpublished value stay off; ``gap_analysis`` shows those as a function of that value.
        """
        h = self.benchmark_hours()
        return GapMechanisms(
            gate_zeeman=self.gate_zeeman_coupling() if self.exchange_is_pulsed else None,
            zeeman_f_run_hz=None if h["zeeman"] is None else 1.0 / (3600.0 * h["zeeman"]),
            exchange_f_run_hz=None if h["exchange"] is None else 1.0 / (3600.0 * h["exchange"]),
            voltage_map=self.voltage_map(),
            j_off_mhz=self.residual_exchange_mhz(),
            idle_ns=0.0,                        # no paper gives a time outside the pulse that its benchmark assigns to the CZ
            calibration_residual=float(self.value("gst_cz_coherent_infidelity", 0.0)),
        )


def load_devices(path: Optional[str] = None) -> Dict[str, Device]:
    """Read ``devices.json`` (or ``path``) into ``{key: Device}``, in file order."""
    p = Path(path) if path is not None else DATA_FILE
    raw = json.loads(p.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    out: Dict[str, Device] = {}
    for key, d in raw["devices"].items():
        out[key] = Device(key=key, label=d["label"], short=d.get("short", d["label"]), citation=d["citation"],
                          arxiv=d["arxiv"], doi=d["doi"], platform=d["platform"], gate=d["gate"], params=d["params"],
                          defaults=defaults, not_reported=list(d.get("not_reported", [])),
                          values_from=d.get("values_from", ""), arxiv_title=d.get("arxiv_title", ""))
    return out


def get_device(key: str) -> Device:
    return load_devices()[key]


def iter_entries(device: Device):
    """Yield ``(name, entry)`` for every parameter of a device, including the global defaults it uses."""
    for name, e in device.params.items():
        yield name, e
    for name, e in device.defaults.items():
        if name not in device.params:
            yield name, e


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def comparison_row(device: Device) -> Dict[str, Any]:
    """Measured fidelity against the modelled CZ at the published gate time and model shape."""
    b = device.budget()
    return {
        "key": device.key,
        "device": device.label,
        "short": device.short,
        "gate": device.gate,
        "measured_fidelity": device.measured_fidelity,
        "measured_uncertainty": device.measured_uncertainty,
        "method": device.method,
        "gate_time_ns": device.cz_time_ns,
        "model_shape": device.model_shape,
        "model_fidelity": b.fidelity,
        "model_infidelity": b.total,
        "measured_infidelity": 1.0 - device.measured_fidelity,
        "gap_ratio": (1.0 - device.measured_fidelity) / b.total,
        "dominant": b.dominant,
        "dominant_label": LABELS[b.dominant],
        "contributions": b.contributions,
        "quasistatic": b.zeeman_quasistatic + b.exchange_quasistatic,
        "zeeman_alpha": list(device.noise_parameters().zeeman_alpha),
        "coherent_at_idle_dez": device.coherent_at_idle(),
        "model_peak_j_mhz": device.model_peak_j_mhz,
        "share_explained": b.total / (1.0 - device.measured_fidelity),
    }


def comparison_table(devices: Optional[Dict[str, Device]] = None) -> List[Dict[str, Any]]:
    devs = load_devices() if devices is None else devices
    return [comparison_row(d) for d in devs.values()]


def best_gate_times(device: Device, shapes: Sequence[str] = ("square", "cosine")) -> Dict[str, CZErrorBudget]:
    """Optimal duration and the modelled fidelity there, per window shape, with J below the device's demonstrated maximum."""
    nm = device.noise_parameters()
    return {s: optimal_gate_time(nm, s, tukey_alpha=device.tukey_alpha, j_cap_mhz=device.j_cap_mhz) for s in shapes}


def format_comparison_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Plain-text version of the README table."""
    w = max(34, max(len(r["device"]) for r in rows))
    head = f"{'device':{w}s} {'measured (method)':30s} {'time':>7s} {'model':>9s} {'meas/model':>10s}  dominant modelled error"
    lines = [head, "-" * len(head)]
    for r in rows:
        meas = f"{100 * r['measured_fidelity']:.2f}% ({r['method']})"
        gap = f"{r['gap_ratio']:.0f}x" if r["gap_ratio"] >= 10 else f"{r['gap_ratio']:.1f}x"
        lines.append(f"{r['device']:{w}s} {meas:30s} {r['gate_time_ns']:5.0f}ns {100 * r['model_fidelity']:8.3f}% "
                     f"{gap:>10s}  {r['dominant_label']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The gap between measured and modelled error
# --------------------------------------------------------------------------- #
CLOSING = ("modelled", "upper bound", "measured")          # statuses whose value counts towards the gap


def _crossing(x, y, target: float) -> Optional[float]:
    """First x, in the order given, at which y reaches ``target`` (log-interpolated); None if it never does."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    for i in range(len(x)):
        if y[i] >= target:
            if i == 0 or y[i - 1] <= 0:
                return float(x[i])
            t = (np.log(target) - np.log(y[i - 1])) / (np.log(y[i]) - np.log(y[i - 1]))
            return float(np.exp(np.log(x[i - 1]) + t * (np.log(x[i]) - np.log(x[i - 1]))))
    return None


def _required(x, y, target: float, name: str, unit: str) -> Dict[str, Any]:
    v = _crossing(x, y, target)
    return {"name": name, "unit": unit, "value": v, "range": [float(min(x)), float(max(x))], "reached": v is not None}


def gap_analysis(device: Device, sweeps: bool = True, n_points: int = 13) -> Dict[str, Any]:
    """
    How much of the gap between the measured and the modelled two-qubit error each ``GAP_TERMS`` mechanism
    closes for one device, at the published gate time and model shape.

    Each term has a ``status``: "modelled" (every input published or derived from published numbers),
    "upper bound" (as modelled, but the experiment fed back on that quantity during the run), "measured"
    (a part of the measured error the paper itself separates), "not published" (an input is missing; the
    term is given as a function of that input in ``sweep``, and ``required`` is the value that would close
    the gap) or "not applicable". ``share`` is the value as a fraction of the gap. ``bounds`` holds limits
    the paper gives on mechanisms the model cannot represent, and ``benchmark`` what the method reports.
    With ``sweeps=False`` the slower function-of-value curves are skipped (``required`` is then given only
    for the closed-form terms).
    """
    nm = device.noise_parameters()

    def budget(m: Optional[GapMechanisms] = None) -> CZErrorBudget:
        return cz_error_budget(nm, device.model_shape, device.cz_time_ns, tukey_alpha=device.tukey_alpha, mechanisms=m)

    base = budget()
    measured = 1.0 - device.measured_fidelity
    gap = measured - base.total
    mech = device.gap_mechanisms()
    full = budget(mech)
    pulsed = device.exchange_is_pulsed
    n = int(n_points)

    def run(**kw) -> Dict[str, float]:
        return budget(GapMechanisms(voltage_map=mech.voltage_map, **kw)).gap

    terms: Dict[str, Dict[str, Any]] = {}

    def put(name, status, value=None, note="", required=None, sweep=None, **extra):
        e = {"label": LABELS[name], "status": status, "value": None if value is None else float(value),
             "share": None if value is None else float(value) / gap, "note": note, "required": required,
             "sweep": sweep}
        e.update(extra)
        terms[name] = e

    unpulsed = "the exchange is always on; there is no exchange pulse"

    # exchange-gate voltage moving the qubit frequencies
    if not pulsed:
        put("gate_zeeman", "not applicable", note=unpulsed)
    elif mech.gate_zeeman is not None:
        k_peak = mech.gate_zeeman.sensitivity_mhz(np.array([base.j_max_mhz]))[:, 0]
        put("gate_zeeman", "modelled", full.gap["gate_zeeman"],
            note="frequency fit against the gate voltage from the paper; the exchange noise is taken to be noise "
                 "on that voltage, so it moves the qubit frequencies as well",
            df_dlnj_peak_mhz=[float(v) for v in k_peak])
    else:
        sw = req = None
        if sweeps:
            x = np.geomspace(0.01, 30.0, n)
            y = [run(gate_zeeman=GateZeemanCoupling.linear((0.0, float(k))))["gate_zeeman"] for k in x]
            sw = {"name": "df_dlnj_mhz", "unit": "MHz", "x": x.tolist(), "y": [float(v) for v in y]}
            req = _required(x, y, gap, "df_dlnj_mhz", "MHz")
        put("gate_zeeman", "not published", note="no fit of the qubit frequencies against the gate voltage; shown "
            "for a shift of one qubit linear in the voltage, df/d(ln J), with the model's exchange noise",
            required=req, sweep=sw)

    # noise slower than the Ramsey average
    hours = device.benchmark_hours()
    feedback = str(device.value("feedback_parameters", ""))
    sw = None
    if sweeps:
        x = np.geomspace(0.5, 1000.0, n)
        ys = [run(zeeman_f_run_hz=1.0 / (3600.0 * h), exchange_f_run_hz=1.0 / (3600.0 * h)) for h in x]
        sw = {k: {"name": "duration_h", "unit": "h", "x": x.tolist(), "y": [float(g[k]) for g in ys]}
              for k in ("slow_zeeman", "slow_exchange")}
    for name, src, fb in (("slow_zeeman", "zeeman", "f_ESR"), ("slow_exchange", "exchange", "V_CZ")):
        s_ = None if sw is None else sw[name]
        req = None if s_ is None else _required(s_["x"], s_["y"], gap, "duration_h", "h")
        h = hours[src]
        if h is None:
            put(name, "not published", note="benchmark duration not published", required=req, sweep=s_)
        else:
            status = "upper bound" if fb in feedback else "modelled"
            note = f"spectrum extended down to 1/({h:g} h)"
            if status == "upper bound":
                note += f"; {fb} was fed back during the run, which removes part of this drift"
            put(name, status, full.gap[name], note=note, required=req, sweep=s_, hours=h)

    # pulse distortion
    if not pulsed:
        put("distortion", "not applicable", note=unpulsed)
    else:
        sw = req = None
        if sweeps:
            x = np.geomspace(500.0, 1.0, n)
            y = [run(bandwidth_mhz=float(b))["distortion"] for b in x]
            sw = {"name": "bandwidth_mhz", "unit": "MHz", "x": x.tolist(), "y": [float(v) for v in y]}
            req = _required(x, y, gap, "bandwidth_mhz", "MHz")
        how = ("the gate voltage is filtered and mapped to J with the paper's J(v)" if mech.voltage_map is not None
               else "J is filtered directly; no J(v) is published")
        put("distortion", "not published", note=f"line bandwidth not published; single-pole low-pass, {how}, "
            "area recalibrated", required=req, sweep=sw)

    # residual exchange outside the pulse
    if not pulsed:
        put("residual_exchange", "not applicable", note=unpulsed)
    else:
        area = None
        if 0 < gap < E_TO_AVG:
            area = float(4.0 * np.arcsin(np.sqrt(gap / E_TO_AVG)) / (2.0 * np.pi * 1e-3))
        j_ref = device.residual_exchange_mhz() or 1e-3 * float(device.value("exchange_echo_floor_khz", 0.0))
        x = np.geomspace(0.1, 300.0, n)                  # below a conditional phase of π (500 MHz ns)
        j_s = j_ref or 0.01
        y = [residual_exchange_error(nm.delta_ez_mhz, j_s, float(a) / j_s) for a in x]
        put("residual_exchange", "not published",
            note="no paper gives a time outside the pulse that its benchmark assigns to the CZ",
            required={"name": "exchange_area_mhz_ns", "unit": "MHz ns", "value": area, "range": [0.1, 300.0],
                      "reached": area is not None},
            sweep={"name": "exchange_area_mhz_ns", "unit": "MHz ns", "x": x.tolist(), "y": [float(v) for v in y]},
            j_reference_mhz=j_ref, time_required_ns=None if not (area and j_ref) else area / j_ref,
            per_100ns=residual_exchange_error(nm.delta_ez_mhz, j_ref, 100.0) if j_ref else None)

    # uncalibrated offsets (quadratic, so the value that closes the gap is closed-form)
    c_f = run(frequency_offset_mhz=(1.0, 1.0))["frequency_offset"]
    put("frequency_offset", "not published", note="rms offset of both qubit frequencies during the gate",
        required={"name": "frequency_offset_khz", "unit": "kHz", "value": 1e3 * float(np.sqrt(gap / c_f)),
                  "reached": True}, sigma_khz=[1e3 * s for s in nm.zeeman_sigma_mhz])
    c_j = run(exchange_offset_rel=1.0)["exchange_offset"]
    put("exchange_offset", "not published", note="rms offset of dJ/J during the gate",
        required={"name": "exchange_offset_rel", "unit": "", "value": float(np.sqrt(gap / c_j)), "reached": True},
        sigma_rel=float(nm.exchange_sigma_rel))

    if mech.calibration_residual:
        put("calibration_residual", "measured", mech.calibration_residual,
            note="coherent part of the GST error generator; the paper gives no cause")
    else:
        put("calibration_residual", "not published", note="no coherent part of the two-qubit error is tabulated")

    bounds: Dict[str, Dict[str, Any]] = {}
    if device.has("single_qubit_primitive_fidelity"):
        v = 1.0 - float(np.mean(device.value("single_qubit_primitive_fidelity")))
        bounds["drive"] = {"label": "heating, leakage or other drive errors", "status": "upper bound", "value": v,
                           "share": v / gap, "note": "the authors' argument: a single-qubit primitive, also two "
                           "CROT pulses, has this error in total (single-qubit RB)"}

    bench: Dict[str, Any] = {"method": device.method}
    if "RB" in device.method and device.has("reference_clifford_fidelity"):
        bench["irb"] = irb_error_bound(float(device.value("reference_clifford_fidelity")), device.measured_fidelity)
    if device.has("gst_cz_stochastic_epsilon"):
        eps, th = float(device.value("gst_cz_stochastic_epsilon")), float(device.value("gst_cz_hamiltonian_theta"))
        bench["gst"] = {"stochastic": E_TO_AVG * eps, "coherent": E_TO_AVG * th**2}
    if device.has("measured_fidelity_gst"):
        bench["gst_infidelity"] = 1.0 - float(device.value("measured_fidelity_gst"))

    closed = float(sum(t["value"] for t in terms.values() if t["status"] in CLOSING))
    return {
        "key": device.key, "short": device.short, "measured": measured, "model": base.total, "gap": gap,
        "terms": {k: terms[k] for k in GAP_TERMS}, "bounds": bounds, "benchmark": bench,
        "closed": closed, "closed_share": closed / gap, "remaining": gap - closed,
        "model_with_terms": full.total,
    }


def _cell(t: Dict[str, Any]) -> str:
    if t["status"] in CLOSING:
        pre = "<= " if t["status"] == "upper bound" else ""
        return f"{pre}{t['value']:.1e} ({100 * t['share']:.0f}%)"
    if t["status"] == "not applicable":
        return "n/a"
    r = t.get("required")
    if r and r.get("value") is not None:
        v = r["value"]
        if r["name"] == "exchange_offset_rel":
            return f"needs {100 * v:.1f}% rms"
        return f"needs {v:.3g} {r['unit']}".rstrip()
    if r and not r.get("reached", True):
        return "not closed in range"
    return "not published"


def format_gap_table(analyses: Sequence[Dict[str, Any]]) -> str:
    """Plain-text table: each gap term against each device (value and share of the gap, or what it would need)."""
    w0 = max(len(LABELS[k]) for k in GAP_TERMS) + 2
    w = 26
    lines = [f"{'':{w0}s}" + "".join(f"{a['short'][:w - 2]:>{w}s}" for a in analyses),
             f"{'measured - model':{w0}s}" + "".join(f"{a['gap']:>{w}.2e}" for a in analyses)]
    lines.append("-" * len(lines[0]))
    for k in GAP_TERMS:
        lines.append(f"{LABELS[k]:{w0}s}" + "".join(f"{_cell(a['terms'][k]):>{w}s}" for a in analyses))
    lines.append("-" * len(lines[0]))
    lines.append(f"{'closed by published values':{w0}s}"
                 + "".join(f"{100 * a['closed_share']:>{w - 1}.0f}%" for a in analyses))
    return "\n".join(lines)
