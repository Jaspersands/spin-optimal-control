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
    CZErrorBudget,
    CZNoiseParameters,
    LABELS,
    coherent_error,
    cz_error_budget,
    fit_zeeman_alpha,
    optimal_gate_time,
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
               zeeman_alpha: Optional[float] = None) -> CZErrorBudget:
        """CZ error budget at ``duration_ns`` (default: the published gate time) with ``shape`` (default: the model shape)."""
        T = self.cz_time_ns if duration_ns is None else float(duration_ns)
        s = self.model_shape if shape is None else shape
        return cz_error_budget(self.noise_parameters(zeeman_alpha), s, T, tukey_alpha=self.tukey_alpha)


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
