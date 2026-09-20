"""
Unit convention used throughout ``spin_optimal_control``.

* Frequencies / energies (``J``, ``ΔB_z``, ``B_0``, valley splitting, ...) are
  **linear frequencies in MHz** (i.e. ``E / h``), the unit experimentalists quote.
* Times (gate durations, time steps) are in **nanoseconds**.
* Rates in noise models are given in the units stated by the parameter name
  (``T1`` in µs, ``f_min`` in Hz, ...) and converted here.

The Schrödinger propagator for a Hamiltonian ``H`` expressed in MHz over a
time step ``dt`` in ns is therefore

    U = exp(-i · 2π · 1e-3 · H · dt)

so that a constant exchange ``J = 10 MHz`` applied for ``50 ns`` accumulates an
exchange angle of ``2π · 10e-3 · 50 = π`` rad (a SWAP).
"""

from __future__ import annotations

import numpy as np

#: Converts (MHz · ns) into radians of phase: ``2π · 1e-3``.
MHZ_NS_TO_RAD: float = 2.0 * np.pi * 1e-3

#: 1 µeV expressed in MHz (E/h). h = 4.135667696e-15 eV·s.
UEV_TO_MHZ: float = 241.798935

#: Convert a rate in 1/µs to 1/ns.
PER_US_TO_PER_NS: float = 1e-3


def phase(f_mhz: float, t_ns: float) -> float:
    """Phase (rad) accumulated by a linear frequency ``f_mhz`` over ``t_ns``."""
    return MHZ_NS_TO_RAD * f_mhz * t_ns


def cycles_to_mhz_ns(cycles: float) -> float:
    """Return the (J · T) product in MHz·ns that yields ``cycles`` full 2π rotations."""
    return cycles * 1e3
