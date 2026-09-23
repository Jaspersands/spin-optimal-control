"""
Native compilation of the Clifford groups for spin-qubit hardware.

Native gates: X/2, Y/2 and their inverses (EDSR pulses of duration t₁q), the
virtual Z/2 = S (a frame update — no pulse, no error) and CZ.

* Every single-qubit Clifford gets a shortest pulse sequence (Dijkstra over
  {±X/2, ±Y/2} with cost 1 and S with cost 0).
* The two-qubit Clifford group C₂ (11 520 elements modulo phase) is organised
  into left cosets of the local group C₁⊗C₁ (576 elements). A breadth-first
  search over "CZ, then a local layer" finds every coset with the minimum
  number of CZs, giving each Clifford a decomposition

      L_k · CZ · L_{k−1} · CZ ⋯ CZ · L_0,     L_i ∈ C₁ ⊗ C₁.

  The known class sizes come out: 576 Cliffords need no CZ, 5184 one, 5184 two
  and 576 three (the SWAP-like class) — 1.5 CZ per Clifford on average.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

_I2 = np.eye(2, dtype=np.complex128)
_CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)


def _rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


NATIVE_1Q: Dict[str, np.ndarray] = {
    "X/2": _rx(np.pi / 2), "-X/2": _rx(-np.pi / 2),
    "Y/2": _ry(np.pi / 2), "-Y/2": _ry(-np.pi / 2),
    "S": np.diag([1, 1j]).astype(np.complex128),          # virtual Z/2
}
PULSE_COST = {"X/2": 1, "-X/2": 1, "Y/2": 1, "-Y/2": 1, "S": 0}


def unitary_key(u: np.ndarray, decimals: int = 5) -> bytes:
    """Phase-invariant hashable key (first non-zero entry made real positive)."""
    flat = u.reshape(-1)
    idx = int(np.argmax(np.abs(flat) > 1e-9))
    phase = np.conj(flat[idx]) / abs(flat[idx])
    v = np.round(u * phase, decimals) + 0.0
    return np.concatenate([v.real.ravel(), v.imag.ravel()]).tobytes()


@dataclass(frozen=True)
class OneQubitClifford:
    unitary: np.ndarray
    gates: Tuple[str, ...]            # applied left to right in time

    @property
    def n_pulses(self) -> int:
        return sum(PULSE_COST[g] for g in self.gates)


@lru_cache(maxsize=1)
def one_qubit_cliffords() -> Tuple[OneQubitClifford, ...]:
    """The 24 single-qubit Cliffords with shortest native pulse sequences (identity first)."""
    start = np.eye(2, dtype=np.complex128)
    best: Dict[bytes, Tuple[int, int, Tuple[str, ...], np.ndarray]] = {}
    heap = [(0, 0, (), 0)]
    mats = {0: start}
    counter = 1
    while heap:
        cost, length, gates, mid = heapq.heappop(heap)
        u = mats[mid]
        k = unitary_key(u)
        if k in best and (best[k][0], best[k][1]) <= (cost, length):
            continue
        best[k] = (cost, length, gates, u)
        if len(best) == 24 and cost > 3:
            break
        if length >= 6:
            continue
        for g, m in NATIVE_1Q.items():
            nu = m @ u
            nk = unitary_key(nu)
            nc, nl = cost + PULSE_COST[g], length + 1
            if nk not in best or (best[nk][0], best[nk][1]) > (nc, nl):
                mats[counter] = nu
                heapq.heappush(heap, (nc, nl, gates + (g,), counter))
                counter += 1
    items = sorted(best.values(), key=lambda t: (t[0], t[1]))
    if len(items) != 24:
        raise RuntimeError(f"expected 24 single-qubit Cliffords, found {len(items)}")
    return tuple(OneQubitClifford(unitary=u, gates=g) for _, _, g, u in items)


@dataclass(frozen=True)
class TwoQubitClifford:
    unitary: np.ndarray
    layers: Tuple[Tuple[int, int], ...]   # local layers (c0, c1) indices into one_qubit_cliffords(), L_0 first
    n_cz: int                             # one CZ between consecutive local layers

    def native_sequence(self) -> List[Tuple[str, object]]:
        """[("local", (c0, c1)), ("cz", None), ("local", …), …] in time order."""
        seq: List[Tuple[str, object]] = []
        for i, layer in enumerate(self.layers):
            if i > 0:
                seq.append(("cz", None))
            seq.append(("local", layer))
        return seq


@lru_cache(maxsize=1)
def two_qubit_cliffords() -> Tuple[TwoQubitClifford, ...]:
    """All 11 520 two-qubit Cliffords with minimum-CZ native decompositions."""
    c1 = one_qubit_cliffords()
    local = [(a, b, np.kron(c1[a].unitary, c1[b].unitary)) for a in range(24) for b in range(24)]
    found: Dict[bytes, TwoQubitClifford] = {}

    def register_coset(rep: np.ndarray, rep_layers: Tuple[Tuple[int, int], ...], n_cz: int) -> bool:
        if unitary_key(rep) in found:
            return False
        for a, b, L in local:
            u = L @ rep
            k = unitary_key(u)
            if k not in found:
                found[k] = TwoQubitClifford(unitary=u, layers=rep_layers + ((a, b),), n_cz=n_cz)
        return True

    # level 0: the local group itself (rep = identity, the local layer is the whole element)
    register_coset(np.eye(4, dtype=np.complex128), (), 0)
    frontier = [(np.eye(4, dtype=np.complex128), ())]
    n_cz = 0
    while len(found) < 11520:
        n_cz += 1
        new_frontier = []
        for rep, rep_layers in frontier:
            for a, b, L in local:
                cand = _CZ @ L @ rep
                layers = rep_layers + ((a, b),)
                if register_coset(cand, layers, n_cz):
                    new_frontier.append((cand, layers))
        if not new_frontier:
            raise RuntimeError("Clifford closure stalled")
        frontier = new_frontier
    return tuple(found.values())


@lru_cache(maxsize=1)
def two_qubit_clifford_index() -> Dict[bytes, int]:
    return {unitary_key(c.unitary): i for i, c in enumerate(two_qubit_cliffords())}


def lookup_two_qubit_clifford(u: np.ndarray) -> int:
    """Index of the Clifford equal to ``u`` up to phase (KeyError if ``u`` is not Clifford)."""
    return two_qubit_clifford_index()[unitary_key(u)]


def compiled_unitary(c: TwoQubitClifford) -> np.ndarray:
    """Multiply the native layers back together (checks the decomposition)."""
    c1 = one_qubit_cliffords()
    U = np.eye(4, dtype=np.complex128)
    for kind, arg in c.native_sequence():
        if kind == "cz":
            U = _CZ @ U
        else:
            a, b = arg
            U = np.kron(c1[a].unitary, c1[b].unitary) @ U
    return U
