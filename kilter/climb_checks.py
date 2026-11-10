"""Is a generated climb plausible, and is it new?

Plausible: every measurement below falls inside the range that 98% of real
training climbs fall in (1st to 99th percentile). Real test climbs pass all
checks 96% of the time, which is the bar to compare against.

New: how much does the climb overlap with its closest training climb?
Overlap = Jaccard similarity of the two sets of board positions (shared holds
divided by all holds in either). 1.0 is an exact copy.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .data import Climb

CHECKS = ["n_holds", "n_hand", "move_max", "start_y", "finish_gap", "height"]
CHECK_LABELS = {
    "n_holds": "number of holds", "n_hand": "number of hand holds",
    "move_max": "largest hand move", "start_y": "start height",
    "finish_gap": "finish below the top hand hold", "height": "start-to-finish height",
}


def measurements(c: Climb) -> dict[str, float]:
    hands = [h for h in c.holds if h.role != "foot"]
    starts = [h.y for h in hands if h.role == "start"] or [np.nan]
    finishes = [h.y for h in hands if h.role == "finish"] or [np.nan]
    ys = sorted((h.y, h.x) for h in hands)
    moves = [np.hypot(b[1] - a[1], b[0] - a[0]) for a, b in zip(ys, ys[1:])] or [0.0]
    top = max((h.y for h in hands), default=np.nan)
    return {
        "n_holds": len(c.holds), "n_hand": len(hands), "move_max": max(moves),
        "start_y": float(np.mean(starts)), "finish_gap": top - float(np.max(finishes)),
        "height": float(np.max(finishes)) - float(np.mean(starts)),
    }


class Plausibility:
    def __init__(self, train: list[Climb], lo: float = 1, hi: float = 99):
        m = np.array([[measurements(c)[k] for k in CHECKS] for c in train])
        self.lo, self.hi = np.nanpercentile(m, lo, axis=0), np.nanpercentile(m, hi, axis=0)

    def check(self, climbs: list[Climb]) -> np.ndarray:
        """[n_climbs, n_checks] booleans, True = inside the real-climb range."""
        m = np.array([[measurements(c)[k] for k in CHECKS] for c in climbs])
        return (m >= self.lo) & (m <= self.hi) & ~np.isnan(m)


class Novelty:
    def __init__(self, train: list[Climb], n_positions: int = 8000):
        sets = {frozenset(h.placement for h in c.holds): c for c in train}  # one row per hold set
        self.train = list(sets.values())
        self.n = n_positions
        self.A = self._matrix([list(s) for s in sets])
        self.sizes = np.asarray(self.A.sum(1)).ravel()

    def _matrix(self, sets: list[list[int]]) -> sparse.csr_matrix:
        rows = np.repeat(np.arange(len(sets)), [len(s) for s in sets])
        cols = np.concatenate([np.asarray(s, dtype=np.int64) for s in sets]) if sets else np.zeros(0, int)
        return sparse.csr_matrix((np.ones(len(cols), np.float32), (rows, cols)), shape=(len(sets), self.n))

    def nearest(self, climbs: list[Climb], chunk: int = 500) -> tuple[np.ndarray, list[Climb]]:
        """For each climb: Jaccard overlap with the most similar training climb, and that climb."""
        best, match = [], []
        for i in range(0, len(climbs), chunk):
            part = climbs[i:i + chunk]
            B = self._matrix([[h.placement for h in c.holds] for c in part])
            inter = (B @ self.A.T).toarray()
            size_b = np.asarray(B.sum(1)).ravel()
            jac = inter / (size_b[:, None] + self.sizes[None, :] - inter)
            j = jac.argmax(1)
            best.extend(jac[np.arange(len(part)), j])
            match.extend(self.train[k] for k in j)
        return np.array(best), match
