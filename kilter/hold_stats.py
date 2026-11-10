"""How hard is each individual hold? Estimated from the climbs that use it.

On the Kilter Original every board position always has the same physical
hold, so "position 1236" means one specific crimp. Trees can't learn that from
a position ID directly, so we give them a number per hold instead: on average,
how much harder than usual (for their angle) are the climbs that use this hold
as a hand hold? This is "target encoding". It's computed only from training
climbs, and out-of-fold on the training set itself, otherwise the model would
see each climb's own grade baked into its features.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .data import Climb

SMOOTHING = 20  # holds used by few climbs are pulled towards 0 ("average hold")


def angle_means(climbs: list[Climb]) -> dict[int, float]:
    by_angle = defaultdict(list)
    for c in climbs:
        by_angle[c.angle].append(c.difficulty)
    return {a: float(np.mean(v)) for a, v in by_angle.items()}


def fit_hold_scores(climbs: list[Climb], role_group: str) -> dict[int, float]:
    """role_group: "hand" (start/hand/finish) or "foot"."""
    means = angle_means(climbs)
    total, count = defaultdict(float), defaultdict(int)
    for c in climbs:
        residual = c.difficulty - means[c.angle]
        for h in c.holds:
            if (h.role == "foot") == (role_group == "foot"):
                total[h.placement] += residual
                count[h.placement] += 1
    return {p: total[p] / (count[p] + SMOOTHING) for p in total}


HOLD_FEATURE_NAMES = ["hand_hold_mean", "hand_hold_max", "hand_hold_min", "foot_hold_mean"]


def hold_features(c: Climb, hand_scores: dict[int, float], foot_scores: dict[int, float]) -> list[float]:
    hs = [hand_scores.get(h.placement, 0.0) for h in c.holds if h.role != "foot"]
    fs = [foot_scores.get(h.placement, 0.0) for h in c.holds if h.role == "foot"] or [0.0]
    return [float(np.mean(hs)), float(np.max(hs)), float(np.min(hs)), float(np.mean(fs))]


def hold_feature_matrix(fit_on: list[Climb], apply_to: list[Climb]) -> np.ndarray:
    hand, foot = fit_hold_scores(fit_on, "hand"), fit_hold_scores(fit_on, "foot")
    return np.array([hold_features(c, hand, foot) for c in apply_to], dtype=np.float32)


def out_of_fold_hold_features(climbs: list[Climb], n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Hold features for the training set, each climb scored by holds fit on the other folds.
    Folds are by climb (uuid), since one climb appears at several angles."""
    rng = np.random.default_rng(seed)
    uuids = sorted({c.uuid for c in climbs})
    fold_of = dict(zip(uuids, rng.integers(0, n_folds, len(uuids))))
    out = np.zeros((len(climbs), len(HOLD_FEATURE_NAMES)), dtype=np.float32)
    for k in range(n_folds):
        idx = [i for i, c in enumerate(climbs) if fold_of[c.uuid] == k]
        rest = [c for c in climbs if fold_of[c.uuid] != k]
        out[idx] = hold_feature_matrix(rest, [climbs[i] for i in idx])
    return out
