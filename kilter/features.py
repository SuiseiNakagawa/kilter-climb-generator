"""Hand-crafted features for one climb, for the boosting model.

Trees need one fixed-length row of numbers per climb, so a variable number of
holds gets summarised: how many holds of each kind, how high and wide the climb
is, how far apart consecutive hand holds are (a proxy for move size), and how
well the hands are supported by footholds. Board units: holes are 8 apart.
"""
from __future__ import annotations

import numpy as np

from .data import Climb

FEATURE_NAMES = [
    "angle",
    "n_holds", "n_start", "n_hand", "n_finish", "n_foot",
    "start_y", "finish_y", "height", "width", "hand_x_std",
    "n_moves", "move_mean", "move_max", "move_dy_max", "move_dx_max",
    "height_per_hand",
    "foot_y_mean_rel_start", "foot_lowest_rel_start", "foot_highest_rel_start",
    "hand_to_foot_mean", "hand_to_foot_max",
    "start_matched",
]


def climb_features(c: Climb) -> list[float]:
    hands = [h for h in c.holds if h.role != "foot"]       # start + hand + finish
    feet = [h for h in c.holds if h.role == "foot"]
    starts = [h for h in hands if h.role == "start"]
    finishes = [h for h in hands if h.role == "finish"]
    H = np.array([[h.x, h.y] for h in hands], float)      # already sorted bottom to top
    F = np.array([[h.x, h.y] for h in feet], float).reshape(-1, 2)

    start_y = np.mean([h.y for h in starts])
    finish_y = np.mean([h.y for h in finishes])
    moves = np.diff(H, axis=0)                             # consecutive hand holds
    dist = np.hypot(moves[:, 0], moves[:, 1]) if len(moves) else np.zeros(1)

    if len(F):
        # for every hand hold, the nearest foothold *below* it (feet go under hands)
        below = [F[F[:, 1] < y] for y in H[:, 1]]
        hand_to_foot = [np.min(np.hypot(*(b - p).T)) if len(b) else 100.0
                        for b, p in zip(below, H)]
        foot_rel = F[:, 1] - start_y
    else:
        hand_to_foot, foot_rel = [100.0], np.array([0.0])

    return [
        c.angle,
        len(c.holds), len(starts), len(hands) - len(starts) - len(finishes), len(finishes), len(feet),
        start_y, finish_y, finish_y - start_y, np.ptp(H[:, 0]), np.std(H[:, 0]),
        len(moves), dist.mean(), dist.max(),
        np.abs(moves[:, 1]).max() if len(moves) else 0.0,
        np.abs(moves[:, 0]).max() if len(moves) else 0.0,
        (finish_y - start_y) / len(hands),
        foot_rel.mean(), foot_rel.min(), foot_rel.max(),
        float(np.mean(hand_to_foot)), float(np.max(hand_to_foot)),
        float(len(starts) == 1),
    ]


def feature_matrix(climbs: list[Climb]) -> np.ndarray:
    return np.array([climb_features(c) for c in climbs], dtype=np.float32)
