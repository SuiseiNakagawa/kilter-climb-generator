"""Turn Kilter's numeric difficulty into V-grades and score predictions.

Kilter stores difficulty as a number (10 = 4a/V0 ... 31 = 8b+/V14). Models
predict that number; for reporting we also round it to a V-grade, because
"off by 1.8 points" means little to a climber and "within one V-grade" does.
"""
from __future__ import annotations

import numpy as np

from .data import DEFAULT_DB, load_grade_names

_GRADES = load_grade_names(DEFAULT_DB)
MIN_DIFFICULTY, MAX_DIFFICULTY = 10, 31


def to_v_number(difficulty) -> np.ndarray:
    """Numeric difficulty (float or array) -> V-grade as an int (V3 -> 3)."""
    d = np.clip(np.rint(np.asarray(difficulty, dtype=float)), MIN_DIFFICULTY, MAX_DIFFICULTY).astype(int)
    lookup = np.array([int(_GRADES[i][1].lstrip("V")) if i in _GRADES else -1
                       for i in range(MAX_DIFFICULTY + 1)])
    return lookup[d]


def score(y_true, y_pred) -> dict[str, float]:
    """MAE in Kilter points and in V-grades, exact V-grade accuracy, % within ±1 V."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    v_true, v_pred = to_v_number(y_true), to_v_number(y_pred)
    return {
        "mae_points": float(np.mean(np.abs(y_true - y_pred))),
        "mae_v": float(np.mean(np.abs(v_true - v_pred))),
        "v_exact": float(np.mean(v_true == v_pred)),
        "v_within_1": float(np.mean(np.abs(v_true - v_pred) <= 1)),
    }
