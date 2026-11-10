"""Grade predictor #1: gradient-boosted trees on hand-crafted features.

    python train_boosting.py

Trains on train, picks nothing on test, and reports on val and test:
  - mean per angle   (baseline: the wall angle alone)
  - boosting         (hand-crafted geometry features, kilter/features.py)
  - boosting + holds (plus how hard each hold is, kilter/hold_stats.py)
Saves the best model to models/boosting.joblib and scores to results/.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from kilter.data import PROJECT_DIR, read_processed
from kilter.features import FEATURE_NAMES, feature_matrix
from kilter.grades import score
from kilter.hold_stats import (HOLD_FEATURE_NAMES, angle_means, fit_hold_scores,
                               hold_feature_matrix, out_of_fold_hold_features)


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=1500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=40,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=0,
    )


def main() -> None:
    splits = {s: read_processed(s) for s in ("train", "val", "test")}
    y = {s: np.array([c.difficulty for c in cs]) for s, cs in splits.items()}
    results = {}

    # baseline: predict the average difficulty of training climbs at that angle
    means = angle_means(splits["train"])
    results["mean per angle"] = {s: score(y[s], [means[c.angle] for c in splits[s]])
                                 for s in ("val", "test")}

    # boosting on geometry features
    X = {s: feature_matrix(cs) for s, cs in splits.items()}
    t = time.time()
    geo = make_model().fit(X["train"], y["train"])
    print(f"boosting: {geo.n_iter_} trees in {time.time() - t:.0f}s")
    results["boosting"] = {s: score(y[s], geo.predict(X[s])) for s in ("val", "test")}

    # boosting on geometry + per-hold difficulty
    Xh = {"train": np.hstack([X["train"], out_of_fold_hold_features(splits["train"])])}
    for s in ("val", "test"):
        Xh[s] = np.hstack([X[s], hold_feature_matrix(splits["train"], splits[s])])
    t = time.time()
    full = make_model().fit(Xh["train"], y["train"])
    print(f"boosting + holds: {full.n_iter_} trees in {time.time() - t:.0f}s")
    results["boosting + holds"] = {s: score(y[s], full.predict(Xh[s])) for s in ("val", "test")}

    print(f"\n{'model':18s} {'split':5s} {'MAE pts':>8s} {'MAE V':>6s} {'exact V':>8s} {'±1 V':>6s}")
    for name, r in results.items():
        for s, m in r.items():
            print(f"{name:18s} {s:5s} {m['mae_points']:8.2f} {m['mae_v']:6.2f} "
                  f"{m['v_exact']:8.1%} {m['v_within_1']:6.1%}")

    # which features matter? shuffle one column at a time on val and see how much worse it gets
    names = FEATURE_NAMES + HOLD_FEATURE_NAMES
    imp = permutation_importance(full, Xh["val"], y["val"], n_repeats=3, random_state=0,
                                 scoring="neg_mean_absolute_error")
    ranked = sorted(zip(names, imp.importances_mean), key=lambda kv: -kv[1])
    print("\nmost important features (val MAE increase when shuffled):")
    for n, v in ranked[:10]:
        print(f"  {n:24s} +{v:.2f}")

    (PROJECT_DIR / "results").mkdir(exist_ok=True)
    (PROJECT_DIR / "models").mkdir(exist_ok=True)
    out = {"scores": results, "feature_importance": dict(ranked),
           "val_predictions": full.predict(Xh["val"]).round(3).tolist()}
    (PROJECT_DIR / "results" / "boosting_results.json").write_text(json.dumps(out, indent=1))
    joblib.dump({"model": full, "feature_names": names,
                 "hand_scores": fit_hold_scores(splits["train"], "hand"),
                 "foot_scores": fit_hold_scores(splits["train"], "foot")},
                PROJECT_DIR / "models" / "boosting.joblib")
    print("\nsaved models/boosting.joblib and results/boosting_results.json")


if __name__ == "__main__":
    main()
