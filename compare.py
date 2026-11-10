"""Compare the grade predictors and make figures for the write-up.

    python train_boosting.py && python train_gnn.py   # first
    python compare.py

Writes results/comparison.md and three charts:
  compare_models.png    % of test climbs graded within one V-grade, per model
  error_by_grade.png    where each model struggles (validation set)
  error_by_ascents.png  error vs how many people climbed it (label noise)
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from kilter.data import PROJECT_DIR, read_processed
from kilter.grades import to_v_number

RESULTS = PROJECT_DIR / "results"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BASELINE = "#a8a7a2"
COLORS = {"boosting": "#2a78d6", "boosting + holds": "#eb6834", "gnn": "#1baf7a"}
LABELS = {"mean per angle": "Angle only (baseline)", "boosting": "Boosting: geometry",
          "boosting + holds": "Boosting: geometry + holds", "gnn": "Graph neural network"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10,
})


def main() -> None:
    boost = json.loads((RESULTS / "boosting_results.json").read_text())
    gnn = json.loads((RESULTS / "gnn_results.json").read_text())
    scores = {**boost["scores"], **gnn["scores"]}

    # table
    lines = ["| Model | Split | MAE (points) | MAE (V-grades) | Exact V | Within ±1 V |",
             "|---|---|---|---|---|---|"]
    for name, r in scores.items():
        for s, m in r.items():
            lines.append(f"| {LABELS[name]} | {s} | {m['mae_points']:.2f} | {m['mae_v']:.2f} | "
                         f"{m['v_exact']:.1%} | {m['v_within_1']:.1%} |")
    (RESULTS / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # 1. headline: within one V-grade on test
    names = list(scores)
    vals = [scores[n]["test"]["v_within_1"] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.barh([LABELS[n] for n in names][::-1], vals[::-1], height=0.6,
                   color=[COLORS.get(n, BASELINE) for n in names][::-1])
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("test climbs graded within one V-grade (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Grade prediction accuracy", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / "compare_models.png", dpi=180)

    # 2 + 3. where do the two best models go wrong? (validation set)
    val = read_processed("val")
    y = np.array([c.difficulty for c in val])
    v_true = to_v_number(y)
    preds = {"boosting + holds": np.array(boost["val_predictions"]),
             "gnn": np.array(gnn["val_predictions"])}

    def by_group(groups, order):
        return {n: [np.mean(np.abs(to_v_number(p)[groups == g] - v_true[groups == g])) for g in order]
                for n, p in preds.items()}

    grades = [g for g in range(0, 12) if (v_true == g).sum() >= 50]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for n, errs in by_group(v_true, grades).items():
        ax.plot(grades, errs, color=COLORS[n], lw=2, marker="o", ms=5, label=LABELS[n])
    ax.set_xticks(grades, [f"V{g}" for g in grades])
    ax.set_xlabel("true grade")
    ax.set_ylabel("mean error (V-grades)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.set_title("Error by grade (validation)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / "error_by_grade.png", dpi=180)

    ascents = np.array([c.ascents for c in val])
    edges = [3, 5, 10, 25, 100, 10**9]
    names_b = ["3–4", "5–9", "10–24", "25–99", "100+"]
    bucket = np.digitize(ascents, edges[1:-1])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for n, errs in by_group(bucket, range(len(names_b))).items():
        ax.plot(names_b, errs, color=COLORS[n], lw=2, marker="o", ms=5, label=LABELS[n])
    counts = [int((bucket == i).sum()) for i in range(len(names_b))]
    ax.set_xticks(range(len(names_b)), [f"{b}\n(n={c:,})" for b, c in zip(names_b, counts)])
    ax.set_xlabel("number of people who climbed it")
    ax.set_ylabel("mean error (V-grades)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.set_title("Error vs. how established the grade is", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / "error_by_ascents.png", dpi=180)
    print("\nsaved results/comparison.md and 3 charts")


if __name__ == "__main__":
    main()
