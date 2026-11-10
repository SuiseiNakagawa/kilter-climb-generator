"""Static figures for the README (PNG, in results/).

    python make_figures.py   # after train_boosting.py and train_gnn.py

  fig_graph_view.png     one test climb as the app shows it, and as the graph model sees it
  fig_predictions.png    15 test climbs with the true grade and both models' guesses
  fig_hold_heatmap.png   which holds show up in harder or easier climbs
"""
from __future__ import annotations

from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize

from kilter.data import PROJECT_DIR, load_board, read_processed
from kilter.features import feature_matrix
from kilter.grades import to_v_number
from kilter.graph import ClimbGNN, build_graph
from kilter.hold_stats import fit_hold_scores, hold_feature_matrix
from train_gnn import predict

RESULTS = PROJECT_DIR / "results"
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
BOARD, HOLE = "#121619", "#2a3237"
LED = {"start": "#3ddc5a", "hand": "#27c8f0", "finish": "#f24bd3", "foot": "#ffa21f"}
GOOD, OFF = "#1a7f4b", "#b4441f"

plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "font.size": 10})


def board_axes(ax, board):
    xs, ys = zip(*board.values())
    ax.set_facecolor(BOARD)
    ax.scatter(xs, ys, s=3, c=HOLE, zorder=1)
    ax.set_xlim(-28, 172)
    ax.set_ylim(-2, 182)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_holds(ax, climb, board, size=70):
    board_axes(ax, board)
    for h in climb.holds:
        ax.scatter(h.x, h.y, s=size * 2.2, facecolors="none", edgecolors=LED[h.role],
                   linewidths=4, alpha=0.25, zorder=2)
        ax.scatter(h.x, h.y, s=size, facecolors="none", edgecolors=LED[h.role],
                   linewidths=1.8, zorder=3)


def draw_graph(ax, climb, graph, board):
    board_axes(ax, board)
    pts = np.array([[h.x, h.y] for h in climb.holds])
    for a, b in {tuple(sorted(e)) for e in graph.edges.T.tolist()}:
        ax.plot(*pts[[a, b]].T, color="#dfe7ec", alpha=0.4, lw=0.9, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=60, c=[LED[h.role] for h in climb.holds],
               edgecolors=BOARD, linewidths=1, zorder=3)


def main() -> None:
    board = load_board()
    train, test = read_processed("train"), read_processed("test")

    # predictions on test from both models
    ckpt = torch.load(PROJECT_DIR / "models" / "gnn.pt", weights_only=False)
    pidx = ckpt["placement_index"]
    model = ClimbGNN(len(pidx), dim=ckpt["args"]["dim"], layers=ckpt["args"]["layers"])
    model.load_state_dict(ckpt["state_dict"])
    graphs = [build_graph(c, pidx) for c in test]
    gnn = predict(model, graphs, ckpt["mean"], ckpt["std"])
    boost = joblib.load(PROJECT_DIR / "models" / "boosting.joblib")
    boost_pred = boost["model"].predict(np.hstack([feature_matrix(test), hold_feature_matrix(train, test)]))
    vt = to_v_number([c.difficulty for c in test])
    vg, vb = to_v_number(gnn), to_v_number(boost_pred)

    # gallery: established grades (20+ ascents); one exact hit per grade V1-V10, plus telling cases
    order = sorted((i for i, c in enumerate(test) if c.ascents >= 20 and len(c.holds) <= 22),
                   key=lambda i: -test[i].ascents)
    picks: list[tuple[int, str]] = []

    def take(cond, tag, n=1):
        for i in [i for i in order if i not in dict(picks) and cond(i)][:n]:
            picks.append((i, tag))

    for g in range(1, 11):
        take(lambda i, g=g: vt[i] == g and vg[i] == g, "graph exact")
    take(lambda i: abs(vg[i] - vt[i]) <= 1 and abs(vb[i] - vt[i]) >= 3, "graph much closer", 2)
    take(lambda i: abs(vb[i] - vt[i]) <= 1 and abs(vg[i] - vt[i]) >= 2, "boosting closer")
    take(lambda i: abs(vg[i] - vt[i]) >= 3 and abs(vb[i] - vt[i]) >= 3, "both miss", 2)

    def verdict(pred, true):
        d = int(pred - true)
        return ("exact", GOOD) if d == 0 else (f"{d:+d} V", GOOD if abs(d) <= 1 else OFF)

    # 1. one climb, two views
    feat = next((i for i, _ in picks if test[i].name == "Black Rock Desert"), picks[0][0])
    c = test[feat]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.9))
    draw_holds(axes[0], c, board, size=90)
    draw_graph(axes[1], c, graphs[feat], board)
    axes[0].set_title("What the climber sees", loc="left", fontsize=11, fontweight="bold")
    axes[1].set_title(f"What the graph model sees ({len(c.holds)} nodes, "
                      f"{graphs[feat].edges.shape[1] // 2} edges)", loc="left", fontsize=11, fontweight="bold")
    gv, gc = verdict(vg[feat], vt[feat])
    bv, bc = verdict(vb[feat], vt[feat])
    fig.suptitle(f'"{c.name}", {c.angle}°, {c.ascents:,} ascents', x=0.02, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.02, 0.035, f"Community grade V{vt[feat]}   ·   graph model V{vg[feat]} ({gv})   ·   "
                          f"boosting V{vb[feat]} ({bv})", fontsize=10.5, color=INK_2)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=col, mew=2, label=r)
               for r, col in LED.items()]
    fig.legend(handles=handles, loc="lower right", ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.99, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(RESULTS / "fig_graph_view.png", dpi=170)
    plt.close(fig)

    # 2. gallery of predictions
    cols = 5
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 3.6 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (i, tag) in zip(axes.ravel(), picks):
        ax.axis("on")
        draw_holds(ax, test[i], board, size=34)
        name = test[i].name if len(test[i].name) <= 22 else test[i].name[:21] + "…"
        ax.set_title(f"{tag.upper()}\n{name}", fontsize=8.5, loc="left", color=INK, linespacing=1.4)
        gv, gc = verdict(vg[i], vt[i])
        bv, bc = verdict(vb[i], vt[i])
        ax.text(0, -0.07, f"true V{vt[i]}", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        ax.text(0, -0.17, f"graph V{vg[i]}", transform=ax.transAxes, fontsize=8.5, va="top", color=INK_2)
        ax.text(0.55, -0.17, gv, transform=ax.transAxes, fontsize=8.5, va="top", color=gc)
        ax.text(0, -0.27, f"boost V{vb[i]}", transform=ax.transAxes, fontsize=8.5, va="top", color=INK_2)
        ax.text(0.55, -0.27, bv, transform=ax.transAxes, fontsize=8.5, va="top", color=bc)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.08, wspace=0.08, hspace=0.62)
    fig.savefig(RESULTS / "fig_predictions.png", dpi=150)
    plt.close(fig)

    # 3. hold difficulty heatmap
    scores = fit_hold_scores(train, "hand")
    uses = Counter(h.placement for c in train for h in c.holds if h.role != "foot")
    cmap = LinearSegmentedColormap.from_list("easy_hard", ["#3987e5", "#60686e", "#e66767"])
    norm = Normalize(-1.5, 1.5)  # V-grades (2 points = 1 V)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    board_axes(ax, board)
    pts = [(board[p], scores[p] / 2) for p in scores if uses[p] >= 30 and p in board]
    xy = np.array([p for p, _ in pts])
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=[v for _, v in pts], cmap=cmap, norm=norm,
                    s=48, edgecolors=BOARD, linewidths=0.6, zorder=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, ticks=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    cb.set_label("climbs using this hand hold vs. average (V-grades)", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title("Which holds make a climb harder", loc="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.02, "Red: shows up in harder climbs than usual for the angle. Blue: easier.\n"
                         "Holds used by fewer than 30 training climbs are left out.",
             fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(RESULTS / "fig_hold_heatmap.png", dpi=170)
    plt.close(fig)
    print("saved results/fig_graph_view.png, fig_predictions.png, fig_hold_heatmap.png")


if __name__ == "__main__":
    main()
