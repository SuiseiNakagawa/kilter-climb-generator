"""Draw a climb on the board, coloured like the Kilter app's LEDs."""
from __future__ import annotations

import matplotlib.pyplot as plt

from .data import Hold

ROLE_COLORS = {"start": "#00DD00", "hand": "#00BFFF", "finish": "#FF00FF", "foot": "#FFA500"}


def draw_climb(ax, holds: list[Hold], board: dict[int, tuple[int, int]], title: str = "") -> None:
    xs, ys = zip(*board.values())
    ax.scatter(xs, ys, s=6, c="#cccccc", zorder=1)
    for role, color in ROLE_COLORS.items():
        pts = [(h.x, h.y) for h in holds if h.role == role]
        if pts:
            px, py = zip(*pts)
            ax.scatter(px, py, s=90, facecolors="none", edgecolors=color,
                       linewidths=2.2, zorder=2, label=role)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
