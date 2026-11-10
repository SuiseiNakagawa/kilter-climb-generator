"""Sanity check: round-trip climbs through the tokenizer and plot a few.

    python prepare_data.py   # first
    python show_climbs.py    # -> results/example_climbs.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from kilter.data import PROCESSED_DIR, PROJECT_DIR, load_board, read_processed
from kilter.plot import ROLE_COLORS, draw_climb
from kilter.tokenizer import ClimbTokenizer


def main() -> None:
    board = load_board()
    tok = ClimbTokenizer.load(PROCESSED_DIR / "tokenizer.json")
    train = read_processed("train")

    # every training climb should survive encode -> decode unchanged
    for c in train:
        back = tok.decode(tok.encode(c), board)
        assert (back["angle"], back["grade"], back["holds"]) == (c.angle, c.grade, c.holds), c.uuid
    print(f"round trip OK for all {len(train)} training climbs")

    # the most-repeated climb at 40° for a spread of grades
    picks = []
    for grade in ["V0", "V2", "V4", "V6", "V8", "V10"]:
        pool = [c for c in train if c.grade == grade and c.angle == 40]
        if pool:
            picks.append(max(pool, key=lambda c: c.ascents))

    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 4.2))
    for ax, c in zip(axes, picks):
        draw_climb(ax, c.holds, board, f"{c.name}\n{c.grade} ({c.font_grade}) at {c.angle}°, "
                                       f"{c.ascents} ascents")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=col, mew=2, label=r)
               for r, col in ROLE_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = PROJECT_DIR / "results" / "example_climbs.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
