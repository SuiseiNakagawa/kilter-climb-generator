"""Load Kilter Board climbs from the Hugging Face SQLite file and decode their holds.

Data: https://huggingface.co/datasets/Vilin97/KilterBoard (kilter_splits.sqlite).
All climbs are on layout 1 (Kilter Board Original). A climb's `frames` string
looks like "p1096r15p1113r15...": each pXXXX is a placement (a hold position on
the board) and each rYY is its role.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_DIR / "kilter_splits.sqlite"
SPLITS = ("train", "val", "test")
LAYOUT_ID = 1

# placement_roles for the Kilter Board Original (product 1)
ROLE_NAMES = {12: "start", 13: "hand", 14: "finish", 15: "foot"}
ROLES = ("start", "hand", "finish", "foot")
FRAME_RE = re.compile(r"p(\d+)r(\d+)")


@dataclass(frozen=True)
class Hold:
    placement: int
    role: str  # "start" | "hand" | "finish" | "foot"
    x: int     # board coordinates; holes are on an 8-unit grid
    y: int


@dataclass
class Climb:
    uuid: str
    name: str
    angle: int
    difficulty: float  # Kilter's averaged numeric grade: 10 = 4a/V0 ... 31 = 8b+/V14
    grade: str         # V-grade, e.g. "V3"
    font_grade: str    # e.g. "6a+"
    ascents: int
    quality: float
    holds: list[Hold]  # sorted bottom-to-top, then left-to-right


def _connect(db: Path | str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db)}?mode=ro", uri=True)


def load_board(db: Path | str = DEFAULT_DB) -> dict[int, tuple[int, int]]:
    """Map every placement on the board to its (x, y) position."""
    with closing(_connect(db)) as con:
        rows = con.execute(
            "SELECT p.id, h.x, h.y FROM placements p "
            "JOIN holes h ON h.id = p.hole_id WHERE p.layout_id = ?",
            (LAYOUT_ID,),
        )
        return {pid: (x, y) for pid, x, y in rows}


def load_grade_names(db: Path | str = DEFAULT_DB) -> dict[int, tuple[str, str]]:
    """Map Kilter's integer difficulty to (font_grade, v_grade), e.g. 16 -> ("6a", "V3")."""
    with closing(_connect(db)) as con:
        rows = con.execute("SELECT difficulty, boulder_name FROM difficulty_grades")
        return {d: tuple(name.split("/")) for d, name in rows}


def parse_frames(frames: str) -> list[tuple[int, int]]:
    """'p1096r15p1113r13' -> [(1096, 15), (1113, 13)]"""
    return [(int(p), int(r)) for p, r in FRAME_RE.findall(frames)]


def load_climbs(
    split: str,
    db: Path | str = DEFAULT_DB,
    min_ascents: int = 3,
    board: dict[int, tuple[int, int]] | None = None,
) -> tuple[list[Climb], Counter]:
    """Load one split and return (climbs, counts of rows dropped per reason).

    Each row is one climb at one angle, so the same climb can appear several
    times at different angles. The splits are by climb, so no climb is in two splits.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    board = board or load_board(db)
    grade_names = load_grade_names(db)
    dropped: Counter = Counter()
    climbs: list[Climb] = []

    with closing(_connect(db)) as con:
        rows = con.execute(
            f"SELECT uuid, name, angle, difficulty_numeric, ascensionist_count, "
            f"quality_average, frames FROM kilter_{split} WHERE layout_id = ?",
            (LAYOUT_ID,),
        )
        for uuid, name, angle, difficulty, ascents, quality, frames in rows:
            if ascents is None or ascents < min_ascents:
                dropped["too_few_ascents"] += 1
                continue
            pairs = parse_frames(frames)
            if any(r not in ROLE_NAMES for _, r in pairs):
                dropped["roles_from_other_board"] += 1
                continue
            if any(p not in board for p, _ in pairs):
                dropped["unknown_placement"] += 1
                continue
            if len({p for p, _ in pairs}) != len(pairs):
                dropped["duplicate_holds"] += 1
                continue
            roles = [ROLE_NAMES[r] for _, r in pairs]
            if not 1 <= roles.count("start") <= 2:
                dropped["needs_1_or_2_starts"] += 1
                continue
            if not 1 <= roles.count("finish") <= 2:
                dropped["needs_1_or_2_finishes"] += 1
                continue

            holds = sorted(
                (Hold(p, ROLE_NAMES[r], *board[p]) for p, r in pairs),
                key=lambda h: (h.y, h.x),
            )
            font, v = grade_names[int(round(difficulty))]
            climbs.append(
                Climb(
                    uuid=uuid,
                    name=name,
                    angle=int(angle),
                    difficulty=float(difficulty),
                    grade=v,
                    font_grade=font,
                    ascents=int(ascents),
                    quality=float(quality or 0.0),
                    holds=holds,
                )
            )
    return climbs, dropped


PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def read_processed(split: str) -> list[Climb]:
    """Read the climbs written by prepare_data.py (same filtering for every model)."""
    climbs = []
    with open(PROCESSED_DIR / f"{split}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            d.pop("ids")
            d["holds"] = [Hold(**h) for h in d["holds"]]
            climbs.append(Climb(**d))
    return climbs


def load_mirror_map(board: dict[int, tuple[int, int]]) -> dict[int, int]:
    """Map each placement to the placement at its left/right mirror position.

    The Kilter Original is symmetric around x = 72 (the Kilter app can mirror
    climbs), so a mirrored climb is the same climb flipped. The database's own
    mirrored_hole_id column is empty for this board, so this uses geometry.
    """
    by_xy = {xy: pid for pid, xy in board.items()}
    return {pid: by_xy[(144 - x, y)] for pid, (x, y) in board.items() if (144 - x, y) in by_xy}
