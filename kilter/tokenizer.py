"""Turn climbs into token sequences for a GPT-style model, and back.

A climb becomes:

    <bos> angle=40 V3 1096:foot 1113:foot ... 1392:finish <eos>

The angle and grade come first, so at generation time you prompt with
"<bos> angle=30 V3" and the model writes the holds. Holds are ordered
bottom-to-top (then left-to-right), so the model "sets" the climb upward.
Each hold token is one placement + role, so one token = one lit-up hold.
"""
from __future__ import annotations

import json
from pathlib import Path

from .data import Climb, Hold, ROLES

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIALS = (PAD, BOS, EOS)  # PAD is id 0


def angle_token(angle: int) -> str:
    return f"angle={angle}"


def hold_token(placement: int, role: str) -> str:
    return f"{placement}:{role}"


def _v_number(grade: str) -> int:
    return int(grade.lstrip("V"))


class ClimbTokenizer:
    def __init__(self, angles: list[int], grades: list[str], holds: list[tuple[int, str]]):
        self.angles = sorted(set(angles))
        self.grades = sorted(set(grades), key=_v_number)
        self.holds = sorted(set(holds), key=lambda h: (h[0], ROLES.index(h[1])))
        self.itos = (
            list(SPECIALS)
            + [angle_token(a) for a in self.angles]
            + self.grades
            + [hold_token(p, r) for p, r in self.holds]
        )
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id = (self.stoi[s] for s in SPECIALS)

    @classmethod
    def fit(cls, climbs: list[Climb]) -> "ClimbTokenizer":
        """Build the vocabulary from the training climbs only."""
        return cls(
            angles=[c.angle for c in climbs],
            grades=[c.grade for c in climbs],
            holds=[(h.placement, h.role) for c in climbs for h in c.holds],
        )

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def is_hold(self, token_id: int) -> bool:
        return ":" in self.itos[token_id]

    def encode_prompt(self, angle: int, grade: str) -> list[int]:
        return [self.bos_id, self.stoi[angle_token(angle)], self.stoi[grade]]

    def encode(self, climb: Climb) -> list[int]:
        """Raises KeyError if the climb uses a hold/role never seen in training."""
        return (
            self.encode_prompt(climb.angle, climb.grade)
            + [self.stoi[hold_token(h.placement, h.role)] for h in climb.holds]
            + [self.eos_id]
        )

    def decode(self, ids: list[int], board: dict[int, tuple[int, int]]) -> dict:
        """Token ids -> {"angle", "grade", "holds"}. Stops at <eos>, skips anything
        that isn't a hold after the prompt (a model may produce junk)."""
        angle = grade = None
        holds: list[Hold] = []
        for i in ids:
            tok = self.itos[i]
            if tok == EOS:
                break
            if tok.startswith("angle="):
                angle = angle if angle is not None else int(tok.split("=")[1])
            elif tok in self.grades:
                grade = grade or tok
            elif ":" in tok:
                p, role = tok.split(":")
                holds.append(Hold(int(p), role, *board[int(p)]))
        return {"angle": angle, "grade": grade, "holds": holds}

    def to_string(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(
            {"angles": self.angles, "grades": self.grades,
             "holds": [list(h) for h in self.holds]}, indent=1))

    @classmethod
    def load(cls, path: Path | str) -> "ClimbTokenizer":
        d = json.loads(Path(path).read_text())
        return cls(d["angles"], d["grades"], [tuple(h) for h in d["holds"]])
