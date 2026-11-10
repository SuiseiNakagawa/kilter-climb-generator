"""Generate climbs with the trained GPT, one hold at a time.

With `rules=True` (the default) some tokens are blocked at each step, so every
climb follows the format of real ones:
  - only hold tokens or <eos> after the prompt
  - no board position used twice
  - holds go bottom to top (the order the model was trained on)
  - at most 2 start and 2 finish holds
  - <eos> only once there is at least one start and one finish
  - optionally (`max_move`), no hand hold farther than that from the previous
    hand hold, so climbs stay within a given reach
This doesn't make a climb good, only well-formed. With `rules=False` the
model can write anything, which shows how much of the format it learned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import ROLES, Climb, Hold
from .gpt import ClimbGPT
from .tokenizer import ClimbTokenizer


@dataclass
class Sample:
    angle: int
    grade: str            # requested grade
    tokens: list[int]     # everything after the prompt, including <eos> if written
    holds: list[Hold]
    finished: bool        # wrote <eos> before running out of length
    wellformed: bool      # only holds then <eos>, no repeats, 1-2 starts and 1-2 finishes

    def as_climb(self, name: str = "generated") -> Climb:
        return Climb(uuid="", name=name, angle=self.angle, difficulty=float("nan"),
                     grade=self.grade, font_grade="", ascents=0, quality=0.0, holds=self.holds)


class ClimbSampler:
    def __init__(self, model: ClimbGPT, tok: ClimbTokenizer, board: dict[int, tuple[int, int]]):
        self.model, self.tok, self.board = model.eval(), tok, board
        V = tok.vocab_size
        self.is_hold = torch.zeros(V, dtype=torch.bool)
        self.placement = torch.zeros(V, dtype=torch.long)
        self.role = torch.full((V,), -1, dtype=torch.long)
        self.key = torch.full((V,), float("inf"))
        self.xy = torch.zeros(V, 2)
        pids = sorted({p for p, _ in tok.holds})
        self.pidx = {p: i for i, p in enumerate(pids)}
        for i, s in enumerate(tok.itos):
            if ":" in s:
                p, r = s.split(":")
                x, y = board[int(p)]
                self.is_hold[i] = True
                self.placement[i] = self.pidx[int(p)]
                self.role[i] = ROLES.index(r)
                self.key[i] = y * 1000 + x   # same order as the training data: by y, then x
                self.xy[i] = torch.tensor([x, y], dtype=torch.float32)
        self.n_placements = len(pids)

    @torch.no_grad()
    def sample(self, prompts: list[tuple[int, str]], temperature: float = 1.0,
               rules: bool = True, seed: int | None = None,
               max_move: float | None = None) -> list[Sample]:
        """max_move: largest allowed distance between consecutive hand holds, in
        board units (about inches). None = no limit."""
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        tok, B = self.tok, len(prompts)
        idx = torch.tensor([tok.encode_prompt(a, gr) for a, gr in prompts])
        used = torch.zeros(B, self.n_placements, dtype=torch.bool)
        n_start = torch.zeros(B, dtype=torch.long)
        n_finish = torch.zeros(B, dtype=torch.long)
        last_key = torch.full((B,), -1.0)
        done = torch.zeros(B, dtype=torch.bool)
        last_hand = torch.full((B, 2), float("nan"))
        start_id, finish_id = ROLES.index("start"), ROLES.index("finish")
        is_hand = self.is_hold & (self.role != ROLES.index("foot"))
        eos = tok.eos_id

        while idx.shape[1] < self.model.block_size and not done.all():
            logits = self.model(idx)[:, -1, :] / temperature
            logits[:, tok.pad_id] = -float("inf")
            logits[:, tok.bos_id] = -float("inf")
            if rules:
                allowed = self.is_hold.expand(B, -1).clone()
                allowed &= ~used.gather(1, self.placement.expand(B, -1)) | ~self.is_hold
                allowed &= self.key.expand(B, -1) > last_key[:, None]
                allowed &= ~((self.role == start_id) & (n_start[:, None] >= 2))
                allowed &= ~((self.role == finish_id) & (n_finish[:, None] >= 2))
                if max_move is not None:
                    d = torch.cdist(last_hand.nan_to_num(0.0), self.xy)          # [B, V]
                    too_far = is_hand & (d > max_move) & ~last_hand[:, :1].isnan()
                    allowed &= ~too_far
                allowed[:, eos] = (n_start >= 1) & (n_finish >= 1)
                stuck = ~allowed.any(1)
                allowed[stuck, eos] = True          # nothing legal left: stop
                logits[~allowed] = -float("inf")
            nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g).squeeze(1)
            nxt[done] = tok.pad_id
            idx = torch.cat([idx, nxt[:, None]], 1)
            h = self.is_hold[nxt] & ~done
            used[h, self.placement[nxt[h]]] = True
            n_start += (h & (self.role[nxt] == start_id)).long()
            n_finish += (h & (self.role[nxt] == finish_id)).long()
            last_key = torch.where(h, self.key[nxt], last_key)
            hh = h & is_hand[nxt]
            last_hand[hh] = self.xy[nxt[hh]]
            done |= nxt == eos

        return [self._to_sample(a, gr, idx[i, 3:].tolist()) for i, (a, gr) in enumerate(prompts)]

    def _to_sample(self, angle: int, grade: str, toks: list[int]) -> Sample:
        tok = self.tok
        finished = tok.eos_id in toks
        toks = toks[: toks.index(tok.eos_id) + 1] if finished else [t for t in toks if t != tok.pad_id]
        body = toks[:-1] if finished else toks
        holds = tok.decode(body, self.board)["holds"]
        roles = [h.role for h in holds]
        wellformed = (finished
                  and all(self.is_hold[t] for t in body)
                  and len({h.placement for h in holds}) == len(holds)
                  and 1 <= roles.count("start") <= 2 and 1 <= roles.count("finish") <= 2)
        # keep the first use of each position and put holds in board order
        seen, uniq = set(), []
        for h in sorted(holds, key=lambda h: (h.y, h.x)):
            if h.placement not in seen:
                seen.add(h.placement)
                uniq.append(h)
        return Sample(angle, grade, toks, uniq, finished, bool(wellformed))
