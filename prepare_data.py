"""Build the tokenized dataset from kilter_splits.sqlite.

    python prepare_data.py                 # keeps climbs with >= 3 ascents
    python prepare_data.py --min-ascents 5 # stricter: fewer climbs, more reliable grades

Writes to data/processed/:
    tokenizer.json            the vocabulary (fit on train only)
    {split}.jsonl             one climb per line, with its holds and token ids
    {split}_ids.npy           int16 array [n_climbs, max_len], padded with 0
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from kilter.data import DEFAULT_DB, PROJECT_DIR, SPLITS, load_board, load_climbs
from kilter.tokenizer import ClimbTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--min-ascents", type=int, default=3)
    ap.add_argument("--out", default=PROJECT_DIR / "data" / "processed")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    board = load_board(args.db)
    splits = {}
    for split in SPLITS:
        climbs, dropped = load_climbs(split, args.db, args.min_ascents, board)
        splits[split] = climbs
        reasons = ", ".join(f"{k} {v}" for k, v in dropped.most_common())
        print(f"{split:5s}: kept {len(climbs):6d}   dropped: {reasons}")

    tok = ClimbTokenizer.fit(splits["train"])
    tok.save(out / "tokenizer.json")
    print(f"\nvocab: {tok.vocab_size} tokens = 3 special + {len(tok.angles)} angles "
          f"+ {len(tok.grades)} grades + {len(tok.holds)} hold:role pairs")

    for split, climbs in splits.items():
        rows, unseen = [], 0
        for c in climbs:
            try:
                ids = tok.encode(c)
            except KeyError:  # hold:role pair that never appears in train
                unseen += 1
                continue
            rows.append((c, ids))
        max_len = max(len(ids) for _, ids in rows)
        arr = np.zeros((len(rows), max_len), dtype=np.int16)
        with open(out / f"{split}.jsonl", "w") as f:
            for i, (c, ids) in enumerate(rows):
                arr[i, : len(ids)] = ids
                f.write(json.dumps({**asdict(c), "ids": ids}) + "\n")
        np.save(out / f"{split}_ids.npy", arr)
        lens = [len(ids) for _, ids in rows]
        extra = f", dropped {unseen} with unseen holds" if unseen else ""
        print(f"{split:5s}: {len(rows)} sequences, length mean {np.mean(lens):.1f} "
              f"max {max_len}{extra}")

    print("\ntrain grades:", dict(sorted(Counter(c.grade for c in splits['train']).items(),
                                         key=lambda kv: int(kv[0][1:]))))
    print("train angles:", dict(sorted(Counter(c.angle for c in splits['train']).items())))

    example = max(splits["train"], key=lambda c: c.ascents)
    print(f'\nexample, "{example.name}" ({example.grade} at {example.angle}°, '
          f"{example.ascents} ascents):\n  {tok.to_string(tok.encode(example))}")


if __name__ == "__main__":
    main()
