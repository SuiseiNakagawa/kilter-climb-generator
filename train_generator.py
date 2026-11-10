"""Train the climb generator (kilter/gpt.py) on data/processed/train_ids.npy.

    python train_generator.py              # 12 epochs; ~55 min on a 2-core CPU, faster on a laptop
    python train_generator.py --epochs 3   # quick check

Keeps the epoch with the lowest validation loss (epoch 7 of 12 in our run; after
that the model starts memorising training climbs). Saves models/generator.pt and
results/generator_training.png.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from kilter.data import PROCESSED_DIR, PROJECT_DIR
from kilter.gpt import ClimbGPT, n_params
from kilter.tokenizer import ClimbTokenizer


@torch.no_grad()
def evaluate(model, ids: torch.Tensor, batch_size: int = 1024) -> float:
    model.eval()
    losses, counts = [], []
    for i in range(0, len(ids), batch_size):
        b = ids[i:i + batch_size]
        n = int(((b[:, 1:] != 0).sum() - 2 * len(b)).item())  # hold + eos targets in batch
        losses.append(model.loss(b).item() * n)
        counts.append(n)
    return sum(losses) / sum(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tok = ClimbTokenizer.load(PROCESSED_DIR / "tokenizer.json")
    train = torch.from_numpy(np.load(PROCESSED_DIR / "train_ids.npy").astype(np.int64))
    val = torch.from_numpy(np.load(PROCESSED_DIR / "val_ids.npy").astype(np.int64))
    block = max(train.shape[1], val.shape[1]) + 4
    pad = lambda x: torch.nn.functional.pad(x, (0, block - x.shape[1]))
    train, val = pad(train), pad(val)

    model = ClimbGPT(tok.vocab_size, block_size=block, dim=args.dim, layers=args.layers,
                     heads=args.heads, dropout=args.dropout, pad_id=tok.pad_id)
    print(f"{n_params(model):,} parameters; {len(train):,} training climbs, "
          f"{int((train != 0).sum()):,} tokens")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)

    history, best = [], (float("inf"), None, 0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t, losses = time.time(), []
        for i in torch.randperm(len(train)).split(args.batch_size):
            b = train[i]
            b = b[:, : int((b != 0).sum(1).max())]  # trim padding to the longest climb
            loss = model.loss(b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        v = evaluate(model, val)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": v})
        print(f"epoch {epoch:2d}  train {np.mean(losses):.3f}  val {v:.3f}  "
              f"(perplexity {math.exp(v):.0f} of {tok.vocab_size} tokens)  {time.time() - t:.0f}s", flush=True)
        if v < best[0]:
            best = (v, {k: x.clone() for k, x in model.state_dict().items()}, epoch)

    print(f"best epoch {best[2]}: val loss {best[0]:.3f}")
    (PROJECT_DIR / "models").mkdir(exist_ok=True)
    torch.save({"state_dict": best[1], "args": vars(args), "block_size": block,
                "vocab_size": tok.vocab_size}, PROJECT_DIR / "models" / "generator.pt")
    (PROJECT_DIR / "results" / "generator_training.json").write_text(json.dumps(history, indent=1))

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train", marker="o", ms=3)
    ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="validation", marker="o", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss per hold (nats)")
    ax.set_title("Generator training")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / "results" / "generator_training.png", dpi=150)
    print("saved models/generator.pt and results/generator_training.png")


if __name__ == "__main__":
    main()
