"""Train the graph neural network grade predictor (kilter/graph.py).

    python train_gnn.py               # ~25 epochs; minutes on a laptop CPU
    python train_gnn.py --epochs 5    # quick check

Keeps the epoch with the best validation MAE and reports val and test with
the same metrics as train_boosting.py. Saves models/gnn.pt and results/.
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from kilter.data import PROJECT_DIR, load_board, read_processed
from kilter.grades import score
from kilter.graph import ClimbGNN, build_graph, collate


def predict(model, graphs, mean, std, batch_size=1024) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            out.append(model(collate(graphs[i:i + batch_size])) * std + mean)
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    board = load_board()
    placement_index = {p: i for i, p in enumerate(sorted(board))}
    t = time.time()
    graphs = {s: [build_graph(c, placement_index) for c in read_processed(s)]
              for s in ("train", "val", "test")}
    print(f"built {sum(map(len, graphs.values()))} graphs in {time.time() - t:.0f}s, "
          f"avg {np.mean([g.edges.shape[1] for g in graphs['train']]):.0f} edges per climb")

    ys = np.array([g.y for g in graphs["train"]])
    mean, std = float(ys.mean()), float(ys.std())   # train on standardised difficulty

    model = ClimbGNN(len(placement_index), dim=args.dim, layers=args.layers)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(graphs["train"]) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    loss_fn = torch.nn.SmoothL1Loss()

    y_val = np.array([g.y for g in graphs["val"]])
    history, best = [], (np.inf, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t, losses = time.time(), []
        order = rng.permutation(len(graphs["train"]))
        for i in range(0, len(order), args.batch_size):
            b = collate([graphs["train"][j] for j in order[i:i + args.batch_size]])
            loss = loss_fn(model(b), (b["y"] - mean) / std)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        val = score(y_val, predict(model, graphs["val"], mean, std))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **val})
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.3f}  val MAE {val['mae_points']:.2f} pts "
              f"/ {val['mae_v']:.2f} V  ±1V {val['v_within_1']:.1%}  ({time.time() - t:.0f}s)")
        if val["mae_points"] < best[0]:
            best = (val["mae_points"], {k: v.clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[1])
    preds = {s: predict(model, graphs[s], mean, std) for s in ("val", "test")}
    scores = {s: score([g.y for g in graphs[s]], preds[s]) for s in ("val", "test")}
    for s, m in scores.items():
        print(f"best model, {s}: MAE {m['mae_points']:.2f} pts / {m['mae_v']:.2f} V, "
              f"exact V {m['v_exact']:.1%}, ±1 V {m['v_within_1']:.1%}")

    (PROJECT_DIR / "models").mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "mean": mean, "std": std,
                "placement_index": placement_index}, PROJECT_DIR / "models" / "gnn.pt")
    (PROJECT_DIR / "results" / "gnn_results.json").write_text(json.dumps(
        {"scores": {"gnn": scores}, "history": history,
         "val_predictions": preds["val"].round(3).tolist()}, indent=1))

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot([h["epoch"] for h in history], [h["mae_points"] for h in history], marker="o", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MAE (Kilter points)")
    ax.set_title("Graph model training")
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / "results" / "gnn_training.png", dpi=150)
    print("saved models/gnn.pt, results/gnn_results.json, results/gnn_training.png")


if __name__ == "__main__":
    main()
