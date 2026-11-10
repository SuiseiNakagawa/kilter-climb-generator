"""Generate climbs with the trained GPT and measure how good they are.

    python train_generator.py      # first
    python evaluate_generator.py   # ~5-10 min

1. Rules: how often does the raw model write a well-formed climb by itself?
2. Grade control: ask for V0-V10 at 25° and 40°, grade every climb with the
   graph model, and compare requested vs. predicted grade.
3. Temperature: how plausibility, novelty and grade control trade off as
   sampling gets more random.

Writes results/generator_results.json, results/generated_climbs.jsonl and the
figures fig_generated.png, fig_grade_control.png, fig_temperature.png.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from kilter.climb_checks import CHECKS, Novelty, Plausibility
from kilter.critic import GradeCritic
from kilter.data import PROCESSED_DIR, PROJECT_DIR, load_board, read_processed
from kilter.gpt import ClimbGPT
from kilter.grades import to_v_number
from kilter.sampling import ClimbSampler
from kilter.tokenizer import ClimbTokenizer

RESULTS = PROJECT_DIR / "results"
GRADES = list(range(0, 11))
ANGLES = [25, 40]
NEAR_COPY = 0.8

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a8a7a2"
BOARD, HOLE = "#121619", "#2a3237"
LED = {"start": "#3ddc5a", "hand": "#27c8f0", "finish": "#f24bd3", "foot": "#ffa21f"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10,
})


def load_generator() -> ClimbGPT:
    ckpt = torch.load(PROJECT_DIR / "models" / "generator.pt", weights_only=False)
    a = ckpt["args"]
    model = ClimbGPT(ckpt["vocab_size"], block_size=ckpt["block_size"], dim=a["dim"],
                     layers=a["layers"], heads=a["heads"], dropout=0.0)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def generate(sampler, prompts, temperature, rules=True, seed=0, batch=400):
    out = []
    for i in range(0, len(prompts), batch):
        out += sampler.sample(prompts[i:i + batch], temperature, rules, seed=seed + i)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="climbs per (angle, grade) request")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    board = load_board()
    tok = ClimbTokenizer.load(PROCESSED_DIR / "tokenizer.json")
    train, test = read_processed("train"), read_processed("test")
    sampler = ClimbSampler(load_generator(), tok, board)
    critic, plaus, novelty = GradeCritic(), Plausibility(train), Novelty(train)
    results = {}

    def assess(samples):
        climbs = [s.as_climb() for s in samples]
        v_pred = to_v_number(critic.predict(climbs))
        checks = plaus.check(climbs)
        jac, match = novelty.nearest(climbs)
        return climbs, v_pred, checks, jac, match

    # ---- 1. does the raw model follow the format?
    prompts = [(40, f"V{g}") for g in range(2, 9)] * 60
    raw = generate(sampler, prompts, 1.0, rules=False, seed=args.seed)
    ruled = generate(sampler, prompts, 1.0, rules=True, seed=args.seed)
    results["rules"] = {
        "raw_wellformed": float(np.mean([s.wellformed for s in raw])),
        "raw_finished": float(np.mean([s.finished for s in raw])),
        "with_rules_plausible": float(plaus.check([s.as_climb() for s in ruled]).all(1).mean()),
        "raw_plausible_if_wellformed": float(plaus.check([s.as_climb() for s in raw if s.wellformed]).all(1).mean()),
    }
    print("rules:", {k: round(v, 3) for k, v in results["rules"].items()})

    # ---- 2. grade control
    prompts = [(a, f"V{g}") for a in ANGLES for g in GRADES for _ in range(args.n)]
    samples = generate(sampler, prompts, 1.0, seed=args.seed + 1)
    climbs, v_pred, checks, jac, match = assess(samples)
    v_req = np.array([int(s.grade[1:]) for s in samples])
    ok = checks.all(1)
    shuffled = np.random.default_rng(0).permutation(v_req)
    test_true = to_v_number([c.difficulty for c in test])
    test_pred = to_v_number(critic.predict(test))
    results["grade_control"] = {
        "n": len(samples),
        "within_1": float(np.mean(np.abs(v_pred - v_req) <= 1)),
        "mae_v": float(np.mean(np.abs(v_pred - v_req))),
        "spearman": float(spearmanr(v_req, v_pred).statistic),
        "shuffled_within_1": float(np.mean(np.abs(v_pred - shuffled) <= 1)),
        "real_test_within_1": float(np.mean(np.abs(test_pred - test_true) <= 1)),
        "plausible": float(ok.mean()),
        "plausible_by_check": dict(zip(CHECKS, checks.mean(0).round(4).tolist())),
        "near_copy": float(np.mean(jac >= NEAR_COPY)),
        "exact_copy": float(np.mean(jac >= 0.999)),
        "median_overlap": float(np.median(jac)),
        "by_grade": {g: {"median": float(np.median(v_pred[v_req == g])),
                         "q25": float(np.percentile(v_pred[v_req == g], 25)),
                         "q75": float(np.percentile(v_pred[v_req == g], 75)),
                         "within_1": float(np.mean(np.abs(v_pred[v_req == g] - g) <= 1))}
                     for g in GRADES},
        "real_by_grade": {g: {"median": float(np.median(test_pred[test_true == g])),
                              "q25": float(np.percentile(test_pred[test_true == g], 25)),
                              "q75": float(np.percentile(test_pred[test_true == g], 75))}
                          for g in GRADES if (test_true == g).sum() >= 30},
    }
    real_jac, _ = novelty.nearest(test[:3000])
    results["real_test_novelty"] = {"near_copy": float(np.mean(real_jac >= NEAR_COPY)),
                                    "exact_copy": float(np.mean(real_jac >= 0.999)),
                                    "median_overlap": float(np.median(real_jac))}
    gc = results["grade_control"]
    print(f"grade control: {gc['within_1']:.1%} within one grade of the request "
          f"(shuffled requests: {gc['shuffled_within_1']:.1%}; real climbs: {gc['real_test_within_1']:.1%}), "
          f"Spearman {gc['spearman']:.2f}")
    print(f"plausible {gc['plausible']:.1%}; near copies {gc['near_copy']:.1%} "
          f"(real test climbs: {results['real_test_novelty']['near_copy']:.1%}); "
          f"exact copies {gc['exact_copy']:.1%} (real: {results['real_test_novelty']['exact_copy']:.1%})")

    with open(RESULTS / "generated_climbs.jsonl", "w") as f:
        for s, vp, k, j, m in zip(samples, v_pred, ok, jac, match):
            f.write(json.dumps({"angle": s.angle, "requested": s.grade, "predicted": f"V{vp}",
                                "plausible": bool(k), "overlap_with_closest": round(float(j), 3),
                                "closest_real_climb": m.name,
                                "holds": [asdict(h) for h in s.holds]}) + "\n")

    # ---- 3. temperature
    temps = [0.6, 0.8, 1.0, 1.2, 1.5]
    prompts_t = [(40, f"V{g}") for g in range(2, 9) for _ in range(40)]
    sweep = []
    for t in temps:
        st = generate(sampler, prompts_t, t, seed=args.seed + 2)
        _, vp, ch, jt, _ = assess(st)
        vr = np.array([int(s.grade[1:]) for s in st])
        sets = Counter((s.grade, frozenset(h.placement for h in s.holds)) for s in st)
        sweep.append({"temperature": t, "plausible": float(ch.all(1).mean()),
                      "new": float(np.mean(jt < NEAR_COPY)), "exact_copy": float(np.mean(jt >= 0.999)),
                      "within_1": float(np.mean(np.abs(vp - vr) <= 1)),
                      "repeats": 1 - len(sets) / len(st)})
        print("temperature", {k: round(v, 3) for k, v in sweep[-1].items()})
    results["temperature"] = sweep
    (RESULTS / "generator_results.json").write_text(json.dumps(results, indent=1))

    figures(samples, climbs, v_pred, ok, jac, board, results)
    print("saved results/generator_results.json, generated_climbs.jsonl and 3 figures")


def figures(samples, climbs, v_pred, ok, jac, board, results) -> None:
    # gallery: one random climb per cell among those that pass the checks and aren't near copies
    rng = np.random.default_rng(1)
    show = [1, 3, 5, 7, 9]
    fig, axes = plt.subplots(len(ANGLES), len(show), figsize=(2.6 * len(show), 3.5 * len(ANGLES)))
    xs, ys = zip(*board.values())
    for r, angle in enumerate(ANGLES[::-1]):
        for c, g in enumerate(show):
            ax = axes[r, c]
            pool = [i for i, s in enumerate(samples)
                    if s.angle == angle and s.grade == f"V{g}" and ok[i] and jac[i] < NEAR_COPY]
            i = int(rng.choice(pool))
            ax.set_facecolor(BOARD)
            ax.scatter(xs, ys, s=3, c=HOLE)
            for h in climbs[i].holds:
                ax.scatter(h.x, h.y, s=75, facecolors="none", edgecolors=LED[h.role], linewidths=4, alpha=0.25)
                ax.scatter(h.x, h.y, s=34, facecolors="none", edgecolors=LED[h.role], linewidths=1.8)
            ax.set_xlim(-28, 172)
            ax.set_ylim(-2, 182)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"asked for V{g} at {angle}°", loc="left", fontsize=9.5, fontweight="bold")
            d = int(v_pred[i]) - g
            col = "#1a7f4b" if abs(d) <= 1 else "#b4441f"
            ax.text(0, -0.06, f"graph model says V{v_pred[i]}", transform=ax.transAxes, fontsize=8.5, va="top", color=col)
            ax.text(0, -0.15, f"{jac[i]:.0%} same holds as closest real climb", transform=ax.transAxes,
                    fontsize=8, va="top", color=INK_2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.07, wspace=0.08, hspace=0.42)
    fig.savefig(RESULTS / "fig_generated.png", dpi=150)
    plt.close(fig)

    # requested vs predicted grade
    gc = results["grade_control"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([0, 10], [0, 10], color=GRID, lw=1.5, ls="--", zorder=1)
    for key, color, label, dx in (("real_by_grade", GREY, "real test climbs (true grade)", 0.12),
                                  ("by_grade", BLUE, "generated climbs (requested grade)", -0.12)):
        g = sorted(int(k) for k in gc[key])
        med = [gc[key][k]["median"] for k in g]
        lo = [gc[key][k]["q25"] for k in g]
        hi = [gc[key][k]["q75"] for k in g]
        x = np.array(g) + dx
        ax.vlines(x, lo, hi, color=color, lw=3, alpha=0.5)
        ax.plot(x, med, color=color, marker="o", ms=6, lw=2, label=label)
    ax.set_xticks(GRADES, [f"V{g}" for g in GRADES])
    ax.set_yticks(GRADES, [f"V{g}" for g in GRADES])
    ax.set_xlabel("grade asked for (generated) or true grade (real)")
    ax.set_ylabel("grade according to the graph model")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Does the generator hit the grade it was asked for?", loc="left", fontweight="bold")
    fig.text(0.01, 0.01, "Dots: median. Bars: middle 50%. Dashed: perfect agreement.", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RESULTS / "fig_grade_control.png", dpi=170)
    plt.close(fig)

    # temperature tradeoff
    sw = results["temperature"]
    t = [s["temperature"] for s in sw]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key, color, label in (("plausible", BLUE, "plausible (passes all checks)"),
                              ("new", ORANGE, "new (< 80% overlap with any real climb)"),
                              ("within_1", AQUA, "within one grade of the request")):
        v = [100 * s[key] for s in sw]
        ax.plot(t, v, color=color, marker="o", ms=6, lw=2, label=label)
    ax.set_xticks(t)
    ax.set_xlabel("sampling temperature (higher = more random)")
    ax.set_ylabel("% of generated climbs")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("More randomness costs quality without adding novelty", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_temperature.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
