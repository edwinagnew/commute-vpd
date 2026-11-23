"""E272 (cell P2.66) figure, CPU: the pairwise Kendall tau between every reading, baseline and the consensus tier over
tasks at matched KL (results/summary.json, set kl_hi), one heatmap. --sketch draws the card's planned figure with
made-up numbers, one panel per rival; --summary FILE plots another summary (the stub).

    ~/commute-agents/fleet/cpu-run python3 experiments/convergence_panel/plot.py [--summary FILE] [--sketch]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from commute.figures import figure, gap_heatmap, save
from commute.jobs import load_experiment

SC: Any = load_experiment("convergence_panel", "score")  # tau: Kendall tau-b, nan on a constant side

HERE = Path(__file__).resolve().parent
NAMES = {"tier": "consensus tier", "share": "re-gating share", "steering": "steering share", "erank": "effective rank",
         "E_b": "E_b over K", "kl": "KL from base", "adl": "ADL", "size": "update norm", "prior": "base prior"}  # fmt: skip


def matrix(means: dict[str, dict[str, float]], sign: dict[str, int], order: list[str]) -> list[list[float]]:
    """tau between every two rows of `means` (task -> value, oriented by sign) on their shared tasks; nan when fewer
    than 3 shared tasks or a constant side."""

    def t(a: str, b: str) -> float:
        ts = sorted(set(means[a]) & set(means[b]))
        x, y = [sign.get(a, 1) * means[a][k] for k in ts], [sign.get(b, 1) * means[b][k] for k in ts]
        return SC.tau(x, y) if len(ts) >= 3 else math.nan

    return [[t(a, b) for b in order] for a in order]


def panel(ax: Any, means: dict[str, dict[str, float]], sign: dict[str, int], title: str) -> list[dict[str, Any]]:
    order = [m for m in NAMES if m in means]
    M = matrix(means, sign, order)
    gap_heatmap(ax, M, [NAMES[m] for m in order], [NAMES[m] for m in order], label="Kendall tau over tasks (oriented: + = both read more reuse)", vmax=1.0, colorbar=False)
    ax.tick_params(axis="x", labelrotation=60)
    ax.set_title(title, fontsize=8)
    return [{"panel": title, "row": a, "col": b, "tau": M[i][j]} for i, a in enumerate(order) for j, b in enumerate(order)]


def real(path: Path) -> None:
    S = json.loads(path.read_text())
    P = json.loads((HERE / "results" / "predictions.json").read_text())
    s = S["kl_hi"]
    means = {m: v for m, v in s["means"].items() if v} | {"tier": S["consensus_tier"]}
    fig, ax = figure((6.5, 5.5))
    rows = panel(ax, means, P["rules"]["orientation"], "Do the readings rank tasks alike, and like the raters? (matched KL)")
    save(fig, HERE / "figures" / "pairwise_tau.png", n=f"n = {len(s['hope2']['tasks'])} tasks, each the mean of 3 seeds at one matched KL", data=rows)


def sketch() -> None:
    tasks = ["pile", "code", "lower", "british", "topic", "french", "yoruba", "squad", "conll"]
    tier = dict(zip(tasks, [3, 3, 3, 3, 2.5, 2, 1, 1.5, 2], strict=True))
    out = dict(zip(tasks, [5, 9, 1, 7, 3, 8, 2, 6, 4], strict=True))  # made-up output size order, unrelated to the tiers
    a = {"tier": tier, "share": {t: tier[t] + 0.1 * i for i, t in enumerate(tasks)}, "steering": {t: tier[t] - 0.05 * i for i, t in enumerate(tasks)},
         "E_b": {t: tier[t] + 0.02 * (i % 3) for i, t in enumerate(tasks)}, "kl": {t: out[t] for t in tasks}, "size": {t: out[t] + 0.5 * (i % 2) for i, t in enumerate(tasks)}}
    b = {"tier": tier, "share": {t: -out[t] + 0.1 * i for i, t in enumerate(tasks)}, "steering": {t: -out[t] for t in tasks},
         "E_b": {t: -out[t] - 0.3 * (i % 2) for i, t in enumerate(tasks)}, "kl": {t: out[t] for t in tasks}, "size": {t: out[t] + 0.5 * (i % 2) for i, t in enumerate(tasks)}}
    sign = {"kl": -1, "size": -1}
    fig, axes = figure((10, 4.6), ncols=2)
    rows = panel(axes[0], a, sign, "Rival a (made up): readings agree with the raters and each other")
    rows += panel(axes[1], b, sign, "Rival b (made up): readings follow output size, as the baselines do")
    fig.suptitle("SKETCH, made-up numbers: pairwise Kendall tau over nine tasks", fontsize=9)
    save(fig, HERE / "figures" / "card_sketch.png", n="n = 9 tasks (made up), each the mean of 3 seeds", data=rows)


if __name__ == "__main__":
    if "--sketch" in sys.argv:
        sketch()
    else:
        real(Path(sys.argv[sys.argv.index("--summary") + 1]) if "--summary" in sys.argv else HERE / "results" / "summary.json")
