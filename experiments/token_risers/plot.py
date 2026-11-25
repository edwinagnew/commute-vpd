"""P2.81 figure: per task, the share of the held-out log-prob gain carried by the top-k token types (solid, mean of 3
seeds, each seed a faint line) against the share of scored positions those same types occupy (dashed: the gain share
if every position gained alike). A curve far above its dashed line means the gain sits on a few types.
Reads results/risers.json; `--sketch` draws the card's planned figure from made-up numbers instead.

    ~/commute-agents/fleet/cpu-run --mem 2G -t 10 python3 experiments/token_risers/plot.py [--sketch]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

from commute.figures import figure, save

HERE = Path(__file__).parent
NAMES = {"lower": "lowercase", "arxiv": "arXiv", "swahili": "Swahili", "yoruba": "Yoruba"}
COLOURS = {"lower": "#4477aa", "arxiv": "#66ccee", "swahili": "#ee6677", "yoruba": "#aa3377"}


def curves(tasks: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, seeds in tasks.items():
        c = [s["concentration"] for s in seeds.values()]
        out[name] = {"k": c[0]["k"], "gain": [x["gain_share"] for x in c], "pos": [x["position_share"] for x in c]}
    return out


def draw(ax: Any, cv: dict[str, dict[str, Any]], title: str) -> list[dict[str, Any]]:
    rows = []
    for name, c in cv.items():
        col, k = COLOURS[name], c["k"]
        for g in c["gain"]:
            ax.plot(k, g, color=col, alpha=0.25, linewidth=0.8)
        mg, mp = [fmean(v) for v in zip(*c["gain"])], [fmean(v) for v in zip(*c["pos"])]
        ax.plot(k, mg, color=col, linewidth=1.8, label=f"{NAMES[name]}: gain share")
        ax.plot(k, mp, color=col, linewidth=1.0, linestyle="--", label=f"{NAMES[name]}: position share")
        rows += [{"task": name, "k": a, "gain_share_mean": b, "position_share_mean": d} for a, b, d in zip(k, mg, mp)]
    ax.set_xscale("log")
    ax.set_xlabel("top-k token types, ranked by summed gain")
    ax.set_ylabel("share of the held-out gain (solid)\nor of scored positions (dashed)")
    ax.set_title(title, loc="left", fontsize=9)
    return rows


def sketch() -> None:
    k = [1, 5, 10, 20, 50, 100, 200, 500]

    def fake(g: list[float], p: list[float]) -> dict[str, Any]:
        return {"k": k, "gain": [g, [x * 0.95 for x in g], [x * 1.05 for x in g]], "pos": [p, p, p]}

    diffuse, spread_pos = [0.03, 0.10, 0.16, 0.24, 0.37, 0.48, 0.60, 0.77], [0.03, 0.10, 0.15, 0.22, 0.33, 0.44, 0.56, 0.72]
    heavy = [0.25, 0.48, 0.58, 0.68, 0.80, 0.88, 0.93, 0.98]
    fig, axs = figure((9, 3.6), ncols=2)
    a = {"lower": fake(heavy, spread_pos), "arxiv": fake(diffuse, spread_pos), "swahili": fake(heavy, spread_pos), "yoruba": fake(heavy, spread_pos)}
    b = {"lower": fake(heavy, spread_pos), "arxiv": fake(diffuse, spread_pos), "swahili": fake(diffuse, spread_pos), "yoruba": fake(diffuse, spread_pos)}
    rows = draw(axs[0], a, "Rival a (made up): Swahili, Yoruba gain on a few types") + draw(axs[1], b, "Rival b (made up): spread like arXiv")
    axs[1].legend(fontsize=6, loc="lower right")
    fig.suptitle("SKETCH, made-up numbers: concentration of the fine-tune's held-out gain", fontsize=10)
    save(fig, HERE / "figures" / "sketch.png", n="made-up numbers; the run: 4 tasks x 3 seeds", data=rows)


def main() -> None:
    if "--sketch" in sys.argv:
        return sketch()
    tasks = json.loads((HERE / "results" / "risers.json").read_text())["tasks"]
    fig, ax = figure((5.5, 4.0))
    rows = draw(ax, curves(tasks), "Where each fine-tune's held-out gain sits")
    ax.legend(fontsize=6, loc="lower right")
    save(fig, HERE / "figures" / "concentration.png", n="4 tasks x 3 seeds (the Fig 5 KL-hi states); faint lines are seeds", data=rows)


if __name__ == "__main__":
    main()
