"""P2.81 figure: per task and seed, how much more of the held-out log-prob gain the top-k token types carry than
their share of the scored positions (gain share minus position share), against k. 0 is the gain spread like the
types' frequency; the filed line between the two rivals is 0.15 at k = 20. arXiv's 5-step seed reads far above the
others (its top types' net share passes 1 because other types lost), so it is drawn clipped at the top edge with its
values written beside it, and the CSV keeps every value.
Reads results/risers.json; `--sketch` draws the card's planned figure from made-up numbers instead.

    ~/commute-agents/fleet/cpu-run --mem 2G -t 10 python3 experiments/token_risers/plot.py [--sketch]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from commute.figures import figure, reference, save

HERE = Path(__file__).parent
NAMES = {"lower": "lowercase", "arxiv": "arXiv", "swahili": "Swahili", "yoruba": "Yoruba"}
COLOURS = {"lower": "#4477aa", "arxiv": "#66ccee", "swahili": "#ee6677", "yoruba": "#aa3377"}
RATED = {"lower": "rated reuse", "arxiv": "rated reuse", "swahili": "rated new", "yoruba": "rated new"}
FILED, FILED_K, TOP = 0.15, 20, 0.95  # the rivals' line, the k it is read at, the y-axis top


def curves(tasks: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per task: k, and per seed its step and excess = gain share - position share at each k."""
    out = {}
    for name, seeds in tasks.items():
        c = {s: r["concentration"] for s, r in seeds.items()}
        out[name] = {"k": next(iter(c.values()))["k"], "seeds": {
            s: {"step": seeds[s]["step"], "excess": [g - p for g, p in zip(x["gain_share"], x["position_share"])]}
            for s, x in c.items()}}
    return out


def draw(ax: Any, cv: dict[str, dict[str, Any]], title: str) -> list[dict[str, Any]]:
    rows = []
    for name, c in cv.items():
        col, k = COLOURS[name], c["k"]
        steps = sorted(v["step"] for v in c["seeds"].values())
        label = f"{NAMES[name]}, {RATED[name]} ({steps[0]}-{steps[-1]} steps)"
        for i, (s, v) in enumerate(c["seeds"].items()):
            shown = [min(e, TOP) for e in v["excess"]]
            ax.plot(k, shown, color=col, marker="o", markersize=2.5, linewidth=1.2, label=label if i == 0 else None)
            if max(v["excess"]) > TOP:
                ax.annotate(f"{NAMES[name]} seed {s}, {v['step']} steps: {min(v['excess']):.2f} to {max(v['excess']):.2f},\n"
                            "off the top (other types lost)", (k[0], TOP), xytext=(2, -16), textcoords="offset points",
                            fontsize=6, color=col)
            elif name == "arxiv":
                ax.annotate(f"{v['step']} steps", (k[-1], shown[-1]), xytext=(3, 0), textcoords="offset points", fontsize=6, color=col, va="center")
            rows += [{"task": name, "seed": s, "step": v["step"], "k": a, "excess": e} for a, e in zip(k, v["excess"])]
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.annotate("0: gain spread like the types' frequency", (1, 0), xytext=(0, -9), textcoords="offset points", fontsize=6, color="#555555")
    ax.plot([FILED_K], [FILED], marker="_", markersize=18, markeredgewidth=2, color="#222222", linestyle="None",
            label=f"filed line between the rivals, {FILED} at top {FILED_K}")
    reference(ax, FILED_K, f"top {FILED_K}")
    ax.set_xscale("log")
    ax.set_xticks(cv[next(iter(cv))]["k"], [str(x) for x in cv[next(iter(cv))]["k"]])
    ax.set_ylim(-0.08, TOP + 0.05)
    ax.set_xlabel("top-k token types, ranked by summed gain")
    ax.set_ylabel("gain share minus position share\nof the top-k types")
    ax.set_title(title, loc="left", fontsize=9)
    return rows


def sketch() -> None:
    k = [1, 5, 10, 20, 50, 100, 200, 500]

    def fake(e: list[float]) -> dict[str, Any]:
        return {"k": k, "seeds": {s: {"step": 100, "excess": [x * f for x in e]} for s, f in (("0", 1.0), ("1", 0.95), ("2", 1.05))}}

    diffuse, heavy = [0.0, 0.0, 0.01, 0.02, 0.04, 0.04, 0.04, 0.05], [0.22, 0.38, 0.43, 0.46, 0.47, 0.44, 0.37, 0.26]
    fig, axs = figure((10, 3.8), ncols=2)
    a = {"lower": fake(heavy), "arxiv": fake(diffuse), "swahili": fake(heavy), "yoruba": fake(heavy)}
    b = {"lower": fake(heavy), "arxiv": fake(diffuse), "swahili": fake(diffuse), "yoruba": fake(diffuse)}
    rows = draw(axs[0], a, "Rival a (made up): Swahili, Yoruba gain on a few types") + draw(axs[1], b, "Rival b (made up): spread like arXiv")
    axs[1].legend(fontsize=6, loc="upper left")
    fig.suptitle("SKETCH, made-up numbers: excess concentration of the fine-tune's held-out gain", fontsize=10)
    save(fig, HERE / "figures" / "sketch.png", n="made-up numbers; the run: 4 tasks x 3 seeds", data=rows)


def main() -> None:
    if "--sketch" in sys.argv:
        return sketch()
    tasks = json.loads((HERE / "results" / "risers.json").read_text())["tasks"]
    fig, ax = figure((6.5, 5.0))
    rows = draw(ax, curves(tasks), "Is each fine-tune's held-out gain concentrated on a few token types?")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    save(fig, HERE / "figures" / "concentration.png", data=rows,
         n="4 tasks x 3 seeds, one line per seed (the Fig 5 KL-hi states), 64 held rows per task; "
           "position share = the share of scored positions the same top-k types occupy")


if __name__ == "__main__":
    main()
