"""Step 0's figure: each saved update's share left outside the subcomponents beside a random change of the same
per-matrix norms (Table 1), on the new decomposition p-9707d6ff.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_instrument_check/plot_step0.py            (after step 0)
    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_instrument_check/plot_step0.py --sketch   (the card's sketch, made-up numbers)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as FG

HERE = Path(__file__).resolve().parent
REAL = FG.Style("update, least-squares split", "#0072B2", "o")
XLABEL = "share of squared norm the split must leave outside"
# Made-up numbers for the card: (update, real, null) per rival.
SKETCH = {
    "If placement is free": [
        ("RL s0", 0.093, 0.097), ("shuffled s0", 0.091, 0.095), ("first maths", 0.088, 0.091), ("TOFU s42", 0.104, 0.106)],
    "If placement is constrained": [
        ("RL s0", 0.021, 0.097), ("shuffled s0", 0.044, 0.095), ("first maths", 0.012, 0.091), ("TOFU s42", 0.030, 0.106)],
}


def draw(ax: Axes, recs: list[dict[str, Any]], iid: float, title: str) -> None:
    """One row per update: its own outside share (blue) and the matched random change's (grey X)."""
    rnd = FG.METHODS["random"]
    for i, r in enumerate(recs):
        ax.plot([r["real"], r["null"]], [i, i], color="#bbbbbb", linewidth=0.8, zorder=1)
        ax.scatter([r["null"]], [i], color=rnd.color, marker=rnd.marker, s=26, zorder=2,
                   label=rnd.label if i == 0 else None)
        ax.scatter([r["real"]], [i], color=REAL.color, marker=REAL.marker, s=26, zorder=3,
                   label=REAL.label if i == 0 else None)
    ax.set_yticks(range(len(recs)), [r["update"] for r in recs])
    ax.set_ylim(len(recs) - 0.5, -1.0)  # rows top-down, with room above the first for the reference label
    ax.set_xlim(0, max(0.15, 1.1 * max(max(r["real"], r["null"]) for r in recs)))
    FG.reference(ax, iid, "iid random, whole model")
    ax.set_xlabel(XLABEL)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7, loc="upper left", frameon=False)


def sketch(out: Path = HERE / "figures") -> None:
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    data: list[dict[str, Any]] = []
    for ax, (title, rows) in zip(axes, SKETCH.items(), strict=True):
        recs = [{"update": u, "real": a, "null": b} for u, a, b in rows]
        draw(ax, recs, 0.10, title)
        data += [{"panel": title, **r} for r in recs]
    fig.suptitle("Planned figure (made-up numbers): do real updates sit inside the subcomponents more than chance?",
                 fontsize=9)
    FG.save(fig, out / "sketch_step0.png", n="made-up numbers; 4 of the 15 saved updates, 196 matrices pooled",
            data=data)


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    t1 = json.loads((results / "step0.json").read_text())["table1"]
    recs = [{"update": n, "real": u["real"], "null": u["null"], "ratio": u["ratio"]} for n, u in t1["updates"].items()]
    fig, ax = FG.figure((5.0, 5.0))
    draw(ax, recs, t1["iid_null"], f"Reading: placement {t1['placement']['reading']}")
    fig.suptitle("Share of each update the least-squares split cannot hold, beside a random change", fontsize=9)
    FG.save(fig, out / "outside_share.png", n="15 saved updates; 196 matrices pooled by each update's norms; new decomposition",
            data=recs)


if __name__ == "__main__":
    sketch() if "--sketch" in sys.argv else main()
