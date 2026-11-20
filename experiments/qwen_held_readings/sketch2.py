"""The card's sketch for P3.13 run 2 (card2.md): made-up numbers in summary2's shape, drawn by plot2.draw, two rows per
rival's outcome. Not a result. S near run 1's (states 0.98-0.997, random changes ~0.8); E187's c as filed.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_held_readings/sketch2.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as F
from commute.jobs import load_experiment

PLOT = load_experiment("qwen_held_readings", "plot2")
HERE = Path(__file__).resolve().parent
ST = json.loads((HERE / "results" / "run2" / "predictions.json").read_text())["settings"]


def made_up(gap: float, shrink: float) -> dict[str, Any]:
    """Each reuse side at 0.99, its new side at 0.99 - gap; the larger side times c loses `shrink` of S (E205's sign)."""
    pairs = {}
    for name, (reuse, new) in ST["pairs"].items():
        saves = {}
        for save in ST["saves"]:
            r = ST["rescale"][f"{name}/{save}"]
            base = {reuse: 0.99, new: 0.99 - gap}
            S = {a: {"saved": [base[a] + 0.003 * (s - 1) for s in range(3)],
                     "random": [0.80 + 0.002 * s for s in range(3)],
                     "c": [base[a] - shrink + 0.003 * (s - 1) for s in range(3)] if a == r["larger"] else [None] * 3,
                     "c_random": [0.78 + 0.002 * s for s in range(3)] if a == r["larger"] else [None] * 3}
                 for a in (reuse, new)}
            saves[save] = {"larger": r["larger"], "c": r["c"], "S": S}
        pairs[name] = {"reuse_side": reuse, "new_side": new, "saves": saves,
                       "outcome": "separates" if gap > 0 else "does not separate"}
    return {"pairs": pairs, "outcome": "rival A" if gap > 0 else "rival B"}


def main() -> None:
    fig, axes = F.figure((F.FULL_WIDTH[0], 7.5), ncols=3, nrows=3)
    a, b = made_up(0.012, 0.004), made_up(0.0, 0.004)
    data = (PLOT.column(axes[:, 0], a, False, True, "A: ") + PLOT.column(axes[:, 1], b, False, False, "B: ")
            + PLOT.column(axes[:, 2], a, True, False, "A or B, random: "))
    fig.suptitle("Sketch (made-up numbers): S per side and seed under rival A (left) and rival B (middle); "
                 "random changes (right)", fontsize=9)
    F.save(fig, HERE / "figures" / "sketch2.png", data=data,
           n="made-up numbers; 3 pairs x 2 saves x 2 sides x 3 seeds, larger side also times E187's c; one Gaussian "
             "change per state at its per-matrix norms")


if __name__ == "__main__":
    main()
