"""E278's figure (no GPU), from results/summary.json (run.py --stage summary): panel 1, E62's copy and lookup per seed;
panel 2, E237's key-grid figure (experiments/key_grid/plot.py: the five tasks in order of the base model's prior of
the trained answer, three seeds as dots, the matched random as a grey band) redrawn with E_b over the needed set K.
Panel 2 says "not read yet" while the grid is absent.

    ~/commute-agents/fleet/cpu-run python3 experiments/eb_dropped/plot.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commute.figures import FULL_WIDTH, column_dots, figure, save

HERE = Path(__file__).resolve().parent
RIVAL_GAP = json.loads((HERE / "results" / "predictions.json").read_text())["settings"]["rival_gap"]
ORDER = json.loads((HERE.parent / "key_grid" / "results" / "summary.json").read_text())["x_order_by_prior"]
NAMES = {"copy": "copy", "rank1": "top", "elicit": "mid", "cipher": "cipher", "prior_cipher": "prior-matched\ncipher"}
YLABEL = "E_b over the needed set K"


def main(summary: Path, out: Path) -> None:
    s = json.loads(summary.read_text())
    fig, (a1, a2) = figure(FULL_WIDTH, ncols=2)
    e62 = s["e62"]
    column_dots(a1, {"copy": e62["copy"], "lookup": e62["lookup"]}, method="regating", ylabel=YLABEL)
    gap = e62.get("mean_gap")
    a1.set_title("E62: copy vs lookup" + (f" (mean gap {gap:+.3f})" if gap is not None else ""), fontsize=8)
    data = [{"panel": "e62", "arm": side, "seed": i, "E_b_K": v} for side in ("copy", "lookup") for i, v in enumerate(e62[side])]
    grid = s["grid"]
    if grid:
        bands = {NAMES[t]: (min(r), max(r)) if (r := [v for v in grid[t]["random"] if v is not None]) else None for t in ORDER if t in grid}
        column_dots(a2, {NAMES[t]: grid[t]["scaled"] if t in grid else [] for t in ORDER}, method="regating", ylabel=YLABEL, bands=bands)
        if any(bands.values()):
            a2.legend(fontsize=7, frameon=False)
        a2.set_xlabel("task, lowest prior of the trained answer first")
        data += [{"panel": "grid", "arm": f"{t} {side}", "seed": i, "E_b_K": v} for t in grid for side in ("scaled", "random") for i, v in enumerate(grid[t][side])]
    else:
        a2.text(0.5, 0.5, "key grid not read yet", ha="center", va="center", transform=a2.transAxes, fontsize=8)
    a2.set_title("Key grid at one update size, randoms as grey band", fontsize=8)
    top = max([RIVAL_GAP * 1.25] + [float(r["E_b_K"]) * 1.1 for r in data if r["E_b_K"] is not None])
    for ax in (a1, a2):  # one scale from 0, so a gap reads against the card's 0.2
        ax.set_ylim(0, top)
    n = "3 seeds per arm (dots); E62 at own size; grid: each task scaled to copy's mean size, grey band = its 3 matched Gaussian randoms"
    save(fig, out, n=n, data=data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=HERE / "results" / "summary.json")
    ap.add_argument("--out", type=Path, default=HERE / "figures" / "eb_over_K.png")
    a = ap.parse_args()
    main(a.summary, a.out)
