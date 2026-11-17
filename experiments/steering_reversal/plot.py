"""E146's figure, from results/stage1.json only (no model).

reversal.png: per pair (rows, each in the steering kind the method table reads it in) and reading (columns, one thing changed
from E143's filed reading), the gap between the side built to reuse and the side built to be new, each value a share of the
fine-tune's own gain in log-odds of its target on the share's read rows. Below zero: the new side reads higher.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/steering_reversal/plot.py [RESULTS FIGURES]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).resolve().parent
ROWS = {"e54_matched": "half-known vs cipher,\nequal size (per key)", "open_pair_c3": "rank-1 vs cipher,\nstep 10 (pooled)",
        "e54_own": "half-known vs cipher, own\nstep (per key; not scored)"}  # fmt: skip
COLS = {"filed": "as filed\n(E143)", "strength1": "strength 1", "layer2": "layer 2,\nboth sides", "layer3": "layer 3,\nboth sides",
        "random_subtracted": "random\nsubtracted", "pair_gain": "pair's\nmean gain"}  # fmt: skip


def main(results: Path = HERE / "results", figures: Path = HERE / "figures") -> None:
    pairs = json.loads((results / "stage1.json").read_text())["score"]["pairs"]
    values = [[pairs[p][r]["gap"] for r in COLS] for p in ROWS]
    counts = [[len(pairs[p][r]["reuse"]) + len(pairs[p][r]["new"]) for r in COLS] for p in ROWS]
    fig, ax = FG.figure(FG.FULL_WIDTH)
    FG.gap_heatmap(ax, values, list(ROWS.values()), list(COLS.values()), label="gap, reuse side minus new side\n(share of each fine-tune's gain)",
                   vmax=0.4, counts=counts)  # fmt: skip
    ax.set_title("Steering's reversal after changing one thing at a time")
    data = [{"pair": p, "reading": r, "gap": pairs[p][r]["gap"], "spread": pairs[p][r]["spread"], "call": pairs[p][r]["call"],
             "reversal": pairs[p][r]["reversal"]} for p in ROWS for r in COLS]  # fmt: skip
    FG.save(fig, figures / "reversal.png", n="seeds per side: 5/5, 8/8, 5/2 (n = both sides); read rows as the share's", data=data)


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:3]))
