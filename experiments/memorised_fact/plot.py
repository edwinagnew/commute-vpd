"""E168's figure (cell P1.16), from results/readings.json only (no model): per item its best re-weighting share G and
best rewiring share H, the twin and fact fine-tunes as one dot per seed, the planted edits and the floor as references
on the same axes, against the adopted cut-offs.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/memorised_fact/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).resolve().parent
CUT_G, CUT_H = 0.8, 0.5  # decisions.md, 2026-11-10; predictions.json "cutoffs" must agree
ROWS = {  # row label -> items in readings.json
    "twin: last number odd,\nfine-tuned": ("twin_s0", "twin_s1"),
    "memorised fact,\nfine-tuned": ("fact_s0", "fact_s1"),
    "planted odd edit\n(exact, control)": ("planted_odd",),
    "planted fact edit": ("planted_fact",),
    "planted floor": ("floor",),
}
FINE_TUNES = ("twin_s0", "twin_s1", "fact_s0", "fact_s1")


def panel(ax, values: dict[str, float], xlabel: str) -> None:
    """Fine-tune seeds in colour, planted references in grey, on the same rows."""
    tuned = {r: [values[i] for i in items if i in FINE_TUNES] for r, items in ROWS.items()}
    planted = {r: [values[i] for i in items if i not in FINE_TUNES] for r, items in ROWS.items()}
    FG.dots(ax, tuned, method="regating", xlabel=xlabel)
    FG.dots(ax, planted, method="random", xlabel=xlabel)
    ax.axvline(0, color="#bbbbbb", linewidth=0.6)
    ax.set_xlim(-0.3, 1.05)


def main() -> None:
    r = json.loads((HERE / "results" / "readings.json").read_text())
    fig, (axg, axh) = FG.figure(FG.FULL_WIDTH, ncols=2)
    panel(axg, r["G"], "re-weighting share G, best over sites and starts")
    FG.reference(axg, CUT_G, f"adopted cut-off {CUT_G}")
    axg.set_title("Re-weighting share", loc="left")
    panel(axh, r["H"], "rewiring share H, best over sites")
    FG.reference(axh, CUT_H, f"cut-off {CUT_H}")
    axh.tick_params(labelleft=False)
    axh.set_title("Rewiring share", loc="left")
    fig.suptitle("The memorised fact reads well under its twin on the re-weighting share;\n"
                 "the planted pair barely separates, and the exact planted edit reaches under two thirds")
    data = [{"item": i, "G": r["G"][i], "H": r["H"][i], "H_from": r["H_saved_from"].get(i, "E168")} for i in r["G"]]
    FG.save(fig, HERE / "figures" / "shares.png",
            n="fine-tunes: 2 seeds each (dots); planted edits and floor: one edit each (grey). "
              "Twin seed 0 and every H of the planted odd edit, floor and twins read in earlier runs on the same states",
            data=data)


if __name__ == "__main__":
    main()
