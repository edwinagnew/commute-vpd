"""P2.2's scorecard redrawn as gaps: task pair by method, each cell the side built to reuse minus the side built to be new.

Reads `results/scorecard.json` only and writes `figures/scorecard_gaps.png`, `.pdf` and `.csv`. The labels are as each
pair was built; this draws no verdict of reused or new. The do-nothing rows have no two sides and are left out. Shares of
the gain are a difference of seed means. The prior and the update size are log10 of the ratio of seed means, since
their units differ between rows; for the rows that saved ln P that is the difference of means over ln 10. Each cell is
a mean over however many seeds that method has on that side. x means not run.

    ~/commute-agents/fleet/cpu-run experiments/method_table/plot_gaps.py
"""

import json
import math
import statistics
from pathlib import Path
from typing import Any

from commute.figures import figure, gap_heatmap, save

HERE = Path(__file__).parent
LADDER_SIZE_STEP = "10"  # the ladder's only saved reading is update size, by step; the first step is drawn
SHARES = {"r": "re-gating", "r_n": "neuron", "r_half": "neuron,\nhalf", "S": "steering"}
RATIOS = {"prior": "prior", "size": "update\nsize"}
PANELS = {"built_pair_sides": "Pairs built to differ", "shuffled_twin_sides": "Real fine-tune against shuffled-label twin"}
FOOTER = (
    "67M model. Reuse side = the first-named task, or the real fine-tune. Gain = log-odds of the trained answer, fine-tuned minus original model. Each cell: difference or log10 ratio of seed means; "
    "corner number = seeds on the side with fewer.\n"
    "re-gating = share of the gain that appears when only which original subcomponents are switched on changes; neuron = the same reading on neuron firing; half = on the MLP input alone;\n"
    "steering = share of the gain from adding the update's activation shift at the trigger;\n"
    "prior = the original model's probability of the answer; update size = norm of the weight change. Tasks: reports/analyses/2026-11-09_analysis_scorecard.md."
)


def sides(row: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    """The two sides' seed values, or None when the method was not run on this pair."""
    value = row["value"]
    if isinstance(value, str):
        return None
    if "by_step" in value:
        value = value["by_step"][LADDER_SIZE_STEP]
    return value["reuse"], value["new"]


def gap(row: dict[str, Any]) -> float | None:
    two = sides(row)
    if two is None:
        return None
    reuse, new = statistics.fmean(two[0]), statistics.fmean(two[1])
    if row["method"] in SHARES:
        return reuse - new
    if row.get("unit") == "ln P, mean":
        return (reuse - new) / math.log(10)
    return math.log10(reuse / new)


def seeds(row: dict[str, Any]) -> int | None:
    """Seeds on the side with fewer; drawn in the corner of the cell, since a cell can rest on one."""
    two = sides(row)
    return None if two is None else min(len(two[0]), len(two[1]))


def grid(rows: list[dict[str, Any]], criterion: str, methods: dict[str, str]) -> tuple[list[str], list[list[float | None]], list[list[int | None]]]:
    chosen = [r for r in rows if r["criterion"] == criterion]
    names = {r["pair"]: r["pair_name"] for r in chosen}
    cell = {(r["pair"], r["method"]): (gap(r), seeds(r)) for r in chosen}
    labels = [name + (f" (size at step {LADDER_SIZE_STEP})" if pair == "open_pair_ladder" else "") for pair, name in names.items()]
    values = [[cell.get((pair, m), (None, None))[0] for m in methods] for pair in names]
    return labels, values, [[cell.get((pair, m), (None, None))[1] for m in methods] for pair in names]


def panel(ax: Any, labels: list[str], values: list[list[float | None]], counts: list[list[int | None]], methods: dict[str, str], title: str, label: str, vmax: float, digits: int) -> None:
    # Rounded first and + 0.0, so a value that rounds to zero prints without a meaningless minus sign.
    shown = [[None if v is None else round(v, digits) + 0.0 for v in row] for row in values]
    gap_heatmap(ax, shown, labels, list(methods.values()), label=label, vmax=vmax, fmt=f".{digits}f")
    ax.set_title(title, fontsize=8)
    for i, row in enumerate(counts):
        for j, c in enumerate(row):
            if c is not None:
                ax.text(j + 0.46, i - 0.42, str(c), ha="right", va="top", fontsize=5.5, color="#555555")


def largest(values: list[list[list[float | None]]]) -> float:
    return max(abs(v) for g in values for row in g for v in row if v is not None)


def main() -> None:
    rows = json.loads((HERE / "results/scorecard.json").read_text())
    fig, axes = figure((13.0, 7.5), ncols=2, nrows=2)
    grids = {(c, name): grid(rows, c, m) for c in PANELS for name, m in (("shares", SHARES), ("ratios", RATIOS))}
    vmax_s, vmax_r = largest([g[1] for k, g in grids.items() if k[1] == "shares"]), largest([g[1] for k, g in grids.items() if k[1] == "ratios"])
    records = []
    for i, (criterion, heading) in enumerate(PANELS.items()):
        labels, values, counts = grids[(criterion, "shares")]
        panel(axes[i][0], labels, values, counts, SHARES, f"{heading}: shares of the gain", "reuse side minus new side, share of the gain", vmax_s, 2)
        records += [{"panel": criterion, "pair": p, "method": m, "gap": v} for p, row in zip(labels, values, strict=True) for m, v in zip(SHARES, row, strict=True)]
        labels, values, counts = grids[(criterion, "ratios")]
        panel(axes[i][1], labels, values, counts, RATIOS, f"{heading}: prior, update size", "log10 of reuse side over new side", vmax_r, 1)
        records += [{"panel": criterion, "pair": p, "method": m, "gap": v} for p, row in zip(labels, values, strict=True) for m, v in zip(RATIOS, row, strict=True)]
    fig.suptitle("Gap between the two built sides, task pair by method (labels as built, no verdict; × = no saved reading)", fontsize=10)
    save(fig, HERE / "figures/scorecard_gaps.png", n=FOOTER, data=records)


if __name__ == "__main__":
    main()
