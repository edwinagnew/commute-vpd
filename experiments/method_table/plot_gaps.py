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


def grid(rows: list[dict[str, Any]], criterion: str, methods: dict[str, str]) -> tuple[list[str], list[list[float | None]]]:
    chosen = [r for r in rows if r["criterion"] == criterion]
    names = {r["pair"]: r["pair_name"] for r in chosen}
    cell = {(r["pair"], r["method"]): gap(r) for r in chosen}
    labels = [name + (f" (size at step {LADDER_SIZE_STEP})" if pair == "open_pair_ladder" else "") for pair, name in names.items()]
    return labels, [[cell.get((pair, m)) for m in methods] for pair in names]


def panel(ax: Any, labels: list[str], values: list[list[float | None]], methods: dict[str, str], title: str, label: str, vmax: float, fmt: str) -> None:
    gap_heatmap(ax, values, labels, list(methods.values()), label=label, vmax=vmax, fmt=fmt)
    ax.set_title(title, fontsize=8)


def largest(values: list[list[list[float | None]]]) -> float:
    return max(abs(v) for g in values for row in g for v in row if v is not None)


def main() -> None:
    rows = json.loads((HERE / "results/scorecard.json").read_text())
    fig, axes = figure((13.0, 7.5), ncols=2, nrows=2)
    grids = {(c, name): grid(rows, c, m) for c in PANELS for name, m in (("shares", SHARES), ("ratios", RATIOS))}
    vmax_s, vmax_r = largest([g[1] for k, g in grids.items() if k[1] == "shares"]), largest([g[1] for k, g in grids.items() if k[1] == "ratios"])
    records = []
    for i, (criterion, heading) in enumerate(PANELS.items()):
        labels, values = grids[(criterion, "shares")]
        panel(axes[i][0], labels, values, SHARES, f"{heading}: shares of the gain", "reuse side minus new side, share of the gain", vmax_s, ".2f")
        records += [{"panel": criterion, "pair": p, "method": m, "gap": v} for p, row in zip(labels, values, strict=True) for m, v in zip(SHARES, row, strict=True)]
        labels, values = grids[(criterion, "ratios")]
        panel(axes[i][1], labels, values, RATIOS, f"{heading}: prior, update size", "log10 of reuse side over new side", vmax_r, ".1f")
        records += [{"panel": criterion, "pair": p, "method": m, "gap": v} for p, row in zip(labels, values, strict=True) for m, v in zip(RATIOS, row, strict=True)]
    fig.suptitle("Gap between the two built sides, task pair by method (labels as built, no verdict; × = not run)", fontsize=10)
    save(fig, HERE / "figures/scorecard_gaps.png", n="each cell: difference or log ratio of seed means, 1 to 8 seeds per side. neuron = the neuron version of the share; half = its MLP-input form; prior = the original model's probability of the trained answer", data=records)


if __name__ == "__main__":
    main()
