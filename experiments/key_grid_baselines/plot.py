"""P2.52's figure (no GPU): E237's five-task graph with the baselines as extra panels. Tasks on the x-axis in E237's order
(lowest prior of the trained answer first), the three seeds as dots at E237's common update size, the matched random
change as a grey band (min to max of the task's three draws) where the reading is defined on it. First panel: E237's R at
VPD subcomponents (the definition reading), from E237's summary.json. Reads whatever item files exist.

    ~/commute-agents/fleet/cpu-run python3 experiments/key_grid_baselines/plot.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from commute.figures import column_dots, figure, save

HERE = Path(__file__).resolve().parent
RUN, E237 = HERE / "results" / "run", HERE.parents[1] / "experiments" / "key_grid" / "results" / "summary.json"
PANELS = {  # name: (y label, figures.METHODS style, how to read one item record)
    "top_k64": ("top-neuron share, k 64", "neuron", lambda r: r["top"]["k64"]["share"]),
    "top_kmatched": ("top-neuron share, k matched", "neuron_half", lambda r: r["top"]["kmatched"]["share"]),
    "r_n": ("neuron share, whole width", "neuron", lambda r: r["whole"]["r_n"]),
    "crosscoder_d": ("crosscoder: shared minus specific part", "regating", lambda r: r["d"]),
}
SEEDS = (0, 1, 2)


def num(x: Any) -> float | None:
    return float(x) if isinstance(x, int | float) and math.isfinite(x) else None


def records() -> tuple[dict[str, Any], dict[str, Any]]:
    """Neuron item records (every shard file) and crosscoder side records, keyed by item."""
    neu = {k: v for p in sorted(RUN.glob("neuron_*.json")) for k, v in json.loads(p.read_text())["items"].items()}
    xc = {(r := json.loads(p.read_text()))["item"]: r for p in sorted((RUN / "crosscoder" / "sides").glob("*.json"))}
    return neu, xc


def column(recs: dict[str, Any], get: Any, task: str, kind: str) -> list[float | None]:
    names = [f"e58/{task}_s{s}_scaled" if kind == "scaled" else f"random_e58/{task}_s{s}" for s in SEEDS]
    return [num(get(recs[n])) if n in recs else None for n in names]


def band(values: list[float | None]) -> tuple[float, float] | None:
    v = [x for x in values if x is not None]
    return (min(v), max(v)) if v else None


def main() -> None:
    summary = json.loads(E237.read_text())
    order, vpd = summary["x_order_by_prior"], summary["table"]["R_vpd"]
    neu, xc = records()
    fig, axes = figure((13.0, 3.0), ncols=1 + len(PANELS))
    column_dots(axes[0], {t: vpd[t]["seeds"] for t in order}, method="neuron_half", ylabel="R at VPD subcomponents (E237)",
                bands={t: vpd[t]["band"] for t in order})  # fmt: skip
    data = [{"panel": "R_vpd", "task": t, "seed": i, "value": v} for t in order for i, v in enumerate(vpd[t]["seeds"])]
    for ax, (name, (ylabel, method, get)) in zip(axes[1:], PANELS.items(), strict=True):
        recs = xc if name == "crosscoder_d" else neu
        seeds = {t: column(recs, get, t, "scaled") for t in order}
        rand = {t: column(recs, get, t, "random") for t in order}
        column_dots(ax, seeds, method=method, ylabel=ylabel, bands={t: band(rand[t]) for t in order})
        data += [{"panel": name, "task": t, "seed": i, "value": v, "random": rand[t][i]} for t in order for i, v in enumerate(seeds[t])]
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Baselines beside the definition reading on E237's five tasks at one update size (tasks: lowest prior first)", fontsize=8)
    n = "3 seeds per task (dots); grey band: E237's 3 random changes per task, same size (absent where the reading is undefined: gain under 1 log-odds)"
    save(fig, HERE / "figures" / "five_tasks_baselines.png", n=n, data=data)


if __name__ == "__main__":
    main()
