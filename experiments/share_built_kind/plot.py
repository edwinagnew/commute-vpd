"""E200's and E202's figures, from results/stage2.json and stage3.json, or results/run2/stage3.json (no model): the re-gating share of each
fine-tune, one row per built kind (`figures.dots`), beside the trained 67M pairs' share ranges from the scorecard,
experiments/method_table/results/final.json, jq '.cells[] | select(.method=="share" and .group=="built") | .reuse, .new'.
A state with no share (gain under 1, or a Cue seed stopped in training) is listed under the figure, not drawn.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/share_built_kind/plot.py [run2]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
SCORECARD = HERE.parent / "method_table" / "results" / "final.json"
ROWS = ("Always on, 300 steps", "Always on, 3000 steps", "Table", "Cue", "XOR")
LABEL = {"Always on, 300 steps": "Always on, 300 steps (re-weighting)", "Always on, 3000 steps": "Always on, 3000 steps (re-weighting)",
         "Table": "Table (rewiring)", "Cue": "Cue (rewiring)", "XOR": "XOR (learned nothing; reference)"}
RUNS = {  # run: (results folder, figure name, footer naming n)
    "run1": (HERE / "results", "share_by_kind.png", "3 seeds per row, 2048 windows per state"),
    "run2": (HERE / "results" / "run2", "share_by_kind_run2.png",
             "3 seeds per row; windows where the hand-made edit raises its top token by 1 log-odds or more over the original model"),
}


def trained_ranges() -> dict[str, tuple[float, float]]:
    cells = [c for c in json.loads(SCORECARD.read_text())["cells"] if c["method"] == "share" and c["group"] == "built"]
    return {side: (min(v for c in cells for v in c[side]), max(v for c in cells for v in c[side])) for side in ("reuse", "new")}


def window_counts(results: Path) -> str:
    """Run 2 records each pair's qualifying windows in stage2.json; run 1 read 2048 per pair."""
    pairs = json.loads((results / "stage2.json").read_text()).get("pairs")
    return "" if pairs is None else "; windows per pair: " + ", ".join(f"{p} {v['n_rows']}" for p, v in pairs.items())


def main(run: str = "run1") -> None:
    results, name, n = RUNS[run]
    states: dict[str, Any] = {}
    for stage in ("stage2.json", "stage3.json"):  # run 2's stage 2 holds the windows, its stage 3 all 15 states
        states |= json.loads((results / stage).read_text()).get("states", {})
    groups: dict[str, list[float]] = {LABEL[g]: [] for g in ROWS}
    recs, empty = [], []
    for state, rec in sorted(states.items()):
        r = rec["r"]
        recs.append({"state": state, "group": rec["group"], "r": r, "gain": rec.get("gain"), "stopped": rec.get("stopped")})
        if r is not None and math.isfinite(r):
            groups[LABEL[rec["group"]]].append(r)
        else:
            empty.append(f"{state}: {rec.get('stopped') or 'gain under 1'}")
    fig, ax = FG.figure(FG.FULL_WIDTH if run == "run2" else FG.ONE_COLUMN)
    rng = trained_ranges()
    for side, color, text in (("reuse", "#0072B2", "trained pairs,\nside built to reuse"), ("new", "#D55E00", "trained pairs,\nside built to be new")):
        lo, hi = rng[side]
        ax.axvspan(lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.annotate(text, ((lo + hi) / 2, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=6, color="#555555")
    FG.dots(ax, groups, method="regating", xlabel="re-gating share (log-odds; 0 none of the gain, 1 all of it)")
    if run == "run2":
        FG.reference(ax, 1.0, "")
        ax.annotate("all of\nthe gain", (1.0, 0.5), xycoords=("data", "axes fraction"), xytext=(3, 0), textcoords="offset points",
                    fontsize=6, color="#555555")
    ax.set_title("Re-gating share of fine-tunes whose kind of change is built in", fontsize=9, pad=18)
    n += window_counts(results) + (f"; not drawn: {', '.join(empty)}" if empty else "")
    FG.save(fig, HERE / "figures" / name, n=n, data=recs)


if __name__ == "__main__":
    main(*sys.argv[1:])
