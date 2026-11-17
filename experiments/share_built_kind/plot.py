"""E200's one figure, from results/stage1.json and stage2.json (no model): the re-gating share of each fine-tune, one row per
built kind (`figures.dots`), beside the trained 67M pairs' share ranges from the scorecard,
experiments/method_table/results/final.json, jq '.cells[] | select(.method=="share" and .group=="built") | .reuse, .new'.
A state with no share (gain under 1, or a Cue seed stopped in training) is listed under the figure, not drawn.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/share_built_kind/plot.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
SCORECARD = HERE.parent / "method_table" / "results" / "final.json"
ROWS = ("Always on, 300 steps", "Always on, 3000 steps", "Table", "Cue", "XOR")
LABEL = {"Always on, 300 steps": "Always on, 300 steps (re-weighting)", "Always on, 3000 steps": "Always on, 3000 steps (re-weighting)",
         "Table": "Table (rewiring)", "Cue": "Cue (rewiring)", "XOR": "XOR (learned nothing; reference)"}


def trained_ranges() -> dict[str, tuple[float, float]]:
    cells = [c for c in json.loads(SCORECARD.read_text())["cells"] if c["method"] == "share" and c["group"] == "built"]
    return {side: (min(v for c in cells for v in c[side]), max(v for c in cells for v in c[side])) for side in ("reuse", "new")}


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    states: dict[str, Any] = {}
    for name in ("stage1.json", "stage2.json"):
        states |= json.loads((results / name).read_text())["states"]
    groups: dict[str, list[float]] = {LABEL[g]: [] for g in ROWS}
    recs, empty = [], []
    for name, rec in sorted(states.items()):
        r = rec["r"]
        recs.append({"state": name, "group": rec["group"], "r": r, "gain": rec.get("gain"), "stopped": rec.get("stopped")})
        if math.isfinite(r):
            groups[LABEL[rec["group"]]].append(r)
        else:
            empty.append(f"{name}: {rec.get('stopped') or 'gain under 1'}")
    fig, ax = FG.figure(FG.ONE_COLUMN)
    rng = trained_ranges()
    for side, color, text in (("reuse", "#0072B2", "trained pairs,\nside built to reuse"), ("new", "#D55E00", "trained pairs,\nside built to be new")):
        lo, hi = rng[side]
        ax.axvspan(lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.annotate(text, ((lo + hi) / 2, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=6, color="#555555")
    FG.dots(ax, groups, method="regating", xlabel="re-gating share (log-odds; 0 none of the gain, 1 all of it)")
    ax.set_title("Re-gating share of fine-tunes whose kind of change is built in", fontsize=9, pad=18)
    n = "3 seeds per row, 2048 windows per state" + (f"; not drawn: {', '.join(empty)}" if empty else "")
    FG.save(fig, out / "share_by_kind.png", n=n, data=recs)


if __name__ == "__main__":
    main()
