"""Cell P2.34's figure (no model): the re-gating share of each fine-tune against its gain on its own windows, one colour per
built kind, E202's 15 states (experiments/share_built_kind/results/run2/stage3.json) beside this run's three size-matched
Always on states (results/stage3.json). The band is Table's mean share plus and minus the filed bar (results/rules.json).

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/share_size_matched/plot.py [sketch]
`sketch` draws the planned figure with made-up shares for the new states, one panel per rival (figures/sketch.png).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
RUN2 = HERE.parent / "share_built_kind" / "results" / "run2" / "stage3.json"
OURS = HERE / "results" / "stage3.json"
BAR = 2 * 0.0405 * math.sqrt(1 / 3 + 1 / 3)  # the filed bar at 3 against 3 (results/predictions.json)
# group -> (built kind, colour, marker); colours Okabe-Ito, kind by colour, group by marker
STYLE = {
    "Always on, 300 steps": ("re-weighting", "#0072B2", "o"),
    "Always on, 3000 steps": ("re-weighting", "#0072B2", "v"),
    "Always on, matched size": ("re-weighting", "#0072B2", "D"),
    "Table": ("rewiring", "#D55E00", "s"),
    "Cue": ("rewiring", "#D55E00", "^"),
    "XOR": ("learned nothing", "#777777", "X"),
}


def records(new: dict[str, Any]) -> list[dict[str, Any]]:
    states = json.loads(RUN2.read_text())["states"] | new
    return [{"state": k, "group": v["group"], "kind": STYLE[v["group"]][0], "gain": v.get("gain"), "r": v["r"]}
            for k, v in sorted(states.items())]


def panel(ax: Any, recs: list[dict[str, Any]], title: str) -> None:
    table = [x["r"] for x in recs if x["group"] == "Table"]
    mid = sum(table) / len(table)
    ax.axhspan(mid - BAR, mid + BAR, color="#D55E00", alpha=0.12, linewidth=0, label="Table's mean ± the bar")
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--", label="share 1: all of the gain")
    for group, (kind, color, marker) in STYLE.items():
        pts = [(x["gain"], x["r"]) for x in recs if x["group"] == group and x["r"] is not None and math.isfinite(x["r"])]
        if pts:
            ax.scatter(*zip(*pts, strict=True), color=color, marker=marker, s=26, alpha=0.85, linewidths=0, label=f"{group} ({kind})")
    ax.set_xscale("log")
    ax.set_xlabel("fine-tune's gain on its windows (log-odds, log scale)")
    ax.set_ylabel("re-gating share")
    ax.set_title(title, fontsize=8)


def main(mode: str = "result") -> None:
    if mode == "sketch":  # made-up shares for the three new states; their gains set near Table's
        fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
        for ax, r, title in ((axes[0], 1.10, "SKETCH, rival A (kind): matched Always on stays near 1.1"),
                             (axes[1], 0.28, "SKETCH, rival B (size): matched Always on falls to Table")):
            new = {f"always_on_big_s{s}": {"group": "Always on, matched size", "gain": 17.0 + s, "r": r + 0.01 * s} for s in range(3)}
            panel(ax, recs := records(new), title)
        axes[1].legend(fontsize=6, loc="center right")
        fig.suptitle("Planned: re-gating share against gain, by built kind (made-up new points)", fontsize=9)
        FG.save(fig, HERE / "figures" / "sketch.png", n="made-up shares for the 3 new states; E202's 15 states as read", data=recs)
        return
    new = json.loads(OURS.read_text())["states"]
    recs = records(new)
    fig, ax = FG.figure(FG.FULL_WIDTH)
    panel(ax, recs, "")
    ax.legend(fontsize=6, loc="center right")
    ax.set_title("Re-gating share against gain: does it follow the built kind or the size of the change?", fontsize=9)
    FG.save(fig, HERE / "figures" / "share_vs_gain.png", data=recs,
            n="3 seeds per group; each state read on its own edit's windows where the edit raises its top token by 1 log-odds or more")


if __name__ == "__main__":
    main(*sys.argv[1:])
