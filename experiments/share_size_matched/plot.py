"""Cell P2.34's figures (no model): the re-gating share of each fine-tune against its gain on its own windows, one colour per
built kind, E202's 15 states (experiments/share_built_kind/results/run2/stage3.json) beside a run's three new states.
E204 (run.py, results/stage3.json): Always on scaled up, band Table's mean plus and minus the filed bar (3 against 3).
Second run (run2.py, results/run2/stage3.json): Table scaled down, band Always on's mean plus and minus the filed bar
(6 against 3).

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/share_size_matched/plot.py [MODE]
MODE: result (E204, default), sketch (E204's planned figure), run2, sketch2 (the second run's planned figure, made-up shares).
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
BAR = {"E204": 2 * 0.0405 * math.sqrt(1 / 3 + 1 / 3), "run2": 2 * 0.0405 * math.sqrt(1 / 6 + 1 / 3)}  # the filed bars
# group -> (built kind, colour, marker); colours Okabe-Ito, kind by colour, group by marker
STYLE = {
    "Always on, 300 steps": ("re-weighting", "#0072B2", "o"),
    "Always on, 3000 steps": ("re-weighting", "#0072B2", "v"),
    "Always on, matched size": ("re-weighting", "#0072B2", "D"),
    "Table": ("rewiring", "#D55E00", "s"),
    "Table, matched size": ("rewiring", "#D55E00", "P"),
    "Cue": ("rewiring", "#D55E00", "^"),
    "XOR": ("learned nothing", "#777777", "X"),
}
# run -> (the groups whose mean the band is centred on, band label, band colour)
BAND = {"E204": (("Table",), "Table's mean ± the bar", "#D55E00"),
        "run2": (("Always on, 300 steps", "Always on, 3000 steps"), "Always on's mean ± the bar", "#0072B2")}


def records(new: dict[str, Any]) -> list[dict[str, Any]]:
    states = json.loads(RUN2.read_text())["states"] | new
    return [{"state": k, "group": v["group"], "kind": STYLE[v["group"]][0], "gain": v.get("gain"), "r": v["r"]}
            for k, v in sorted(states.items())]


def panel(ax: Any, recs: list[dict[str, Any]], title: str, run: str) -> None:
    groups, label, colour = BAND[run]
    ref = [x["r"] for x in recs if x["group"] in groups]
    mid = sum(ref) / len(ref)
    ax.axhspan(mid - BAR[run], mid + BAR[run], color=colour, alpha=0.12, linewidth=0, label=label)
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--", label="share 1: all of the gain")
    for group, (kind, color, marker) in STYLE.items():
        pts = [(x["gain"], x["r"]) for x in recs if x["group"] == group and x["r"] is not None and math.isfinite(x["r"])]
        if pts:
            ax.scatter(*zip(*pts, strict=True), color=color, marker=marker, s=26, alpha=0.85, linewidths=0, label=f"{group} ({kind})")
    ax.set_xscale("log")
    ax.set_xlabel("fine-tune's gain on its windows (log-odds, log scale)")
    ax.set_ylabel("re-gating share")
    ax.set_title(title, fontsize=8)


def sketch(run: str) -> None:
    """Made-up shares for the three new states, one panel per rival."""
    group, name, gain, out = {"E204": ("Always on, matched size", "always_on_big_s", 17.0, "sketch.png"),
                              "run2": ("Table, matched size", "table_small_s", 1.7, "sketch_run2.png")}[run]
    rivals = {"E204": ((1.10, "SKETCH, rival A (kind): matched Always on stays near 1.1"),
                       (0.28, "SKETCH, rival B (size): matched Always on falls to Table")),
              "run2": ((0.30, "SKETCH, K (kind): small Table stays low, near full Table"),
                       (1.12, "SKETCH, S (size): small Table rises to Always on"))}[run]
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    recs: list[dict[str, Any]] = []
    for ax, (r, title) in zip(axes, rivals, strict=True):
        new = {f"{name}{s}": {"group": group, "gain": gain + 0.05 * s * gain, "r": r + 0.01 * s} for s in range(3)}
        panel(ax, recs := records(new), title, run)
    axes[1].legend(fontsize=6, loc="center right")
    fig.suptitle("Planned: re-gating share against gain, by built kind (made-up new points)", fontsize=9)
    FG.save(fig, HERE / "figures" / out, n="made-up shares for the 3 new states; E202's 15 states as read", data=recs)


def result(run: str) -> None:
    path, out = {"E204": (HERE / "results" / "stage3.json", "share_vs_gain.png"),
                 "run2": (HERE / "results" / "run2" / "stage3.json", "share_vs_gain_run2.png")}[run]
    recs = records(json.loads(path.read_text())["states"])
    fig, ax = FG.figure(FG.FULL_WIDTH)
    panel(ax, recs, "", run)
    ax.legend(fontsize=6, loc="center right")
    ax.set_title("Re-gating share against gain: does it follow the built kind or the size of the change?", fontsize=9)
    FG.save(fig, HERE / "figures" / out, data=recs,
            n="3 seeds per group; each state read on its own edit's windows where the edit raises its top token by 1 log-odds or more")


def main(mode: str = "result") -> None:
    {"result": lambda: result("E204"), "sketch": lambda: sketch("E204"),
     "run2": lambda: result("run2"), "sketch2": lambda: sketch("run2")}[mode]()


if __name__ == "__main__":
    main(*sys.argv[1:])
