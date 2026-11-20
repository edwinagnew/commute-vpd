"""Cell P3.13 run 2: one figure from results/run2/summary.json (summary2.py). One row of panels per pair; in each panel,
per save, three lines: the reuse side and the new side at saved size, and the larger side times E187's c (size-matched).
Left: S of each state per seed; right: S of the Gaussian change at the same per-matrix norms (the random reference), on
the same lines. The rule's gap is the reuse line minus the new line (saved), or with the larger side's c line in place
of its saved line (matched).

    ~/commute-agents/fleet/cpu-run python experiments/qwen_held_readings/plot2.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from commute import figures as F

HERE = Path(__file__).resolve().parent
REUSE, NEW = "#0072B2", "#D55E00"
TITLE = {"copy_cipher": "copy / cipher", "facts": "known / invented", "gold": "gold / shuffled gold"}


def lines(pair: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """(label, save, arm, which) per line, top to bottom: which is 'saved' or 'c'."""
    out = []
    for save, c in pair["saves"].items():
        out += [(f"{pair['reuse_side']}, {save}", save, pair["reuse_side"], "saved"),
                (f"{pair['new_side']}, {save}", save, pair["new_side"], "saved"),
                (f"{c['larger']} x {c['c']:.2f}, {save}", save, c["larger"], "c")]
    return out


def panel(ax: Axes, name: str, pair: dict[str, Any], random: bool, label_lines: bool) -> list[dict[str, Any]]:
    """Each line's S per seed (filled: saved size; open diamond: times c); a 'no change' S is not drawn."""
    data = []
    rows = lines(pair)
    for i, (_lab, save, arm, which) in enumerate(rows):
        key = ("c_random" if which == "c" else "random") if random else which
        vals = pair["saves"][save]["S"][arm][key]
        color = REUSE if arm == pair["reuse_side"] else NEW
        xs = [v for v in vals if v is not None]
        ax.scatter(xs, [i] * len(xs), s=22, marker="o" if which == "saved" else "D", linewidths=1,
                   facecolors=color if which == "saved" else "none", edgecolors=color)
        data += [{"pair": name, "save": save, "arm": arm, "read": key, "seed_index": j, "S": v} for j, v in enumerate(vals)]
    ax.set_yticks(range(len(rows)), [r[0] for r in rows] if label_lines else [], fontsize=7)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("S of the Gaussian change, same norms" if random else "held-model reading S", fontsize=7)
    return data


def column(axes: Any, summary: dict[str, Any], random: bool, label_lines: bool, head: str) -> list[dict[str, Any]]:
    """One panel per pair, top to bottom; `head` starts each panel's title."""
    data = []
    for ax, (name, pair) in zip(axes, summary["pairs"].items(), strict=True):
        ax.set_title(f"{head}{TITLE.get(name, name)}" + ("" if random else f": {pair['outcome']}"), fontsize=7.5)
        data += panel(ax, name, pair, random, label_lines)
    return data


def main() -> None:
    summary = json.loads((HERE / "results" / "run2" / "summary.json").read_text())
    fig, axes = F.figure((F.FULL_WIDTH[0], 7.5), ncols=2, nrows=3)
    data = column(axes[:, 0], summary, False, True, "") + column(axes[:, 1], summary, True, False, "random: ")
    fig.suptitle(f"Qwen pair library, S per side and seed beside its random change: {summary['outcome']}", fontsize=9)
    F.save(fig, HERE / "figures" / "run2_S.png", data=data,
           n="3 seeds per side (dots); blue the side built to reuse, orange the side built to be new; open diamonds the "
             "larger side times E187's c; S fitted on 64 of each arm's own 128 rows, read on the other 64")


if __name__ == "__main__":
    main()
