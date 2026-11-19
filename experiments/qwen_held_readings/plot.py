"""Cell P3.13 run 1: one figure, three panels, from results/summary.json (run.py --summary). Per E167 step-75 state
(RL, the collapsed shuffled-reward run, own-samples SFT; seeds 0 and 70): (left) the held-model reading S beside a
random change at the state's per-matrix norms; (middle) the re-gating share on p-015e0710 minus the random curve at the
state's own size, with the band; (right) effective rank beside the Gaussian floor. No reuse or new label.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_held_readings/plot.py
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as F

HERE = Path(__file__).resolve().parent
ARM_NAME = {"R": "RL", "N": "shuffled reward\n(collapsed)", "F": "own-samples SFT"}
STATE, RANDOM = ("#1f5fa8", "o"), ("#888888", "x")


def rows(items: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted(items, key=lambda k: ("RNF".index(k[0]), k))


def label(name: str) -> str:
    arm, seed = name.split("_s")
    return f"{ARM_NAME[arm]}, seed {seed}"


def pair_panel(ax: Axes, items: Mapping[str, Mapping[str, Any]], key: str, ref: str, xlabel: str) -> list[dict[str, Any]]:
    """Each state's `key` (dot) beside its random reference `ref` (x), one row per state; a missing value is not drawn."""
    data = []
    for i, name in enumerate(rows(items)):
        v, r = items[name].get(key), items[name].get(ref)
        for x, (color, marker) in ((v, STATE), (r, RANDOM)):
            if x is not None:
                ax.scatter([x], [i], color=color, marker=marker, s=26)
        data.append({"state": name, key: v, ref: r})
    ax.set_yticks(range(len(items)), [label(n) for n in rows(items)], fontsize=7)
    ax.set_xlabel(xlabel)
    return data


def share_panel(ax: Axes, items: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Share minus the random curve at the state's own K00 (dot), the band around 0 (grey); 'unreadable' written."""
    data = []
    for i, name in enumerate(rows(items)):
        it = items[name]
        if it.get("gap") is not None:
            ax.scatter([it["gap"]], [i], color=STATE[0], marker=STATE[1], s=26, zorder=3)
        else:
            ax.annotate(it.get("share_class", "not read"), (0, i), fontsize=6.5, ha="center", va="center")
        if it.get("band") is not None:
            ax.fill_betweenx([i - 0.3, i + 0.3], -it["band"], it["band"], color="#dddddd", lw=0)
        data.append({"state": name, "gap": it.get("gap"), "band": it.get("band"), "class": it.get("share_class")})
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_yticks(range(len(items)), [])
    ax.set_xlabel("share minus random curve\n(grey: random band)")
    return data


def draw(axes: Any, items: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    data = pair_panel(axes[0], items, "S", "S_random", "held-model reading S\n(x: random change, same norms)")
    data += share_panel(axes[1], items)
    data += pair_panel(axes[2], items, "erank", "erank_floor", "effective rank of the update\n(x: Gaussian floor)")
    axes[2].set_yticks(range(len(items)), [])
    return data


def main() -> None:
    items = json.loads((HERE / "results" / "summary.json").read_text())["states"]
    fig, axes = F.figure((F.FULL_WIDTH[0], 3.4), ncols=3)
    data = draw(axes, items)
    fig.suptitle("Qwen step-75 updates: S, re-gating share and effective rank, each beside its random reference")
    F.save(fig, HERE / "figures" / "readings.png", data=data,
           n="6 states (3 runs x seeds 0, 70); S fitted on 64 maths rows, read on 64 others; share on 128 maths rows "
             "on p-015e0710 (MLP matrices only), random curve 3 draws x 3 sizes; effective rank over 196 matrices")


if __name__ == "__main__":
    main()
