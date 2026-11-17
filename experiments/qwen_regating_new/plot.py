"""E132 (cell P3.1): the re-gating share per saved Qwen update, new decomposition beside old, from results/summary.json.

One figure: each state's share with its 95% row-resample interval, the band between the two anchor 2 seeds drawn as the
seed spread, and the title counting the update pairs the filed rule separates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute.figures import dots, figure, save

HERE = Path(__file__).resolve().parent
LABELS = {
    "e40": "fine-tune, step 10",
    "gold_s0": "gold, step 100",
    "a2long_s0": "anchor 2, seed 0",
    "a2long_s1": "anchor 2, seed 1",
    "gauss": "random, off span",
    "inspan": "random, live columns",
}


def separated(table: dict[str, Any]) -> tuple[int, int]:
    pairs = [p for p in table["pairs"] if p["kind"] == "update-update"]
    return sum(p.get("separated", False) for p in pairs), len(pairs)


def panel(ax: Any, table: dict[str, Any], name: str) -> list[dict[str, Any]]:
    states = table["states"]
    groups = {label: [states[key]["regating"]] for key, label in LABELS.items()}
    dots(ax, groups, method="regating", xlabel="re-gating share")
    for i, key in enumerate(LABELS):
        ax.hlines(i, *states[key]["boot95"], color="#0072B2", linewidth=1)
    seeds = sorted(states[k]["regating"] for k in ("a2long_s0", "a2long_s1"))
    ax.axvspan(*seeds, color="#BBBBBB", alpha=0.5, linewidth=0)
    k, m = separated(table)
    ax.set_title(f"{name}: {k} of {m} update pairs", fontsize=9)
    return [{"decomposition": name, "state": key, "share": states[key]["regating"], "low": states[key]["boot95"][0],
             "high": states[key]["boot95"][1]} for key in LABELS]


def main() -> None:
    summary = json.loads((HERE / "results/summary.json").read_text())
    fig, axes = figure(size=(7.5, 3.2), ncols=2)
    data = panel(axes[0], summary["old"], "old decomposition") + panel(axes[1], summary["new"], "new decomposition")
    axes[1].sharex(axes[0])
    axes[1].set_yticklabels([])
    fig.suptitle("Separated pairs by the filed rule, against the seed gap (grey band, the two anchor 2 seeds)", fontsize=9)
    save(fig, HERE / "figures/share_new_old.png", n="128 maths rows of 256 tokens; bars are 95% row-resample intervals",
         data=data)


if __name__ == "__main__":
    main()
