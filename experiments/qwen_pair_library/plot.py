"""Cell P3.5's one figure: per built side, greedy exact match under the cue on the held-out items and on the trained
items, at the matched 5x save and at the last save (step 300), three seeds as dots. From run 3's (E165)
results/library_run3/<side>_s<seed>.json only; no model. The twins (run 2) read other measures and stay in the table.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_pair_library/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
RUN3 = HERE / "results" / "library_run3"
SIDES = {"copy": "copy, reuse", "cipher": "cipher, new", "known": "known fact, reuse", "invented": "invented fact, new"}
SEEDS = (0, 1, 2)
STATES = {"5x": "matched size: first save at 5x the masked error", "last": "last save, step 300 (sizes not matched)"}
SERIES = {"held": ("held-out items", "#0072B2", "o", -0.12), "trained": ("trained items", "#D55E00", "s", 0.12)}


def records() -> list[dict[str, Any]]:
    out = []
    for side in SIDES:
        for seed in SEEDS:
            states = json.loads((RUN3 / f"{side}_s{seed}.json").read_text())["states"]
            for state in STATES:
                s = states[state]
                out.append({"side": side, "seed": seed, "state": state, "step": s["step"], "dw_norm": s["dw_norm"],
                            "held": s["held"]["value"], "trained": s["trained"]["value"]})
    return out


def main() -> None:
    data = records()
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    for ax, (state, title) in zip(axes, STATES.items(), strict=True):
        for key, (label, color, marker, dy) in SERIES.items():
            for i, side in enumerate(SIDES):
                xs = [r[key] for r in data if r["side"] == side and r["state"] == state]
                ax.scatter(xs, [i + dy] * len(xs), color=color, marker=marker, s=22, alpha=0.85, linewidths=0,
                           label=label if i == 0 else None)
        ax.set_yticks(range(len(SIDES)), list(SIDES.values()) if state == "5x" else [""] * len(SIDES))
        ax.set_ylim(len(SIDES) - 0.5, -0.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("greedy exact match under the cue")
        ax.set_title(title, fontsize=8, pad=14)
        FG.reference(ax, 0.9, "learned bar")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Both sides are learned only at the last save; there cipher fails on held-out keys", fontsize=9)
    FG.save(fig, HERE / "figures" / "library.png", data=data,
            n="3 seeds per side; 256 held-out keys (facts: 200 facts in an unseen phrasing), 256 trained items. "
              "The unchanged model reads 0 on every side and both item sets.")


if __name__ == "__main__":
    main()
