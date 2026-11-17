"""E151's figure, from results/stage2.json only (no model).

readings.png: one row per pair (A, B, D, then C, whose share was seen before filing), three panels. Prior: the gap in
decades (side built to reuse minus new side), one point, with the 0.5-decade bar. Steering at strength 1 and the share:
every seed of both sides, each minus the new side's mean, so the new side sits about 0 and the side built to reuse at the
gap; the 0.1 bar of E143's rule is drawn on both sides of 0.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/disagreeing_pairs/plot.py [RESULTS FIGURES [STEM]]
(STEM amend1_stage reads Amendment 1's amend1_stage2.json; default stage, run 1's.)
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as FG

HERE = Path(__file__).resolve().parent
ROWS = {"A_copy_common": "A copy vs common-word cipher", "B_elicit_common": "B half-known vs common-word cipher",
        "D_rank1_oneword": "D rank-1 vs one-word map", "C_rank1_classshuffled": "C rank-1 vs class shuffled (seen)"}  # fmt: skip
PANELS = {"steering": ("steering", "steering at strength 1, minus new side's mean"), "share": ("regating", "re-gating share, minus new side's mean")}


def centred(sides: Sequence[Sequence[Any]]) -> tuple[list[float], list[float]]:
    reuse, new = ([float(v) for v in s] for s in sides)  # float(): a saved nan is the string "nan"
    finite = [v for v in new if not math.isnan(v)]
    m = statistics.fmean(finite) if finite else math.nan
    return [v - m for v in reuse], [v - m for v in new]


def seeds_panel(ax: Axes, inputs: Mapping[str, Any], score: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    method, xlabel = PANELS[key]
    sides = {p: centred(inputs[p][key]) for p in ROWS}
    FG.dots(ax, {ROWS[p]: s[0] for p, s in sides.items()}, method=method, xlabel=xlabel, offset=-0.12)
    FG.dots(ax, {ROWS[p]: s[1] for p, s in sides.items()}, method="random", xlabel=xlabel, offset=0.12)
    for x in (-0.1, 0.1):
        FG.reference(ax, x, f"{x:+.1f}")
    ax.set_title({"steering": "steering, strength 1", "share": "re-gating share"}[key])
    return [{"pair": p, "reading": key, "side": side, "seed": i, "value_minus_new_mean": v, "call": score[p][key]["call"]}
            for p, s in sides.items() for side, vals in zip(("reuse", "new"), s, strict=True) for i, v in enumerate(vals)]  # fmt: skip


def prior_panel(ax: Axes, inputs: Mapping[str, Any], score: Mapping[str, Any]) -> list[dict[str, Any]]:
    FG.dots(ax, {ROWS[p]: [inputs[p]["prior_gap"]] for p in ROWS}, method="prior", xlabel="prior gap, decades (reuse minus new)")
    for x in (-0.5, 0.5):
        FG.reference(ax, x, f"{x:+.1f}")
    ax.set_title("original model's prior")
    return [{"pair": p, "reading": "prior", "gap_decades": inputs[p]["prior_gap"], "call": score[p]["prior"]["call"],
             "condition": score[p]["condition"], "share_reading": score[p]["reading"]} for p in ROWS]  # fmt: skip


def main(results: Path = HERE / "results", figures: Path = HERE / "figures", stem: str = "stage") -> None:
    s2 = json.loads((results / f"{stem}2.json").read_text())
    inputs, score = s2["inputs"], s2["score"]["pairs"]
    fig, axes = FG.figure((11.0, 4.0), ncols=3)
    data = prior_panel(axes[0], inputs, score)
    for ax, key in zip(axes[1:], PANELS, strict=True):
        data += seeds_panel(ax, inputs, score, key)
        ax.set_yticks([])
    fig.suptitle(f"Where prior and strength-1 steering disagree, the share's direction: outcome '{s2['score']['outcome']}'")
    n = ("Colour: side built to reuse; grey crosses: new side. 3 seeds a side for steering and the share; the prior is one value"
         " a map; pair C is never counted")  # fmt: skip
    FG.save(fig, figures / "readings.png", n=n, data=data)


if __name__ == "__main__":
    a = sys.argv[1:4] + [str(HERE / "results"), str(HERE / "figures"), "stage"][len(sys.argv[1:4]) :]
    main(Path(a[0]), Path(a[1]), a[2])
