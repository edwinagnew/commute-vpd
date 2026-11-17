"""E199's figure, from results/stage2.json only (no model).

readings.png: one row for the built pair, three panels. Prior: the gap in decades (side built to reuse minus new side),
one point, with the 0.5-decade bar. Steering at strength 1 and the share: every seed of both sides, each minus the new
side's mean (E151's `centred`), with the 0.1 bar of E143's rule on both sides of 0. `draw` also makes the plan's sketch.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/disagreeing_pairs2/plot.py [RESULTS FIGURES]
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from commute import figures as FG
from commute.jobs import load_experiment

P151 = load_experiment("disagreeing_pairs", "plot")  # centred: each side minus the new side's mean
HERE = Path(__file__).resolve().parent
PANELS = {"steering": ("steering", "steering at strength 1, minus new side's mean"), "share": ("regating", "re-gating share, minus new side's mean")}


def draw(rows: Mapping[str, Mapping[str, Any]], path: Path, title: str, note: str) -> None:
    """rows[label] = {"prior_gap": decades, "steering": (reuse seeds, new seeds), "share": (reuse seeds, new seeds)}."""
    fig, axes = FG.figure((11.0, 1.6 + 0.7 * len(rows)), ncols=3)
    FG.dots(axes[0], {r: [x["prior_gap"]] for r, x in rows.items()}, method="prior", xlabel="prior gap, decades (reuse minus new)")
    for x in (-0.5, 0.5):
        FG.reference(axes[0], x, f"{x:+.1f}")
    axes[0].set_title("original model's prior")
    data: list[dict[str, Any]] = [{"row": r, "reading": "prior", "value": x["prior_gap"]} for r, x in rows.items()]
    for ax, key in zip(axes[1:], PANELS, strict=True):
        method, xlabel = PANELS[key]
        sides = {r: P151.centred(x[key]) for r, x in rows.items()}
        FG.dots(ax, {r: s[0] for r, s in sides.items()}, method=method, xlabel=xlabel, offset=-0.12)
        FG.dots(ax, {r: s[1] for r, s in sides.items()}, method="random", xlabel=xlabel, offset=0.12)
        for x in (-0.1, 0.1):
            FG.reference(ax, x, f"{x:+.1f}")
        ax.set_title({"steering": "steering, strength 1", "share": "re-gating share"}[key])
        ax.set_yticks([])
        data += [{"row": r, "reading": key, "side": side, "seed": i, "value_minus_new_mean": v}
                 for r, s in sides.items() for side, vals in zip(("reuse", "new"), s, strict=True) for i, v in enumerate(vals)]  # fmt: skip
    fig.suptitle(title)
    FG.save(fig, path, n=note, data=data)


def main(results: Path = HERE / "results", figures: Path = HERE / "figures") -> None:
    s2 = json.loads((results / "stage2.json").read_text())
    sc = s2["score"]
    label = f"{sc['pair']['reuse side']} (reuse) vs {sc['pair']['new side']} (new)"
    draw({label: s2["inputs"]}, figures / "readings.png", f"Where prior and strength-1 steering disagree, the share reads '{sc['reading']}'",
         "Colour: side built to reuse; grey crosses: new side. 3 seeds a side for steering and the share; the prior is one value a map")  # fmt: skip


if __name__ == "__main__":
    a = sys.argv[1:3] + [str(HERE / "results"), str(HERE / "figures")][len(sys.argv[1:3]) :]
    main(Path(a[0]), Path(a[1]))
