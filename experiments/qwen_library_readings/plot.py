"""E169's scorecard (CPU, after the pods): the cheap readings on the pair library, side A minus side B, one row of panels
per save. Joins results/<pair>/stage{1,2}.json with the library's stored priors, applies rules.py, writes
results/scorecard.json and figures/scorecard.png. No reuse or new label anywhere, and no joined score.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_library_readings/plot.py

Each reading is its own column with its own colour scale (the units differ), shared by both saves. A cell is the mean
over seeds of A minus B, with the seed range under it. A floored reading (steering, effective rank, overlap) prints the
pair's random-floor band under the range, and a cell the rules read "inside floor" is hatched: a gap that the rows and
any size in the set's range give a random change.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from commute import figures as F
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment

R = load_experiment("qwen_library_readings", "rules")
HERE = Path(__file__).resolve().parent
RESULTS, LIBRARY = HERE / "results", EXPERIMENTS / "qwen_pair_library" / "results"


@dataclass(frozen=True)
class Reading:
    key: str
    label: str
    fmt: str
    floor: bool  # read on random changes too, so it has a floor band


READINGS = (
    Reading("size", "update size\n|δW|/|W0|", ".4f", False),
    Reading("prior_rank", "prior: median\nanswer rank, W0", ".0f", False),
    Reading("prior_logp", "prior: mean log P\n(library, stored)", ".1f", False),
    Reading("steer", "steering\n(share of gain)", ".2f", True),
    Reading("erank", "effective rank\nof the update", ".0f", True),
    Reading("overlap", "top-neuron\noverlap", ".3f", True),
)
PAIRS = {
    "copy_cipher": "copy − cipher",
    "facts": "known − invented facts",
    "gold": "gold: real − shuffled",
    "template": "template: real − shuffled\n(fails fit rule)",
}
Gaps = Mapping[str, Mapping[str, Sequence[float | None]]]  # pair -> reading -> A minus B per seed (empty: not read)
Bands = Mapping[str, Mapping[str, float | None]]  # pair -> floored reading -> band


def column_vmax(panels: Sequence[Gaps]) -> dict[str, float]:
    """One colour scale per reading, shared by every panel, so the saves can be compared."""
    out = {}
    for r in READINGS:
        means = [abs(_mean(v)) for g in panels for p in PAIRS if (v := _clean(g.get(p, {}).get(r.key)))]
        out[r.key] = max(means, default=1.0) or 1.0
    return out


def _clean(v: Sequence[float | None] | None) -> list[float]:
    """A cell's seed gaps, or none at all when any seed is missing (drawn x, "not run")."""
    return [] if not v or any(x is None for x in v) else [float(x) for x in v if x is not None]


def _mean(v: Sequence[float]) -> float:
    return sum(v) / len(v)


def scorecard(axes: Sequence[Axes], gaps: Gaps, bands: Bands, vmax: Mapping[str, float], *, save_name: str) -> list[dict[str, Any]]:
    """One save's row of panels: a column per reading, a row per pair. Returns the plotted numbers."""
    data = []
    for j, (ax, r) in enumerate(zip(axes, READINGS, strict=True)):
        cells = [_clean(gaps.get(p, {}).get(r.key)) for p in PAIRS]
        means = [_mean(v) if v else None for v in cells]
        F.gap_heatmap(ax, [[m] for m in means], list(PAIRS.values()), [""], label=f"A − B, {r.label.split(chr(10))[0]}",
                      vmax=vmax[r.key], fmt=r.fmt, colorbar=False)
        ax.set_title(r.label, fontsize=7)
        ax.set_xticks([])
        if j:
            ax.set_yticks([])
        for i, (p, v, m) in enumerate(zip(PAIRS, cells, means, strict=True)):
            band = bands.get(p, {}).get(r.key) if r.floor else None
            if v and m is not None:
                ink = "white" if abs(m) > 0.6 * vmax[r.key] else "#444444"
                under = f"{min(v):{r.fmt}} to {max(v):{r.fmt}}" + ("" if band is None else f"\nfloor ±{band:{r.fmt}}")
                ax.text(0, i + 0.33, under, ha="center", va="center", fontsize=5, color=ink)
            if r.floor and v and R.floored(v, band) == "inside floor":
                ax.add_patch(Rectangle((-0.5, i - 0.5), 1, 1, fill=False, hatch="//", edgecolor="#aaaaaa", linewidth=0))
            data.append({"save": save_name, "pair": p, "reading": r.key, "mean_gap": m,
                         "seeds": ";".join(format(x, "g") for x in v), "floor_band": band})
    axes[0].set_ylabel(save_name)
    return data


# ----------------------------------------------------------------------------- the join


def library_prior(pair: str) -> dict[str, list[float]]:
    """The library's stored mean log P prior per side and seed (w0.prior), no recompute."""
    p = R.PAIRS[pair]
    return {side: [json.loads((LIBRARY / p.run / f"{side}_s{s}.json").read_text())["w0"]["prior"] for s in R.SEEDS]
            for side in (p.a, p.b)}


def pair_gaps(pair: str, states: Mapping[str, Mapping[str, Any]], save: str) -> dict[str, list[float | None]]:
    p, prior = R.PAIRS[pair], library_prior(pair)
    out: dict[str, list[float | None]] = {}
    for r in READINGS:
        gaps: list[float | None] = []
        for i, s in enumerate(R.SEEDS):
            if r.key == "prior_logp":
                a, b = prior[p.a][i], prior[p.b][i]
            else:
                a, b = (states.get(f"{side}_s{s}_{save}", {}).get(r.key) for side in (p.a, p.b))
            gaps.append(None if a is None or b is None else a - b)
        out[r.key] = gaps
    return out


def join() -> dict[str, Any]:
    out: dict[str, Any] = {"pairs": {}, "missing": []}
    for pair, p in R.PAIRS.items():
        s1, s2 = (RESULTS / pair / f"stage{n}.json" for n in (1, 2))
        if not (s1.is_file() and s2.is_file()):
            out["missing"].append(pair)
            continue
        band = json.loads(s1.read_text())["band"]
        states = json.loads(s2.read_text())["states"]
        rec: dict[str, Any] = {"band": band, "saves": {}}
        for save in p.saves:
            gaps = pair_gaps(pair, states, save)
            cells = {r: R.floored(gaps[r], band[r]) for r in R.FLOORED} | {r: R.signed(gaps[r]) for r in R.SIGNED}
            steps = sorted({st["step"] for st in states.values() if st["save"] == save})
            rec["saves"][save] = {"gaps": gaps, "cells": cells, "steps": steps}
        out["pairs"][pair] = rec
    late = {p: v["saves"]["last"]["cells"] for p, v in out["pairs"].items()}
    out["card_call"] = R.card_call(late)
    return out


def main() -> None:
    S = join()
    (RESULTS / "scorecard.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({"missing": S["missing"], "card_call": S["card_call"],
                      "cells": {p: {s: v["cells"] for s, v in r["saves"].items()} for p, r in S["pairs"].items()}}), flush=True)
    panels = {save: {p: r["saves"][save]["gaps"] for p, r in S["pairs"].items() if save in r["saves"]} for save in ("last", "5x")}
    bands = {p: r["band"] for p, r in S["pairs"].items()}
    steps = {save: sorted({t for r in S["pairs"].values() if save in r["saves"] for t in r["saves"][save]["steps"]}) for save in panels}
    fig, axes = F.figure((10.0, 5.5), ncols=len(READINGS), nrows=2)
    vmax = column_vmax(list(panels.values()))
    span = {s: f"steps {t[0]}–{t[-1]}" if len(t) > 1 else f"step {t[0]}" if t else "not read" for s, t in steps.items()}
    data = scorecard(axes[0], panels["last"], bands, vmax, save_name=f"late save (steps {', '.join(map(str, steps['last']))})")
    data += scorecard(axes[1], panels["5x"], bands, vmax, save_name=f"matched 5x save ({span['5x']})")
    fig.suptitle("Cheap readings of the pair library, side A minus side B: built pairs and real-minus-shuffled twins")
    F.save(fig, HERE / "figures" / "scorecard.png", data=data,
           n="3 seeds per side, paired by seed; cell = mean A − B, seed range and the pair's random-floor band under it; "
             "colour scaled per reading (red: A higher); hatched = inside the floor band. Template twin: late save only.")


if __name__ == "__main__":
    main()
    hard_exit(0)
