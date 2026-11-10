"""E140's figures, from results/stage2.json and stage3.json only (no model).

shares.png: per pair, one row per method and arm, each seed's share of its own gain in the label contrast on the 256 read rows
(the share, steering per word and pooled, the neuron version at k 64 and k matched); each real row's label gives the method's
real-minus-twin gap and its call under the filed rule. numerator_gain.png (the user's item 5): the same rows with the numerator
itself, beside each fine-tune's gain.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/real_vs_twin/plot.py
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
PAIRS, ARMS, SEEDS = ("case", "repeat"), ("real", "twin"), (0, 1, 2)
STYLE = {"share": "regating", "steer_perword": "steering", "steer_pooled": "steering", "neuron_k64": "neuron", "neuron_kmatched": "neuron", "gain": "size"}
NAMES = {"share": "share", "steer_perword": "steering, per word", "steer_pooled": "steering, pooled", "neuron_k64": "neurons, k 64",
         "neuron_kmatched": "neurons, k matched", "gain": "fine-tune's gain"}  # fmt: skip
N = "3 fine-tune seeds per arm, 256 read rows per pair; label contrast = logit A - logit B, signed toward the window's own label"


def reader(s2: dict[str, Any], s3: dict[str, Any], field: str) -> Callable[[str, str], float]:
    """(method, state) -> `field` ("share" or "numerator") in the label contrast; steering's numerator is its rise."""
    st = s3["states"]

    def get(m: str, n: str) -> float:
        if m == "gain":
            return st[n]["gain"]["contrast"]
        if m == "share":
            return s2["states"][n][field]["contrast"]
        if m.startswith("steer"):
            kind = "perkey" if m == "steer_perword" else "pooled"
            return st[n]["steering"][kind]["share" if field == "share" else "rise"]["contrast"]
        return st[n]["neuron"][m.split("_")[1]][field]["contrast"]

    return get


def records(get: Callable[[str, str], float], methods: tuple[str, ...], score: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for p in PAIRS:
        for m in methods:
            for a in ARMS:
                c = score.get(m, {}).get(p)
                tag = f" (gap {float(c['gap']):+.2f}, {c['call']})" if c and a == "real" else ""
                out += [{"pair": p, "method": m, "arm": a, "seed": s, "row": f"{NAMES[m]}, {a}{tag}", "value": float(get(m, f"{p}_{a}_s{s}"))} for s in SEEDS]
    return out


def draw(recs: list[dict[str, Any]], title: str, xlabel: str, zero: str, out: Path) -> None:
    fig, axes = FG.figure((7.5, 5.0), ncols=2)
    for ax, p in zip(axes, PAIRS, strict=True):
        mine = [r for r in recs if r["pair"] == p]
        rows = list(dict.fromkeys(r["row"] for r in mine))
        for style in dict.fromkeys(STYLE[r["method"]] for r in mine):
            groups: dict[str, list[float]] = {row: [] for row in rows}
            for r in (r for r in mine if STYLE[r["method"]] == style):
                groups[r["row"]].append(r["value"])
            FG.dots(ax, groups, method=style, xlabel=xlabel)
        FG.reference(ax, 0.0, zero)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(f"{p} label", fontsize=9)
    vals = [r["value"] for r in recs if r["value"] == r["value"]]
    pad = 0.05 * (max(vals, default=1.0) - min(vals, default=0.0) or 1.0)
    for ax in axes:  # shared axes: both pairs on one scale, 0 included
        ax.set_xlim(min(0.0, *vals) - pad, max(0.0, *vals) + pad)
    fig.suptitle(title, fontsize=9)
    FG.save(fig, out, n=N, data=recs)


def main(results: Path = HERE / "results", figures: Path = HERE / "figures") -> None:
    s2, s3 = (json.loads((results / f"stage{i}.json").read_text()) for i in (2, 3))
    sc = s3["score"]["contrast"]
    methods = ("share", "steer_perword", "steer_pooled", "neuron_k64", "neuron_kmatched")
    draw(records(reader(s2, s3, "share"), methods, sc), f"Does any reading separate a real fine-tune from its shuffled twin? Filed outcome: {s3['score']['outcome']}",
         "share of the fine-tune's gain in the label contrast", "writes nothing", figures / "shares.png")  # fmt: skip
    draw(records(reader(s2, s3, "numerator"), (*methods, "gain"), {}), "Each reading's numerator beside the fine-tune's gain, in the label contrast",
         "change in the label contrast at the marker", "no change", figures / "numerator_gain.png")  # fmt: skip


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:3]))
