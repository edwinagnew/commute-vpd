"""E189's figure, from results/stage2.json only (no model).

jobs.png, two rows by two readings (retention of the moved neurons' Pile jobs; the gain kept under top-240 per block):
  top     per built pair, the gap = mean over the side built to reuse minus mean over the side built to be new (dot), the band
          the filed call needs it to pass (grey bar from -band to +band, the larger within-side seed range), and the same gap
          for the two sides' random updates of matched size (cross);
  bottom  per pair, each side's mean (filled: reuse side, open: new side) beside its random update's reading (cross) and,
          for the gain kept, the side's mean under a random 240 neurons per block (plus).

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/neuron_jobs/plot.py [RESULTS FIGURES]
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as FG

HERE = Path(__file__).resolve().parent
ROWS = {"e46_copy_random": "copy vs random maps", "e53_lowercase_cipher": "lowercase vs ciphers", "e54_matched": "half-known vs cipher, equal size",
        "e54_own": "half-known vs cipher, own step", "open_pair_c3": "rank-1 vs cipher", "open_pair_c3_copy": "copy vs cipher",
        "e62_copy_lookup": "copy vs lookup", "e58_rank1_96": "rank-1 vs cipher, 96 keys", "e58_copy_prior_cipher": "copy vs prior cipher, 96 keys"}  # fmt: skip
LABELS = {"retention": "Pile-job retention of the 128 most moved neurons", "kept": "share of the gain kept under top-240 per block"}


def mean(v: Sequence[float]) -> float:
    return statistics.fmean(v) if v and not any(math.isnan(x) for x in v) else math.nan


def records(score: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for p in ROWS:
        for r in LABELS:
            s = score[p][r]
            rec = {"pair": p, "reading": r, "gap": s["gap"], "band": s["band"], "call": s["call"],
                   "random_gap": s["floor"]["reuse"]["random"] - s["floor"]["new"]["random"]}  # fmt: skip
            for side in ("reuse", "new"):
                rec[f"{side}_mean"], rec[f"{side}_random_update"] = s["floor"][side]["mean"], s["floor"][side]["random"]
                rec[f"{side}_random_k"] = mean(score[p]["kept_random"]["values"][side]) if r == "kept" else math.nan
            out.append(rec)
    return out


def gaps(ax: Axes, rows: Sequence[dict[str, Any]], reading: str) -> None:
    for i, r in enumerate(rows):
        if not math.isnan(r["band"]):
            ax.hlines(i, -r["band"], r["band"], color="#DDDDDD", linewidth=6, zorder=0)
    FG.dots(ax, {ROWS[r["pair"]]: [r["random_gap"]] for r in rows}, method="random", xlabel=f"{LABELS[reading]},\nreuse side minus new side")
    FG.dots(ax, {ROWS[r["pair"]]: [r["gap"]] for r in rows}, method="neuron", xlabel=f"{LABELS[reading]},\nreuse side minus new side")
    FG.reference(ax, 0.0, "no difference")


def sides(ax: Axes, rows: Sequence[dict[str, Any]], reading: str) -> None:
    color = FG.METHODS["neuron"].color
    for i, r in enumerate(rows):
        for side, dy, face in (("reuse", -0.15, color), ("new", 0.15, "none")):
            y = i + dy
            ax.scatter([r[f"{side}_mean"]], [y], marker="s", s=22, facecolors=face, edgecolors=color, label=f"{side} side mean" if i == 0 else None)
            ax.scatter([r[f"{side}_random_update"]], [y], marker="X", s=22, color="#555555", label="random update" if i == 0 and side == "reuse" else None)
            if reading == "kept":
                ax.scatter([r[f"{side}_random_k"]], [y], marker="+", s=30, color="#999999", label="random 240 neurons" if i == 0 and side == "reuse" else None)
    ax.set_yticks(range(len(rows)), [ROWS[r["pair"]] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(f"{LABELS[reading]},\nside mean")
    ax.legend(fontsize=6, loc="best")


def main(results: Path = HERE / "results", figures: Path = HERE / "figures") -> None:
    recs = records(json.loads((results / "stage2.json").read_text())["score"])
    fig, axes = FG.figure((7.5, 7.0), ncols=2, nrows=2)
    for j, reading in enumerate(LABELS):
        rows = [r for r in recs if r["reading"] == reading]
        gaps(axes[0][j], rows, reading)
        sides(axes[1][j], rows, reading)
    for ax in axes.flat:
        ax.tick_params(axis="both", labelsize=7)
        ax.xaxis.label.set_fontsize(7)
    for ax in axes[:, 1]:
        ax.set_yticklabels([])
    calls = {r: sum(x["call"] != "no call" and x["call"] != "unreadable" for x in recs if x["reading"] == r) for r in LABELS}
    fig.suptitle(f"Goodfire-style neuron readings separate the built sides on {calls['retention']} of {len(ROWS)} pairs (retention) "
                 f"and {calls['kept']} (gain kept under top-240)\n(top: gap, grey seed-spread band, cross random updates; "
                 "bottom: side means against their floors)", fontsize=8)  # fmt: skip
    FG.save(fig, figures / "jobs.png", n="2 to 8 saved fine-tune seeds per side; 2,000 Pile windows of 512 tokens; read rows as the "
            "share's, the read position; 4 blocks of 3072 neurons, 32 per block for retention; one random update per side", data=recs)  # fmt: skip


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:3]))
