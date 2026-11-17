"""E155's figure, from results/stage2.json only (no model).

gaps.png, one panel per reading at k 240: per built pair (rows), the gap = mean over the side built to reuse minus mean over
the side built to be new (dot), the band the filed call needs the gap to pass (grey bar from -band to +band: the larger
within-side seed range), and the same gap for the two sides' random updates of matched size (cross). The do-nothing state
reads gap 0 on both readings (the dashed line).

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/neuron_overlap/plot.py [RESULTS FIGURES]
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as FG

HERE = Path(__file__).resolve().parent
ROWS = {"e46_copy_random": "copy vs random maps", "e53_lowercase_cipher": "lowercase vs ciphers", "e54_matched": "half-known vs cipher, equal size",
        "e54_own": "half-known vs cipher, own step", "open_pair_c3": "rank-1 vs cipher", "open_pair_c3_copy": "copy vs cipher",
        "e62_copy_lookup": "copy vs lookup"}  # fmt: skip
LABELS = {"overlap": "top-240 overlap (Jaccard),\nreuse side minus new side", "on_both": "share of the firing change on neurons in\nboth top sets, reuse side minus new side"}


def records(score: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"pair": p, "reading": r, "gap": score[p][r]["gap"], "band": score[p][r]["band"], "call": score[p][r]["call"],
             "random_gap": score[p][r]["floor"]["reuse"]["random"] - score[p][r]["floor"]["new"]["random"]}
            for p in ROWS for r in LABELS]  # fmt: skip


def draw(ax: Axes, recs: Sequence[dict[str, Any]], reading: str) -> None:
    rows = [r for r in recs if r["reading"] == reading]
    for i, r in enumerate(rows):
        ax.hlines(i, -r["band"], r["band"], color="#DDDDDD", linewidth=6, zorder=0)
    FG.dots(ax, {ROWS[r["pair"]]: [r["random_gap"]] for r in rows}, method="random", xlabel=LABELS[reading])
    FG.dots(ax, {ROWS[r["pair"]]: [r["gap"]] for r in rows}, method="neuron", xlabel=LABELS[reading])
    FG.reference(ax, 0.0, "no difference")
    ax.tick_params(axis="both", labelsize=7)
    ax.xaxis.label.set_fontsize(7)


def main(results: Path = HERE / "results", figures: Path = HERE / "figures") -> None:
    recs = records(json.loads((results / "stage2.json").read_text())["score"])
    fig, axes = FG.figure((7.5, 3.6), ncols=2)
    for ax, reading in zip(axes, LABELS):
        draw(ax, recs, reading)
    axes[1].set_yticklabels([])
    n_reuse = sum(r["call"] == "reuse side higher" for r in recs if r["reading"] == "overlap")
    n_new = sum(r["call"] == "new side higher" for r in recs if r["reading"] == "overlap")
    fig.suptitle(f"Top-firing neurons after a fine-tune: the reuse side overlaps more on {n_reuse} of {len(ROWS)} pairs, "
                 f"the new side on {n_new}\n(dot: gap; grey: seed-spread band; cross: random updates of matched size)", fontsize=8)  # fmt: skip
    FG.save(fig, figures / "gaps.png", n="2 to 8 saved fine-tune seeds per side; read rows as the share's (75 to 500 per task), "
            "the read position only; 4 blocks of 3072 neurons; one random update per side", data=recs)  # fmt: skip


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:3]))
