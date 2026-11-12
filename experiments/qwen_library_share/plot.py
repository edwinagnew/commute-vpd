"""P3.8's one figure, from results/summary.json (run.py --summary): per pair and save, side A's re-gating share minus
side B's. Left, the mean over seeds as a heatmap (hatched: inside the random-floor band); right, every seed's gap as a
dot, the random change's gap as an X and the band shaded. No reuse or new label anywhere: a description.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_library_share/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from commute import figures as F

HERE = Path(__file__).resolve().parent
PAIR_NAMES = {"copy_cipher": "copy − cipher", "facts": "known − invented", "gold": "gold − shuffled gold"}
KIND_NAMES = {"late": "late save\n(step 300;\ngold 100)", "matched": "matched size\n(15x floor;\ngold 5x)"}
GAP = "re-gating share, side A minus side B"


def mean(gaps: list[float | None]) -> float | None:
    return None if not gaps or any(g is None for g in gaps) else sum(g for g in gaps if g is not None) / len(gaps)


def heat(ax: Axes, summary: dict[str, Any], vmax: float) -> None:
    pairs, kinds = list(PAIR_NAMES), list(KIND_NAMES)
    cells = [[summary[f"{p}/{k}"] for k in kinds] for p in pairs]
    F.gap_heatmap(ax, [[mean(c["gaps"]) for c in row] for row in cells], list(PAIR_NAMES.values()),
                  list(KIND_NAMES.values()), label="share, A − B, seed mean", vmax=vmax, fmt=".4f",
                  counts=[[sum(g is not None for g in c["gaps"]) for c in row] for row in cells])
    for i, row in enumerate(cells):
        for j, c in enumerate(row):
            if c["cell"] == "inside band":
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", edgecolor="#888888", lw=0))


def strip(ax: Axes, summary: dict[str, Any]) -> None:
    keys = [f"{p}/{k}" for p in PAIR_NAMES for k in KIND_NAMES]
    names = [f"{PAIR_NAMES[summary[k]['pair']]}, {summary[k]['kind']}" for k in keys]
    for i, k in enumerate(keys):
        if summary[k]["band"] is not None:
            b = summary[k]["band"]
            ax.fill_betweenx([i - 0.3, i + 0.3], -b, b, color="#dddddd", zorder=0, lw=0)
    F.dots(ax, {n: [g for g in summary[k]["gaps"] if g is not None] for n, k in zip(names, keys, strict=True)},
           method="regating", xlabel=GAP)
    rnd = F.METHODS["random"]
    for i, k in enumerate(keys):
        if summary[k]["random_gap"] is not None:
            ax.scatter([summary[k]["random_gap"]], [i], color=rnd.color, marker=rnd.marker, s=30)
        if summary[k].get("random_gap_norm") is not None:  # matched norms: expected unscored, so usually absent
            ax.scatter([summary[k]["random_gap_norm"]], [i], color=rnd.color, marker="x", s=18, alpha=0.6)
    ax.axvline(0, color="#999999", lw=0.6)
    reg = F.METHODS["regating"]
    ax.legend(handles=[Line2D([], [], color=reg.color, marker=reg.marker, ls="", label="one seed's gap"),
                       Line2D([], [], color=rnd.color, marker=rnd.marker, ls="", label="random, matched KL"),
                       Line2D([], [], color=rnd.color, marker="x", ls="", alpha=0.6, label="random, matched norms"),
                       Patch(color="#dddddd", label="random-floor band")], fontsize=7, ncols=2,
              loc="upper center", bbox_to_anchor=(0.2, -0.18), frameon=False)


def draw(axes: Any, summary: dict[str, Any], vmax: float) -> list[dict[str, Any]]:
    heat(axes[0], summary, vmax)
    strip(axes[1], summary)
    return [{"key": k, "seed": i, "gap": g, "random_gap": c["random_gap"], "band": c["band"], "cell": c["cell"]}
            for k, c in summary.items() for i, g in enumerate(c["gaps"])]


def vmax_of(*summaries: dict[str, Any]) -> float:
    seen = [abs(g) for s in summaries for c in s.values() for g in [*c["gaps"], c["random_gap"]] if g is not None]
    return max(seen, default=0.01) or 0.01


def main() -> None:
    summary = json.loads((HERE / "results" / "summary.json").read_text())
    fig, axes = F.figure(F.FULL_WIDTH, ncols=2)
    data = draw(axes, summary, vmax_of(summary))
    fig.suptitle("Re-gating share of the pair library's sides, side A minus side B, against random changes of the same size")
    F.save(fig, HERE / "figures" / "share_gaps.png", data=data,
           n="3 seeds per side (dots); one random change per side and save, at seed 0's per-matrix norms (x) and "
             "scaled to its KL (X); 128 rows of 256 tokens per side; band = |matched-KL random gap| + E132's seed "
             "spread 0.002; hatched = mean inside the band")


if __name__ == "__main__":
    main()
