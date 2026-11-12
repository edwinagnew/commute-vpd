"""The scorecard: the cheap readings on the pair library, side A minus side B, one row of panels per save.

Each reading is its own column with its own colour scale (the units differ), shared by both saves. A cell is the mean
over seeds of A minus B, with the seed range under it. An update-dependent reading (steering, effective rank, overlap)
carries its random-floor band in its column title, and a cell whose mean lies inside the band is hatched: a gap that
size alone, read on random changes, could give. No reuse or new label anywhere, and no joined score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from commute import figures as F


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
    Reading("steer", "steering", ".2f", True),
    Reading("erank", "effective rank\nof the update", ".0f", True),
    Reading("overlap", "top-neuron\noverlap", ".3f", True),
)
PAIRS = {
    "copy_cipher": "copy − cipher",
    "facts": "known − invented facts",
    "gold": "gold: real − shuffled",
    "template": "template: real − shuffled\n(fails fit rule)",
}
Gaps = Mapping[str, Mapping[str, Sequence[float]]]  # pair -> reading -> A minus B per seed (empty: not read)


def column_vmax(panels: Sequence[Gaps]) -> dict[str, float]:
    """One colour scale per reading, shared by every panel, so the saves can be compared."""
    out = {}
    for r in READINGS:
        means = [abs(sum(v) / len(v)) for g in panels for p in PAIRS if (v := g.get(p, {}).get(r.key))]
        out[r.key] = max(means, default=1.0) or 1.0
    return out


def scorecard(axes: Sequence[Axes], gaps: Gaps, bands: Mapping[str, float], vmax: Mapping[str, float], *,
              save_name: str) -> list[dict[str, Any]]:
    """One save's row of panels: a column per reading, a row per pair. Returns the plotted numbers.

    No colour bar: every cell prints its number, and the colour only scales within a reading's column."""
    data = []
    for j, (ax, r) in enumerate(zip(axes, READINGS, strict=True)):
        cells = [gaps.get(p, {}).get(r.key) or [] for p in PAIRS]
        means = [sum(v) / len(v) if v else None for v in cells]
        F.gap_heatmap(ax, [[m] for m in means], list(PAIRS.values()), [""], label=f"A − B, {r.label.split(chr(10))[0]}",
                      vmax=vmax[r.key], fmt=r.fmt, colorbar=False)
        band = bands.get(r.key) if r.floor else None
        ax.set_title(r.label + ("" if band is None else f"\nfloor band ±{band:{r.fmt}}"), fontsize=7)
        ax.set_xticks([])
        if j:
            ax.set_yticks([])
        for i, (v, m) in enumerate(zip(cells, means, strict=True)):
            if v and m is not None:
                ink = "white" if abs(m) > 0.6 * vmax[r.key] else "#444444"
                ax.text(0, i + 0.3, f"{min(v):{r.fmt}} to {max(v):{r.fmt}}", ha="center", va="center", fontsize=5, color=ink)
            if band is not None and m is not None and abs(m) <= band:
                ax.add_patch(Rectangle((-0.5, i - 0.5), 1, 1, fill=False, hatch="//", edgecolor="#aaaaaa", linewidth=0))
            data.append({"save": save_name, "pair": list(PAIRS)[i], "reading": r.key, "mean_gap": m,
                         "seeds": ";".join(format(x, "g") for x in v), "floor_band": band})
    axes[0].set_ylabel(save_name)
    return data
