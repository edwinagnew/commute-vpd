"""P2.9's one figure: per task pair, the gap between the two built sides over the larger within-side range, for the
update's effective rank beside the re-gating share, on the same saved states. |value| > 1 separates (E51's rule).

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/update_svd/plot.py
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as F

HERE = Path(__file__).resolve().parent
TITLE = "Does the update's effective rank separate each pair's sides as the share does?"
XLABEL = "(reuse side minus new side) / larger within-side range"


def draw(ax: Axes, rows: Sequence[dict[str, Any]]) -> None:
    """rows: {"pair": name, "r": ratio, "erank": ratio}; one row per pair, the two readings side by side."""
    groups = {row["pair"]: row for row in rows}
    F.dots(ax, {k: [v["r"]] for k, v in groups.items()}, method="regating", xlabel=XLABEL, offset=-0.15)
    F.dots(ax, {k: [v["erank"]] for k, v in groups.items()}, method="spectrum", xlabel=XLABEL, offset=0.15)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.axvspan(-1, 1, color="#E4E4E4", zorder=0)
    F.reference(ax, -1.0, "grey: does not separate")


def legend(ax: Axes, n_rows: int) -> None:
    ax.figure.legend(handles=[ax.collections[0], ax.collections[n_rows]], fontsize=7, loc="outside right upper",
                     labels=[F.METHODS["regating"].label, F.METHODS["spectrum"].label])


def main() -> None:
    res = json.loads((HERE / "results" / "results.json").read_text())
    rows = [{"pair": v["name"], "r": v["score"]["r"]["ratio"], "erank": v["score"]["erank"]["ratio"],
             "n_reuse": len(v["reuse"]), "n_new": len(v["new"])} for v in res["pairs"].values()]
    fig, ax = F.figure(F.FULL_WIDTH)
    draw(ax, rows)
    legend(ax, len(rows))
    ax.set_title(TITLE, fontsize=9)
    F.save(fig, HERE / "figures" / "gaps.png", n=f"{len(rows)} task pairs, 2 to 8 saved seeds per side", data=rows)


if __name__ == "__main__":
    main()
