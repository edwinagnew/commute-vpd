"""E233's figure: the re-gating share, the update's effective rank and the neurons for half the gain, each seed a point,
British spelling against its scrambled twin at the scored size (results/scored/score.json, written by `run.py score`).

    python experiments/british_style/plot.py              (rehearse: plot.py rehearse, to real_vs_twin_rehearse.png)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from commute.figures import FULL_WIDTH, dots, figure, save

HERE = Path(__file__).resolve().parent
SIDES = {"reuse": "British spelling", "new": "scrambled twin"}
PANELS = (("share", "regating", "re-gating share r"), ("erank", "spectrum", "effective rank of the update"),
          ("k_half", "neuron_half", "neurons for half the gain (log2)"))  # fmt: skip


def main(folder: str) -> None:
    res = json.loads((HERE / "results" / folder / "score.json").read_text())
    size = max(res["sizes"], key=float)
    fig, axes = figure((FULL_WIDTH[0], 2.6), ncols=3)
    rows = []
    for ax, (m, style, label) in zip(axes, PANELS, strict=True):
        c = res["sizes"][size][m]
        groups = {SIDES[s]: c[s] for s in SIDES}
        if m == "k_half":
            ax.set_xscale("log", base=2)
        dots(ax, groups, method=style, xlabel=label)
        ax.set_title(f"{c['call']}, gap {float(c['gap']):.2f} ({c['filed']})", fontsize=8)
        rows += [{"method": m, "side": side, "value": v, "size": size} for side, vs in groups.items() for v in vs]
    fig.suptitle(f"British spelling against its scrambled twin at update size {size}: card {res['card']}", fontsize=9)
    name = "real_vs_twin.png" if folder == "scored" else f"real_vs_twin_{folder}.png"
    seeds = len(res["steps"]["real"])
    save(fig, HERE / "figures" / name, n=f"{seeds} seeds per side, 500 read windows each", data=rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scored")
