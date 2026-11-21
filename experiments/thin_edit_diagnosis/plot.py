"""P1.47 figure, from the saved item JSONs only (no model, no GPU): per item, how far F moves when E242's stood-in set S
takes M's mechanism, with its fitted gains and with every gain forced to 1, as a share of KL(F || M). With --sketch,
the same figure with made-up numbers, one panel per rival (the card's sketch).

    ~/commute-agents/fleet/cpu-run python3 experiments/thin_edit_diagnosis/plot.py
    ~/commute-agents/fleet/cpu-run python3 experiments/thin_edit_diagnosis/plot.py --sketch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as FG

HERE = Path(__file__).resolve().parent
ITEMS = ("random_size_241001", "random_kl_241002", "x2", "x10", "rebuilt", "new_few", "new_thin")
SHORT = {"random_size_241001": "random\n(size)", "random_kl_241002": "random\n(KL)", "x2": "x2", "x10": "x10", "rebuilt": "rebuilt",
         "new_few": "new_few", "new_thin": "new_thin"}  # fmt: skip
FIT, ONE = FG.METHODS["regating"], FG.METHODS["neuron"]


def panel(ax: Axes, rows: list[dict[str, Any]], title: str) -> None:
    x = range(len(rows))
    ax.scatter(x, [r["q_fit"] for r in rows], color=FIT.color, marker=FIT.marker, s=26, label="fitted gains (E242's stand-in)")
    ax.scatter(x, [r["q_one"] for r in rows], color=ONE.color, marker=ONE.marker, s=26, label="every gain set to 1")
    ax.axhline(0.1, color="#777777", linewidth=0.8, linestyle="--", label="wording A's bar, 0.1")
    ax.set_xticks(list(x), [SHORT[r["item"]] for r in rows], fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("KL(F || F with S stood in) / KL(F || M)")
    ax.set_title(title, fontsize=8)


def sketch() -> None:
    a = [0.10, 0.10, 0.08, 0.10, 0.10, 0.10, 0.10]
    panels = {"rival a: gains absorb the new computation": [0.1, 0.1, 0.9, 1.5, 0.4, 0.3, 0.8],
              "rival b: the change sits in the few kept parts": [0.1, 0.1, 0.9, 1.5, 0.4, 0.12, 0.11]}  # fmt: skip
    fig, axs = FG.figure(FG.FULL_WIDTH, ncols=2)
    data = []
    for ax, (t, one) in zip(axs, panels.items(), strict=True):
        rows = [{"item": i, "q_fit": f, "q_one": o, "panel": t} for i, f, o in zip(ITEMS, a, one, strict=True)]
        panel(ax, rows, t)
        data += rows
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle("SKETCH, made-up numbers: does the thin new edit stand in because of its gains?", fontsize=9)
    FG.save(fig, HERE / "figures" / "sketch.png", n="one reading per item on all of D (E242's windows)", data=data)


def main() -> None:
    rows = []
    for i in ITEMS:
        p = HERE / "results" / "items" / f"{i}.json"
        if p.is_file():
            d = json.loads(p.read_text())["main"]
            rows.append({"item": i, "q_fit": d["q_fit"], "q_one": d["q_one"]})
    fig, ax = FG.figure()
    panel(ax, rows, "The thin new edit's stood-in set with gains fitted and set to 1")
    ax.legend(fontsize=7, frameon=False)
    FG.save(fig, HERE / "figures" / "gains_to_one.png", n="one reading per item on all of D (E242's windows)", data=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketch", action="store_true")
    sketch() if ap.parse_args().sketch else main()
