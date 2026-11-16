"""The card's sketch: plot.py's figure drawn with made-up curves (reuse side rising faster than the new side on the first
two pairs, the two sides together on the rest), before any curve was read.

    ~/commute-agents/fleet/cpu-run python experiments/e184_curves/sketch.py
"""

from __future__ import annotations

from typing import Any

from commute.jobs import load_experiment

P: Any = load_experiment("e184_curves", "plot")


def made_up(top: float, rate: float) -> dict[str, Any]:
    best = [top * (1 - (1 - rate) ** n) for n in range(P.CAP + 1)]
    return {"best": best, "last_n": P.CAP}


def main() -> None:
    items: dict[str, Any] = {}
    for r, (a, b, _) in enumerate(P.PAIRS):
        for s in P.SEEDS:
            ra = 0.5 if r < 2 else 0.6
            items[f"{a}_s{s}"] = {"lists": {lst: made_up(0.9, ra) for lst in P.LISTS}}
            items[f"{b}_s{s}"] = {"lists": {lst: made_up(0.9 if r >= 2 else 0.4, 0.6 if r >= 2 else 0.1) for lst in P.LISTS}}
    band = {"lo": made_up(0.3, 0.1)["best"], "hi": made_up(0.4, 0.15)["best"]}
    P.draw(items, band, P.HERE / "figures" / "sketch.png", "made-up numbers: the planned figure, drawn before reading")


if __name__ == "__main__":
    main()
