"""The card's sketch for P3.8: made-up numbers, one row of panels per rival's outcome, drawn by plot.draw. Not a result.
Random gaps and the band are set near E132's numbers (shares 0.991-0.999, random-vs-random gap 0.001, seed spread 0.002).

    ~/commute-agents/fleet/cpu-run python experiments/qwen_library_share/sketch.py
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from commute import figures as F
from commute.jobs import load_experiment

PLOT = load_experiment("qwen_library_share", "plot")
HERE = Path(__file__).resolve().parent
SPREAD = 0.002
RANDOM_GAP = {"copy_cipher": 0.001, "facts": -0.0015, "gold": 0.0005}  # made up
RIVAL_A = {"copy_cipher": 0.012, "facts": 0.009, "gold": 0.001}  # mean seed gap per pair, made up
RIVAL_B = {"copy_cipher": 0.0015, "facts": -0.001, "gold": 0.0005}


def made_up(means: dict[str, float], seed: int) -> dict[str, Any]:
    rnd, out = random.Random(seed), {}
    for pair, m in means.items():
        for kind in ("late", "matched"):
            gaps: list[float | None] = [m + rnd.uniform(-0.0015, 0.0015) for _ in range(3)]
            band = abs(RANDOM_GAP[pair]) + SPREAD
            out[f"{pair}/{kind}"] = {"pair": pair, "kind": kind, "gaps": gaps, "random_gap": RANDOM_GAP[pair],
                                     "band": band, "cell": "inside band" if abs(m) <= band else "beyond band"}
    return out


def main() -> None:
    a, b = made_up(RIVAL_A, 0), made_up(RIVAL_B, 1)
    fig, axes = F.figure((10.0, 8.0), ncols=2, nrows=2)
    vmax = PLOT.vmax_of(a, b)
    data = PLOT.draw(axes[0], a, vmax) + PLOT.draw(axes[1], b, vmax)
    axes[0][0].set_title("Rival A: built pairs beyond the band", fontsize=8)
    axes[1][0].set_title("Rival B: every gap inside the band", fontsize=8)
    fig.suptitle("Sketch (made-up numbers): re-gating share, side A minus side B, by pair and save")
    F.save(fig, HERE / "figures" / "sketch.png", data=data,
           n="made-up numbers; 3 seeds per side (dots); one random change per side (X); band = |random gap| + 0.002")


if __name__ == "__main__":
    main()
