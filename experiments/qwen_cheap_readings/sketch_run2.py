"""The card's sketch for P3.6 run 2: made-up numbers, one panel per rival, drawn by plot_run2.gap_dots. Not a result.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_cheap_readings/sketch_run2.py
"""

from __future__ import annotations

import random
from pathlib import Path

from commute import figures as F
from commute.jobs import load_experiment

PLOT = load_experiment("qwen_cheap_readings", "plot_run2")
HERE = Path(__file__).resolve().parent
BAND = 0.05  # made up: about 1.5x run 1's two-draw spread (0.024-0.033), the range of four draws


def made_up(centre: float, spread: float, seed: int) -> dict[str, dict[str, float]]:
    rnd = random.Random(seed)
    return {n: {"overlap_rel_gap": centre + rnd.uniform(-spread, spread)} for ns in PLOT.ARMS.values() for n in ns}


def main() -> None:
    fig, axes = F.figure(F.FULL_WIDTH, ncols=2)
    data = PLOT.gap_dots(axes[0], made_up(-2.5, 1.5, 0), BAND, "Rival A: below the floor")
    data += PLOT.gap_dots(axes[1], made_up(0.0, 0.04, 1), BAND, "Rival B: at the floor")
    axes[1].set_yticks([])
    for ax in axes:
        ax.set_xlim(-5, 1)
    fig.suptitle("Sketch: 19 saved Qwen updates against their matched-norm random floors")
    F.save(fig, HERE / "figures" / "sketch_run2.png", n="made-up numbers; 19 saved updates in 12 arms; floor = mean of 4 draws", data=data)


if __name__ == "__main__":
    main()
