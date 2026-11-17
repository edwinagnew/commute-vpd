"""The card's sketch for P3.9: made-up numbers, one row of panels per rival's outcome, drawn by P3.8's plot.draw on
the frozen-neuron decomposition p-015e0710. Not a result. Rival A: the neuron-basis share separates the built pairs
beyond the band; rival B: like p-9707d6ff, every gap stays inside the band.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_frozen_decomp/sketch.py
"""

from __future__ import annotations

from pathlib import Path

from commute import figures as F
from commute.jobs import load_experiment

PLOT = load_experiment("qwen_library_share", "plot")
SKETCH = load_experiment("qwen_library_share", "sketch")
HERE = Path(__file__).resolve().parent
RIVAL_A = {"copy_cipher": 0.015, "facts": 0.010, "gold": 0.001}  # mean seed gap per pair, made up
RIVAL_B = {"copy_cipher": 0.0015, "facts": -0.001, "gold": 0.0005}


def main() -> None:
    a, b = SKETCH.made_up(RIVAL_A, 2), SKETCH.made_up(RIVAL_B, 3)
    fig, axes = F.figure((10.0, 8.0), ncols=2, nrows=2)
    vmax = PLOT.vmax_of(a, b)
    data = PLOT.draw(axes[0], a, vmax) + PLOT.draw(axes[1], b, vmax)
    axes[0][0].set_title("Rival A: neuron-basis share, built pairs beyond the band", fontsize=8)
    axes[1][0].set_title("Rival B: as on p-9707d6ff, every gap inside the band", fontsize=8)
    fig.suptitle("Sketch (made-up numbers): re-gating share on the frozen-neuron decomposition p-015e0710")
    F.save(fig, HERE / "figures" / "sketch.png", data=data,
           n="made-up numbers; 3 seeds per side (dots); one random change per side (X); band = |random gap| + 0.002")


if __name__ == "__main__":
    main()
