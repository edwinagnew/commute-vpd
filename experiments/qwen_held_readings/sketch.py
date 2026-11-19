"""The card's sketch for P3.13 run 1: made-up numbers, one row of panels per rival's outcome, drawn by plot.draw. Not a
result. Erank numbers near P3.6's (RL 1,362-1,368, floor ~1,815); S of the random change made up.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_held_readings/sketch.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from commute import figures as F
from commute.jobs import load_experiment

PLOT = load_experiment("qwen_held_readings", "plot")
HERE = Path(__file__).resolve().parent
NAMES = ("R_s0", "R_s70", "N_s0", "N_s70", "F_s0", "F_s70")


def made_up(s_state: dict[str, float], gap: dict[str, float | None]) -> dict[str, dict[str, Any]]:
    out = {}
    for i, n in enumerate(NAMES):
        arm = n[0]
        out[n] = {"S": s_state[arm] + 0.004 * (i % 2), "S_random": 0.70 + 0.003 * (i % 2),
                  "gap": gap[arm], "band": 0.05 if gap[arm] is not None else None,
                  "share_class": "does not apply" if arm == "F" else "unreadable",
                  "erank": {"R": 1365.0, "N": 1380.0, "F": 1440.0}[arm] + 8 * (i % 2), "erank_floor": 1815.0}
    return out


def main() -> None:
    a = made_up({"R": 0.93, "N": 0.90, "F": 0.95}, {"R": 0.12, "N": 0.08, "F": None})
    b = made_up({"R": 0.71, "N": 0.70, "F": 0.69}, {"R": 0.01, "N": -0.02, "F": None})
    fig, axes = F.figure((F.FULL_WIDTH[0], 6.0), ncols=3, nrows=2)
    data = PLOT.draw(axes[0], a) + PLOT.draw(axes[1], b)
    axes[0][0].set_title("Rival A: M's units carry the update", fontsize=8)
    axes[1][0].set_title("Rival B: no closer than a random change", fontsize=8)
    fig.suptitle("Sketch (made-up numbers): S, share and effective rank of the step-75 states")
    F.save(fig, HERE / "figures" / "sketch.png", data=data,
           n="made-up numbers; 6 states (3 runs x 2 seeds); one random change per state at its per-matrix norms")


if __name__ == "__main__":
    main()
