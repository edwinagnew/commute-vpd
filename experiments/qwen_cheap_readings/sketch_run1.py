"""The card's sketch for P3.6 run 1: made-up numbers, one panel per rival. Not a result.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_cheap_readings/sketch_run1.py
"""

from __future__ import annotations

import random
from pathlib import Path

from commute import figures as F

HERE = Path(__file__).resolve().parent
NAMES = ["anchor 1 (2)", "anchor 2 (2)", "RL (2)", "RL null (2)", "RL F (1)", "GSM8K gold SFT (3)", "E45 SFT (3)", "known pool (2)", "never solved (2)"]
SIZES = [2, 2, 2, 2, 1, 3, 3, 2, 2]
FLOOR = 1850.0  # Gaussian erank at the four Qwen shapes, 1648-1953 (scratch measurement, card)


def made_up(centre: float, spread: float, seed: int) -> list[list[float]]:
    rnd = random.Random(seed)
    return [[centre + rnd.uniform(-spread, spread) * (1 + i % 3) for _ in range(n)] for i, n in enumerate(SIZES)]


def main() -> None:
    fig, axes = F.figure(F.FULL_WIDTH, ncols=2)
    data = []
    for ax, (rival, centre, spread) in zip(axes, (("A: low-rank updates", 1100.0, 120.0), ("B: noise-like updates", 1845.0, 6.0)), strict=True):
        vals = made_up(centre, spread, 0)
        F.dots(ax, dict(zip(NAMES, vals, strict=True)), method="spectrum", xlabel="effective rank of the update (made-up)")
        F.reference(ax, FLOOR, "random floor")
        ax.set_xlim(600, 2000)
        ax.set_title(f"Rival {rival}", fontsize=9)
        data += [{"rival": rival, "arm": n, "erank": v} for n, vs in zip(NAMES, vals, strict=True) for v in vs]
    axes[1].set_yticks([])
    fig.suptitle("Sketch: 19 saved Qwen updates against a matched-norm random floor")
    F.save(fig, HERE / "figures" / "sketch_run1.png", n="made-up numbers; 19 saved updates (seeds in brackets), floor = mean of two Gaussian draws", data=data)


if __name__ == "__main__":
    main()
