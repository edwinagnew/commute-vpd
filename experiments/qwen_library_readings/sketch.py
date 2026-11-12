"""The card's sketch for P3.7: made-up numbers, one row of panels per rival, drawn by plot.scorecard. Not a result.

The run's figure has the same layout with one row per save (late, matched); here both rows are the late save, one per
rival's outcome. Size and prior gaps are set near the library page's numbers; the rest are invented.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_library_readings/sketch.py
"""

from __future__ import annotations

import random
from pathlib import Path

from commute import figures as F
from commute.jobs import load_experiment

PLOT = load_experiment("qwen_library_readings", "plot")
HERE = Path(__file__).resolve().parent
BANDS = {p: {"steer": 0.1, "erank": 25.0, "overlap": 0.03} for p in ("copy_cipher", "facts", "gold", "template")}  # made up
SHARED = {  # near the library page: |δW| over an assumed |W0| of 1,100; prior log P from its table
    "copy_cipher": {"size": -0.0025, "prior_rank": 3.0, "prior_logp": -1.0},
    "facts": {"size": -0.0019, "prior_rank": -40.0, "prior_logp": -0.3},
    "gold": {"size": -0.0001, "prior_rank": 1.0, "prior_logp": 0.3},
    "template": {"size": -0.0001, "prior_rank": 0.0, "prior_logp": 0.8},
}
RIVAL_A = {"copy_cipher": (0.25, -150.0, 0.08), "facts": (0.2, -120.0, 0.06), "gold": (0.05, -10.0, 0.01), "template": (0.0, 5.0, 0.0)}
RIVAL_B = {"copy_cipher": (0.04, -15.0, -0.02), "facts": (-0.03, -10.0, -0.01), "gold": (0.02, 3.0, 0.0), "template": (0.0, 2.0, 0.01)}


def seeds(mean: float, rnd: random.Random) -> list[float]:
    return [mean + abs(mean) * rnd.uniform(-0.2, 0.2) for _ in range(3)]


def made_up(update: dict[str, tuple[float, float, float]], seed: int) -> dict[str, dict[str, list[float]]]:
    rnd = random.Random(seed)
    out = {}
    for pair, base in SHARED.items():
        means = {**base, **dict(zip(("steer", "erank", "overlap"), update[pair], strict=True))}
        out[pair] = {k: seeds(m, rnd) for k, m in means.items()}
    return out


def main() -> None:
    a, b = made_up(RIVAL_A, 0), made_up(RIVAL_B, 1)
    fig, axes = F.figure((10.0, 5.5), ncols=len(PLOT.READINGS), nrows=2)
    vmax = PLOT.column_vmax([a, b])
    data = PLOT.scorecard(axes[0], a, BANDS, vmax, save_name="Rival A: beyond the floor")
    data += PLOT.scorecard(axes[1], b, BANDS, vmax, save_name="Rival B: size only")
    fig.suptitle("Sketch: cheap readings on the pair library, side A minus side B at the late save (step 300; twins 100)")
    F.save(fig, HERE / "figures" / "sketch.png", data=data,
           n="made-up numbers; 3 seeds per side; cell = mean A − B over seeds, seed range under it; colour scaled per "
             "reading (red: A higher); hatched = inside the random-floor band")


if __name__ == "__main__":
    main()
