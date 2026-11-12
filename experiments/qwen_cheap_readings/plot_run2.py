"""P3.6 run 2's figure from results/run2_stage3.json: left, each update's top-set overlap beside its own random floor (the
mean of four draws at the update's per-matrix norms), on the overlap's own scale; right, every contrast on every reading
as the gap between means over the larger seed range (beyond +-1 separates).

    ~/commute-agents/fleet/cpu-run python experiments/qwen_cheap_readings/plot_run2.py [STAGE3 OUT.png]
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from commute import figures as F
from commute.jobs import load_experiment

R2 = load_experiment("qwen_cheap_readings", "rules_run2")
HERE = Path(__file__).resolve().parent
SINGLES = {"RL, F arm": "F_s0", "GSM8K SFT, E40 step 100": "u1", "gold recipe": "u2", "GSM8K SFT, E40 step 10": "e40", "E45 gold SFT": "gold_s0"}
ARMS = {**R2.PAIRS, **{k: (v,) for k, v in SINGLES.items()}}
# The pools in words: E120's problems the unchanged model passed, and those it never solved in 64 samples.
SHOWN = {"known pool": "passed pool", "known vs": "passed vs", "RL null": "RL shuffled-reward null"}
READING_NAMES = {"erank": "effective rank", "overlap": "top-set overlap", "on_both": "change on both sets", "size": "update size",
                 "prior_rank": "prior's rank", "steer": "steering", "random": "random steering"}  # fmt: skip


def shown(name: str) -> str:
    for old, new in SHOWN.items():
        name = name.replace(old, new)
    return name


def overlap_dots(ax: Axes, per: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each update's overlap and its floor on one axis, one row per arm (two seeds where the arm has them)."""
    real = {shown(a): [per[n]["overlap"] for n in ns] for a, ns in ARMS.items()}
    floor = {shown(a): [per[n]["overlap_floor"] for n in ns] for a, ns in ARMS.items()}
    F.dots(ax, real, method="neuron", xlabel="top-set overlap, before against after\n(1 = the same top 7.8% of neurons)")
    F.dots(ax, floor, method="random", xlabel="top-set overlap, before against after\n(1 = the same top 7.8% of neurons)")
    handles = [Line2D([], [], color=F.METHODS[m].color, marker=F.METHODS[m].marker, linestyle="none", label=lab)
               for m, lab in (("neuron", "real update"), ("random", "random change, same per-matrix norms"))]  # fmt: skip
    ax.legend(handles=handles, loc="center left", fontsize=7, frameon=False)
    ax.set_xlim(0.4, 1.02)
    ax.set_title("Every update moves the top set more than its random floor", fontsize=9)
    return [{"panel": "overlap", "arm": a, "update": n, "overlap": per[n]["overlap"], "overlap_floor": per[n]["overlap_floor"]}
            for a, ns in ARMS.items() for n in ns]  # fmt: skip


def contrast_map(ax: Axes, contrasts: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows, cols = list(contrasts), list(READING_NAMES)
    values = [[contrasts[r][c]["ratio"] if c in contrasts[r] else None for c in cols] for r in rows]
    F.gap_heatmap(ax, values, [shown(r) for r in rows], [READING_NAMES[c] for c in cols], vmax=R2.CLIP, fmt=".1f",
                  label="gap between means over the larger seed range\n(beyond +-1 separates; clipped at +-10; x not read)")  # fmt: skip
    ax.set_xticks(range(len(cols)), [READING_NAMES[c] for c in cols], rotation=40, ha="right")
    ax.set_title("Which reading separates which contrast, two seeds a side", fontsize=9)
    return [{"panel": "contrasts", "contrast": r, "reading": c, "ratio": v} for r, vs in zip(rows, values, strict=True) for c, v in zip(cols, vs, strict=True)]


def main(stage3: Path, out: Path) -> None:
    s3 = json.loads(stage3.read_text())
    fig, (left, right) = F.figure((12.0, 5.5), ncols=2)
    data = overlap_dots(left, s3["per_update"])
    data += contrast_map(right, s3["contrasts"])
    fig.suptitle("Top-set overlap and effective rank of 19 saved Qwen updates, against a random floor and seed spread\n"
                 "(a description of each update, not a reuse label)")  # fmt: skip
    F.save(fig, out, n="19 saved updates; overlap floor = mean of 4 random draws per update at its own per-matrix norms, "
           "effective-rank floor = mean of 2; 2 seeds per arm where the arm has them", data=data)  # fmt: skip


if __name__ == "__main__":
    a = sys.argv[1:] or [str(HERE / "results" / "run2_stage3.json"), str(HERE / "figures" / "run2.png")]
    main(Path(a[0]), Path(a[1]))
