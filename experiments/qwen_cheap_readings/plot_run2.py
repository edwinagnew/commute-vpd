"""P3.6 run 2's figure from results/run2_stage{1,3}.json: left, each update's top-set overlap against its own random floor;
right, every contrast on every reading as the gap between means over the larger seed range (beyond +-1 separates).

    ~/commute-agents/fleet/cpu-run python experiments/qwen_cheap_readings/plot_run2.py [STAGE3 STAGE1 OUT.png]
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as F
from commute.jobs import load_experiment

R2 = load_experiment("qwen_cheap_readings", "rules_run2")
HERE = Path(__file__).resolve().parent
SINGLES = {"RL, F arm": "F_s0", "GSM8K SFT, E40 step 100": "u1", "gold recipe": "u2", "GSM8K SFT, E40 step 10": "e40", "E45 gold SFT": "gold_s0"}
ARMS = {**R2.PAIRS, **{k: (v,) for k, v in SINGLES.items()}}
READING_NAMES = {"erank": "effective rank", "overlap": "top-set overlap", "on_both": "change on both sets", "size": "update size",
                 "prior_rank": "prior's rank", "steer": "steering", "random": "random steering"}  # fmt: skip


def gap_dots(ax: Axes, per: Mapping[str, Mapping[str, Any]], band: float, title: str) -> list[dict[str, Any]]:
    """Each update's overlap minus its floor, over the floor's distance from 1.0, one row per arm; the band either side."""
    groups = {arm: [per[n]["overlap_rel_gap"] for n in names if per[n]["overlap_rel_gap"] is not None] for arm, names in ARMS.items()}
    F.dots(ax, groups, method="neuron", xlabel="overlap minus its random floor\n(units: the floor's distance from 1.0)")
    ax.axvspan(-band, band, color="#999999", alpha=0.35, linewidth=0, label="random floor and its band")
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    ax.set_title(title, fontsize=9)
    return [{"panel": title, "arm": a, "update": n, "overlap_rel_gap": per[n]["overlap_rel_gap"]} for a, ns in ARMS.items() for n in ns]


def contrast_map(ax: Axes, contrasts: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows, cols = list(contrasts), list(READING_NAMES)
    values = [[contrasts[r][c]["ratio"] if c in contrasts[r] else None for c in cols] for r in rows]
    F.gap_heatmap(ax, values, rows, [READING_NAMES[c] for c in cols], vmax=R2.CLIP, fmt=".1f",
                  label="gap between means over the larger seed range\n(beyond +-1 separates; x not read)")  # fmt: skip
    ax.set_xticks(range(len(cols)), [READING_NAMES[c] for c in cols], rotation=40, ha="right")
    ax.set_title("Which reading separates which contrast", fontsize=9)
    return [{"panel": "contrasts", "contrast": r, "reading": c, "ratio": v} for r, vs in zip(rows, values, strict=True) for c, v in zip(cols, vs, strict=True)]


def main(stage3: Path, stage1: Path, out: Path) -> None:
    s3, s1 = json.loads(stage3.read_text()), json.loads(stage1.read_text())
    band, draws = s1["band"]["overlap"], len(next(iter(s1["refs"].values()))["draws"])
    fig, (left, right) = F.figure((11.0, 5.0), ncols=2)
    c = s3["counts"]
    data = gap_dots(left, s3["per_update"], band, f"Overlap against a matched random change: {c.get('below floor', 0)} below, "
                    f"{c.get('at floor', 0)} at, {c.get('above floor', 0)} above")  # fmt: skip
    data += contrast_map(right, s3["contrasts"])
    fig.suptitle(f"Top-set overlap of 19 saved Qwen updates: call {s3['call']} (a description, not a reuse label)")
    F.save(fig, out, n=f"19 saved updates; floor = mean of {draws} random draws per update at its own per-matrix norms; 2 seeds per arm", data=data)


if __name__ == "__main__":
    a = sys.argv[1:] or [str(HERE / "results" / f"run2_stage{n}.json") for n in (3, 1)] + [str(HERE / "figures" / "run2.png")]
    main(Path(a[0]), Path(a[1]), Path(a[2]))
