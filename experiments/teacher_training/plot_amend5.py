"""Figures for teacher_training Amendment 5, why the harness and the budget sweep disagree and whether a copy is unique
(plan: reports/plans/2026-10-23_plan_teacher-training.md -> "Amendment 5"), read from the run's JSONs only. The
scoring is the run's own (`amend5.score`, in the results file under "score").

  amend5.png       every Part 1 reading's KL at the unchanged model and at the r1 SGD teacher's do-nothing state, on one
                   log axis, grouped by the random start the adversary took; the teacher's ratio to W0 beside each
  amend5_cos.png   the cosines between weight changes: each arm's two batch orders, each copy against the fine-tune,
                   the two arms, and the random-direction null, against the filed 0.8 and 0.5 lines

Usage (repo root): uv run python experiments/teacher_training/plot_amend5.py [--results DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
START_COLOR = {"harness": "#7b4fc9", "budget": "#1f9e89"}  # the two random starts
START_MARK = {"harness": "o", "budget": "s"}
ARM_COLOR = {"A": "#2a78d6", "B": "#eb6834"}  # plot.py's colours for the same arms
# (reading, its name, the start it took), left to right; the 20-step defaults first in each group
READINGS = [
    ("harness_b0", "harness\nrows 1-128", "harness"),
    ("harness_b1", "harness\nrows 129-256", "harness"),
    ("start.bf16", "budget code,\nharness's start,\nharness readout", "harness"),
    ("start", "budget code,\nharness's start", "harness"),
    ("harness160_b0", "harness\n160 steps", "harness"),
    ("budget", "budget code\nrows 1-128", "budget"),
    ("budget.bf16", "budget code,\nharness readout", "budget"),
    ("rows256", "budget code\nall 256 rows", "budget"),
    ("budget160", "budget code\n160 steps", "budget"),
]
Rec = dict[str, Any]


def style(ax: Axes) -> None:
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_readings(res: Rec, out: Path) -> None:
    sc = res["score"]
    w0, ratio = sc["w0"], sc["ratios"]
    teacher = {k: w0[k] * ratio[k] for k in ratio}
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for x, (key, _, start) in enumerate(READINGS):
        c = START_COLOR[start]
        ax.plot([x, x], [w0[key], teacher[key]], color=c, lw=1, alpha=0.4)
        ax.scatter(x, w0[key], s=44, marker=START_MARK[start], facecolors="white", edgecolors=MUTED, zorder=3)
        ax.scatter(x, teacher[key], s=48, marker=START_MARK[start], color=c, zorder=3)
        ax.text(x + 0.12, teacher[key], f"{ratio[key]:.2f}×", fontsize=8, va="center", ha="left", color=INK)
    zero = res["zero_step"]["harness"]
    ax.axhline(zero, color=MUTED, ls=":", lw=1)
    ax.text(len(READINGS) - 0.5, zero * 1.03, f"unchanged model at the harness's random start, no step taken: {zero:.2f}",
            fontsize=7, ha="right", va="bottom", color=MUTED)
    split = 4.5
    ax.axvline(split, color=GRID, lw=1)
    ax.text(2, 12, "from the harness's random start", ha="center", fontsize=9, color=START_COLOR["harness"])
    ax.text(6.5, 12, "from the budget code's random start", ha="center", fontsize=9, color=START_COLOR["budget"])
    ax.set_xticks(range(len(READINGS)), [name for _, name, _ in READINGS], fontsize=7.5)
    ax.set_xlim(-0.5, len(READINGS) - 0.3)
    ax.set_yscale("log")
    ax.set_ylim(0.18, 15)
    ax.set_yticks([0.2, 0.5, 1, 2, 5, 10], ["0.2", "0.5", "1", "2", "5", "10"])
    ax.set_ylabel("KL the adversary reaches, nats per position")
    ax.set_title("Does the teacher's reading depend on rows, readout, budget, or the random start?\n"
                 "The start: the teacher reads about 1.1 to 1.3 nats from one start and 6 to 7 from the other, while the unchanged "
                 "model reads 0.66 to 0.79 under both.\nLabels: the teacher's KL over the unchanged model's under the same reading", fontsize=9, loc="left")
    handles = [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=MUTED, label="unchanged model W₀"),
               Line2D([], [], marker="o", ls="", color=START_COLOR["harness"], label="r1 SGD teacher left in Δ, harness's start"),
               Line2D([], [], marker="s", ls="", color=START_COLOR["budget"], label="r1 SGD teacher left in Δ, budget code's start")]
    ax.legend(handles=handles, loc="center left", fontsize=7.5, frameon=False, bbox_to_anchor=(0.0, 0.62))
    style(ax)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.2)
    fig.savefig(out / "amend5.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out / 'amend5.png'}")


def fig_cosines(res: Rec, pred: Rec, out: Path) -> None:
    cos, cp = res["part2"]["cos"], res["part2"]["copies"]
    same, many = pred["amend5"]["same_way"], pred["amend5"]["many"]
    null = max(cos["null"].values())

    def p(k: str) -> str:
        return f"{cp[k]['p_target_1000']:.3f}"

    rows = [  # (label, value, colour)
        (f"task-text copies, batch order 0 against 1\nP(target) {p('A0')} and {p('A1')}", cos["seeds"]["A"], ARM_COLOR["A"]),
        (f"Pile-text copies, batch order 0 against 1\nP(target) {p('B0')} and {p('B1')}", cos["seeds"]["B"], ARM_COLOR["B"]),
        ("task-text copy, order 0, against the fine-tune δW", cos["with_dW"]["A0"], ARM_COLOR["A"]),
        ("task-text copy, order 1, against δW", cos["with_dW"]["A1"], ARM_COLOR["A"]),
        ("Pile-text copy, order 0, against δW", cos["with_dW"]["B0"], ARM_COLOR["B"]),
        ("Pile-text copy, order 1, against δW", cos["with_dW"]["B1"], ARM_COLOR["B"]),
        ("task-text against Pile-text copy, order 0", cos["A_vs_B_seed0"], INK),
        ("random direction against δW or a copy, largest", null, MUTED),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for y, (_, v, c) in enumerate(rows):
        ax.plot([0, v], [y, y], color=c, lw=2, solid_capstyle="round")
        ax.scatter(v, y, s=44, color=c, zorder=3)
        ax.text(v + 0.015, y, f"{v:.2f}" if v > 0.01 else f"{v:.0e}", fontsize=8, va="center")
    ax.axvline(same, color="g", ls="--", lw=0.8)
    ax.axvline(many, color=INK, ls=":", lw=0.8)
    ax.text(same + 0.01, len(rows) - 0.4, f"filed 'same way' line {same}", fontsize=7, color="g", va="bottom")
    ax.text(many - 0.01, len(rows) - 0.4, f"below {many}: one of many placements", fontsize=7, ha="right", va="bottom")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_ylim(len(rows) - 0.3, -0.9)
    ax.set_xlabel("cosine between weight changes, all 24 matrices together")
    ax.set_title("Does a second batch order give the same copy?\nOn task text nearly; on Pile text not: the two "
                 "Pile-text copies share\nthe behaviour but point 65 degrees apart", fontsize=9, loc="left")
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.38, right=0.97, top=0.84, bottom=0.12)
    fig.savefig(out / "amend5_cos.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out / 'amend5_cos.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    res = json.loads((args.results / "teacher_training_amend5.json").read_text())
    pred = json.loads((args.results / "predictions_amend5.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    fig_readings(res, args.out)
    fig_cosines(res, pred, args.out)


if __name__ == "__main__":
    main()
