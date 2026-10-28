"""Figure for E41 Amendment 1, the emoticon step-10 task-text copy at batch orders 1 and 2 (plan:
reports/plans/2026-10-28_plan_e41-teacher-path.md -> "Amendment 1"), read from the two runs' result files. The scoring is
the run's own (`path_amend1.py`, under "score" in the results file).

  e41_amend1_seeds.png  the Pile adversarial ratio of the step-10 copy at batch orders 0, 1 and 2 beside the step-10
                        fine-tune left in Δ (do-nothing) and the copy of the finished fine-tune at orders 0 and 1 (the
                        positive control G2 is order 1); each state's two 128-row blocks as small markers; the pass and
                        fail lines, the unchanged model's adversary-seed band (all rows) and its second block's highest
                        reading over those seeds; order 0's block values are hard-coded from E41's log (see SEED0_BLOCKS)

Usage (repo root): uv run python experiments/teacher_training/plot_e41_amend1.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from commute.faith_score import LINE

HERE = Path(__file__).resolve().parent
# Per-block ratios of E41's order-0 copies, from the [faith] lines of results/large/run_e41.log (lines 164-165 and
# 149-150) over W0's seed-0 blocks 0.6794 / 0.7182; E41 stored no blocks.
SEED0_BLOCKS = {"10": (0.7218 / 0.6794, 1.4397 / 0.7182), "1000": (0.6702 / 0.6794, 0.6703 / 0.7182)}
# The unchanged model's second block at adversary seed 1, its highest of seeds 0-2, over seed 0 (run_e41_amend1.log line 10)
W0_BLOCK2_MAX = 0.8277 / 0.7182
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
COPY, FINISHED = "#2a78d6", "#9cc3ee"  # plot_e41.py's task-text blue; a lighter blue for the finished teacher's copies
Rec = dict[str, Any]


def states(a1: Rec, e41: Rec) -> list[tuple[float, str, float, tuple[float, float], str]]:
    """(x, label, harness ratio, the two blocks' ratios, colour) per state, left to right."""
    nothing = a1["teachers"]["10"]["nothing"]
    out = [(0.0, "doing nothing:\nstep-10 fine-tune\nleft in Δ", nothing["pgd"], tuple(nothing["pgd_blocks"]), MUTED)]
    out.append((1.3, "copy of step 10\nbatch order 0\n(E41)", e41["tasks"]["emoticon"]["A"]["oneshot"]["10"]["pgd"],
                SEED0_BLOCKS["10"], COPY))
    for i, s in enumerate(("1", "2")):
        c = a1["copies"][s]
        out.append((2.3 + i, f"copy of step 10\nbatch order {s}", c["pgd"], tuple(c["pgd_blocks"]), COPY))
    out.append((5.0, "copy of step 1000\nbatch order 0\n(E41)", e41["tasks"]["emoticon"]["A"]["oneshot"]["1000"]["pgd"],
                SEED0_BLOCKS["1000"], FINISHED))
    g2 = a1["G2"]["record"]
    out.append((6.0, "copy of step 1000\nbatch order 1\n(control)", g2["pgd"], tuple(g2["pgd_blocks"]), FINISHED))
    return out


def fig_seeds(a1: Rec, e41: Rec, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    floor = a1["W0"]["pgd_floor"]
    ax.axhspan(1, 1 + floor, color=GRID, zorder=0)
    ax.text(-0.45, 1 + floor / 2, "unchanged model, 3 adversary seeds", fontsize=7.5, color=MUTED, va="center")
    for y, txt in ((LINE / (1 + floor), "pass at or below"), (LINE * (1 + floor), "fail at or above")):
        ax.axhline(y, color=INK, ls=":", lw=1)
        ax.text(6.45, y + 0.02, f"{txt} {y:.2f}", fontsize=7.5, va="bottom", ha="right")
    ax.axhline(W0_BLOCK2_MAX, color=MUTED, ls="--", lw=0.8)
    ax.text(4.7, W0_BLOCK2_MAX + 0.01, f"unchanged model, second 128 rows,\nhighest of 3 adversary seeds: {W0_BLOCK2_MAX:.2f}",
            fontsize=7.5, va="bottom", ha="right", color=MUTED)
    rows = states(a1, e41)
    for i, (x, _, r, (b1, b2), c) in enumerate(rows):
        ax.scatter(x, r, s=80, marker="o", color=c, zorder=4, label="all 256 rows: the check's ratio" if i == 1 else None)
        ax.annotate(f"{r:.2f}", (x, r), xytext=(0, 9), textcoords="offset points", fontsize=8.5, ha="center")
        ax.plot([x - 0.18, x - 0.18], [1, b1], color=c, lw=0.6, alpha=0.5, zorder=2)
        ax.plot([x + 0.18, x + 0.18], [1, b2], color=c, lw=0.6, alpha=0.5, zorder=2)
        ax.scatter(x - 0.18, b1, s=34, marker="v", facecolors="white", edgecolors=c, lw=1.2, zorder=3,
                   label="first 128 rows alone" if i == 1 else None)
        ax.scatter(x + 0.18, b2, s=34, marker="^", facecolors="white", edgecolors=c, lw=1.2, zorder=3,
                   label="second 128 rows alone" if i == 1 else None)
    ax.set_xticks([x for x, *_ in rows], [lab for _, lab, *_ in rows], fontsize=8)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0.8, 4.0)
    ax.set_ylabel("Pile adversarial ratio to the unchanged model")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    fig.suptitle("Does the copy of the step-10 fine-tune pass at other batch orders?\n"
                 "One of three passes (1.28) and two sit between the lines (1.55, 1.51); it spans 21% across batch orders,\n"
                 "the finished fine-tune's copy 3%, with one adversary start each. Nearly all the excess is in the second 128 rows",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "e41_amend1_seeds.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    a1 = json.loads((HERE / "results" / "teacher_training_e41_amend1.json").read_text())
    e41 = json.loads((HERE / "results" / "teacher_training_e41.json").read_text())
    assert a1["done"] and a1["card"] == e41["card"] == "NVIDIA H100 80GB HBM3"
    args.out.mkdir(parents=True, exist_ok=True)
    fig_seeds(a1, e41, args.out)


if __name__ == "__main__":
    main()
