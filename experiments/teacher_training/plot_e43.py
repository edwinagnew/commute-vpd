"""Figures for E43, teacher training plus the remainder of the fine-tune in Δ (plan:
reports/plans/2026-10-28_plan_e43-teacher-remainder.md), read from `results/teacher_training_e43.json` only. The scoring is
the run's own (`remainder.py`, under "score" in the results file).

  e43_check.png    the Pile adversarial ratio of the four states at steps 10 and 1000: do-nothing, least squares,
                   teacher training's own state and teacher training plus remainder; the pass and fail lines and the
                   unchanged model's seed band
  e43_removal.png  KL to the fine-tune left after putting Δ' back to Δ, as a share of the unchanged model's, on held-out
                   task text and on Pile text, for the same states
  e43_delta.png    how much weight Δ' holds, ||Δ' - Δ||² / ||δW||², for the same states, beside the plan's expected band

Usage (repo root): uv run python experiments/teacher_training/plot_e43.py [--out DIR]
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

from commute.faith_score import LINE

HERE = Path(__file__).resolve().parent
STEPS = ("10", "1000")
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
# do-nothing and teacher training's own keep E41's colours (grey, blue)
STATES = {
    "nothing": ("do-nothing: fine-tune left in Δ", MUTED, "s"),
    "least_squares": ("least-squares split", "#7b52ab", "^"),
    "teacher_training": ("teacher training's own state (Δ untouched)", "#2a78d6", "o"),
    "plus_remainder": ("teacher training plus remainder (exact fine-tuned weights)", "#eb6834", "D"),
}
OFF = dict(zip(STATES, (-0.21, -0.07, 0.07, 0.21), strict=True))
Rec = dict[str, Any]


def style(ax: Axes) -> None:
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(range(len(STEPS)), [f"step {k}" for k in STEPS])
    ax.set_xlim(-0.5, len(STEPS) - 0.5)
    ax.set_xlabel("the fine-tune, stopped at")


def points(ax: Axes, res: Rec, key: str, names: tuple[str, ...], fmt: str, label: bool = True) -> None:
    for name in names:
        lab, c, m = STATES[name]
        xs = [i + OFF[name] for i in range(len(STEPS))]
        ys = [res["steps"][k][name][key] for k in STEPS]
        ax.scatter(xs, ys, s=64, marker=m, color=c, zorder=3, label=lab if label else None)
        for x, y in zip(xs, ys, strict=True):
            ax.annotate(format(y, fmt), (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5)


def fig_check(res: Rec, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.6))
    floor = res["W0"]["pgd_floor"]
    ax.axhspan(1, 1 + floor, color=GRID, zorder=0)
    ax.text(-0.47, 1 + floor / 2, "unchanged model, 3 adversary seeds", fontsize=7.5, color=MUTED, va="center")
    for y, txt in ((LINE / (1 + floor), "pass at or below"), (LINE * (1 + floor), "fail at or above")):
        ax.axhline(y, color=INK, ls=":", lw=1)
        ax.text(0.5, y + 0.03, f"{txt} {y:.2f}", fontsize=7.5, va="bottom", ha="center")
    points(ax, res, "pgd", tuple(STATES), ".2f")
    style(ax)
    ax.set_ylim(0.8, 7.6)
    ax.set_ylabel("Pile adversarial ratio to the unchanged model")
    ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    r = res["score"]["P3_check_paired"]["per"]
    fig.suptitle("Does putting the rest of the fine-tune into Δ keep teacher training's faithfulness?\n"
                 f"No: plus remainder reads {r['10']['ratio_over_own']:.1f}× and {r['1000']['ratio_over_own']:.1f}× "
                 "teacher training's own ratio, above do-nothing at both steps", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "e43_check.png", dpi=150)
    plt.close(fig)


def fig_removal(res: Rec, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    names = ("nothing", "least_squares", "plus_remainder")
    for ax, key, title, lines in (
        (axes[0], "b_task", "held-out task text (the trigger windows)", ((0.05, "kept the behaviour at or below 0.05"),)),
        (axes[1], "b_pile", "Pile text (256 validation rows)",
         ((0.2, "inert at or below 0.2"), (0.5, "carries at or above 0.5"))),
    ):
        for y, txt in lines:
            ax.axhline(y, color=INK, ls=":", lw=1)
            ax.text(-0.47, y * 1.07, txt, fontsize=7.5, va="bottom")
        points(ax, res, key, names, ".3g", label=ax is axes[1])
        style(ax)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylim(5e-4, 3)
    axes[0].set_ylabel("KL to the fine-tune with Δ' put back to Δ,\nshare of the unchanged model's KL")
    axes[1].text(0.5, 2.0, "do-nothing reads 1 by construction; teacher training's own state\nreads the same as plus "
                 "remainder: putting Δ' back lands on the same model", fontsize=7.5, ha="center", color=MUTED)
    axes[1].legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.suptitle("How much of the fine-tune's behaviour is lost when Δ' goes back to Δ?\n"
                 "Almost none on task text; on Pile text a quarter at step 10 and two-fifths at step 1000, "
                 "more than least squares loses", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "e43_removal.png", dpi=150)
    plt.close(fig)


def fig_delta(res: Rec, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.axhspan(0.9, 1.1, color="#fbe3d6", zorder=0)
    ax.text(-0.47, 0.05, "shaded: the plan's expectation\nfor plus remainder, 0.9 to 1.1", fontsize=7.5, va="top")
    points(ax, res, "delta_share", tuple(STATES), ".2g")
    style(ax)
    ax.set_yscale("log")
    ax.set_ylim(3e-10, 30)
    ax.set_ylabel("weight in Δ' beyond Δ, ‖Δ' − Δ‖² / ‖δW‖², all 24 matrices")
    ax.legend(fontsize=7.5, loc="center right", frameon=False)
    c = [res["steps"][k]["plus_remainder"]["cos_R_copy"] for k in STEPS]
    fig.suptitle("How large is the remainder put into Δ?\n"
                 f"At step 10, five times the fine-tune's own update; cosine with the copy's own weight change "
                 f"{c[0]:.2f} at step 10, {c[1]:.2f} at step 1000", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "e43_delta.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    res = json.loads((HERE / "results" / "teacher_training_e43.json").read_text())
    assert res["done"] and res["card"] == "NVIDIA H100 80GB HBM3"
    args.out.mkdir(parents=True, exist_ok=True)
    fig_check(res, args.out)
    fig_removal(res, args.out)
    fig_delta(res, args.out)


if __name__ == "__main__":
    main()
