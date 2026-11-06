"""E121's three figures, from results/copy_gap_ladder.json (job 54638):

    python experiments/copy_gap_ladder/plot.py
      -> figures/ladder.png     input-weighted cosine to dW per rung, task windows and Pile, with the rerun line
      -> figures/remainder.png  P(target) of each state and of the same state with its remainder removed (W0 + k dW)
      -> figures/function.png   functional cosine at the trigger per rung, with the rerun line
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
OUT = json.loads((RESULTS / "copy_gap_ladder.json").read_text())
RUNGS = ["R0", "R1", "R2", "R3", "R3b", "R4"]
LABEL = {
    "R0": "rerun of the\nfine-tune",
    "R1": "all weights,\nKL to fine-tuned,\nlr 3e-5",
    "R2": "all weights,\nKL to fine-tuned,\nlr 1e-5",
    "R3": "U, V only,\nAdam 1e-5\n(teacher training)",
    "R3b": "U, V only,\nplain SGD 0.01",
    "R4": "U, V only,\nAdam 1e-5, from\nleast-squares split",
}
SET_COLOUR = {"task": "#2a78d6", "pile": "#eb6834"}  # dataviz default slots 1-2
SET_NAME = {"task": "held-out task windows", "pile": "Pile text"}
FULL, REMOVED = "#2b2b2b", "#1baf7a"
INK, MUTED, BAND = "#2b2b2b", "#8a8a85", "#e6e5df"


def orders(r: str) -> list[str]:
    return [f"{r}_o1", f"{r}_o2"]


def axes_rungs(ax: Any, region_y: float = 0.02) -> None:
    ax.set_xticks(range(len(RUNGS)), [LABEL[r] for r in RUNGS], fontsize=7.5)
    ax.set_xlim(-0.5, len(RUNGS) - 0.5)
    ax.axvspan(2.5, 5.5, color="#f4f3ee", zorder=0)
    ax.text(4, region_y, "trained in the decomposition's U, V", ha="center", fontsize=7.5, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)


def rerun_band(ax: Any, y: float, m: float, text: str) -> None:
    ax.axhspan(y - m, y + m, color=BAND, zorder=0.5)
    ax.axhline(y, color=MUTED, ls="--", lw=1)
    ax.text(-0.45, y - m - 0.01, text, va="top", fontsize=7.5, color=MUTED)


def ladder() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    axes_rungs(ax)
    ax.axhline(1.0, color=INK, lw=0.8)
    ax.text(-0.45, 1.0, "the fine-tune itself, regenerated: 1.000", va="bottom", fontsize=7.5, color=INK)
    for s, dx in (("task", -0.12), ("pile", 0.12)):
        cos, line, m = OUT["reads"][s]["cos"], OUT["score"][s]["line"][0], OUT["score"][s]["margin"]
        if s == "task":
            rerun_band(ax, line, m, "rerun line ± margin")
        ax.axhline(line, color=SET_COLOUR[s], ls=":", lw=0.8)
        for i, r in enumerate(RUNGS):
            v = [cos[n] for n in orders(r)]
            ax.plot([i + dx] * 2, v, "o", ms=3.5, color=SET_COLOUR[s], alpha=0.5)
            ax.plot(i + dx, sum(v) / 2, "_", ms=14, mew=2, color=SET_COLOUR[s], label=SET_NAME[s] if i == 0 else None)
        ax.plot(5 + dx, cos["R4_start"], "o", ms=7, mfc="none", mec=SET_COLOUR[s], label=f"least-squares split before training, {SET_NAME[s]}")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("input-weighted cosine of the change to the fine-tune's δW")
    ax.set_title("Which difference moves the change away from the fine-tune's? Most: training U, V; part: the lower lr", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left", frameon=False, bbox_to_anchor=(0.0, 0.06))
    ax.text(-0.45, 0.06, "do-nothing (W0): no change, cosine undefined", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "ladder.png", dpi=150)


def remainder() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    axes_rungs(ax, region_y=0.75)
    b, ex, line = OUT["behaviour"], OUT["explain"], OUT["line"]
    bar = line["p_teacher"] - 0.05  # the filed copies rule's P(target) side
    ax.axhline(line["p_teacher"], color=INK, lw=0.8)
    ax.text(-0.45, line["p_teacher"] + 0.01, f"the fine-tune: {line['p_teacher']:.3f}", fontsize=7.5, color=INK)
    ax.axhline(bar, color=MUTED, ls="--", lw=1)
    ax.text(-0.45, bar - 0.01, "keeps the behaviour above this line", va="top", fontsize=7.5, color=MUTED)
    ax.axhline(line["p_w0"], color=MUTED, lw=1)
    ax.text(-0.45, line["p_w0"] + 0.01, f"do-nothing W0: {line['p_w0']:.3f}", fontsize=7.5, color=MUTED)
    for i, r in enumerate(RUNGS):
        full = [b[n]["p_target"] for n in orders(r)]
        cut = [ex[n]["without_remainder"]["p_target"] for n in orders(r)]
        ax.plot([i - 0.1] * 2, full, "o", ms=5, color=FULL, label="the trained state" if i == 0 else None)
        ax.plot([i + 0.1] * 2, cut, "s", ms=5, color=REMOVED, label="the same state with its remainder removed: k δW" if i == 0 else None)
        k = sum(ex[n]["k"] for n in orders(r)) / 2
        ax.text(i + 0.18, min(cut) - 0.035, f"k = {k:.2f}", va="top", fontsize=7.5, color=REMOVED)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("P(target) at the trigger, held-out windows")
    ax.set_title("Does the part along δW alone keep the behaviour? Yes for reruns; no when trained at lr 1e-5 from zero", fontsize=10)
    ax.legend(fontsize=7.5, loc="center left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "remainder.png", dpi=150)


def function() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    axes_rungs(ax)
    b, p4 = OUT["behaviour"], OUT["score"]["P4_function"]
    ax.axhline(b["F0"]["fcos"], color=INK, lw=0.8)
    ax.text(-0.45, 1.0, "the fine-tune itself, regenerated: 1.000", va="bottom", fontsize=7.5, color=INK)
    rerun_band(ax, p4["rerun"], p4["margin"], "rerun mean ± filed margin")
    for i, r in enumerate(RUNGS):
        v = [b[n]["fcos"] for n in orders(r)]
        ax.plot([i] * 2, v, "o", ms=5, color=INK, alpha=0.6)
        ax.plot(i, p4["per_rung"][r], "_", ms=16, mew=2, color=INK, label="batch orders 1, 2 and their mean" if i == 0 else None)
    ax.plot(5, b["R4_start"]["fcos"], "o", ms=7, mfc="none", mec=INK, label="least-squares split before training")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("cosine of the change in trigger logits to the fine-tune's")
    ax.set_title("Is the function at the trigger as close as a rerun's? Not for teacher training", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left", frameon=False, bbox_to_anchor=(0.0, 0.06))
    ax.text(-0.45, 0.06, "do-nothing (W0): no change, cosine undefined", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "function.png", dpi=150)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    ladder()
    remainder()
    function()


if __name__ == "__main__":
    main()
