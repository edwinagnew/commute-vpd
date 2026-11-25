"""Figures for E49 (row_blocks), read from results/score.json and results/a.json.
Plan: reports/plans/2026-10-29_plan_e49-row-blocks.md; analysis: reports/analyses/2026-10-29_analysis_e49-row-blocks.md.

  blocks.png      per new block of 128 Pile val rows (512-1023), each state's ratio to W0 at adversary seeds 0-2
                  (small points) and their median (large marker); teacher-trained decompositions on the left, the
                  do-nothing references on the right, same axes; pass and fail lines, the "near the line" band and
                  W0 itself (ratio 1) drawn on both
  hard_block.png  rows 128-255 at seeds 0-2: the ratio of the two SGD teacher-trained decompositions and the LoRA's,
                  W0's own KL at each seed in the tick label, the same lines

Usage (repo root): ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/row_blocks/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "results" / "score.json").read_text())
A = json.loads((HERE / "results" / "a.json").read_text())
FIG = HERE / "figures"
PASS, FAIL, NEAR = 1.40, 1.61, (1.53, 1.69)
BLOCKS = ["512", "640", "768", "896"]
COLOR = {"adam": "#c8443a", "sgd": "#2a78d6", "sgd_r8": "#7b5ea7", "lora": "#2f9e6e"}
TRAINED = [("tt_adam", "adam", "Adam fine-tune"), ("tt_sgd", "sgd", "plain SGD fine-tune"),
           ("tt_sgd_r8", "sgd_r8", "SGD held to rank 8"), ("tt_lora", "lora", "rank-8 LoRA")]
NOTHING = [("nothing_adam", "adam", "Adam fine-tune"), ("nothing_sgd", "sgd", "plain SGD fine-tune"),
           ("nothing_lora", "lora", "rank-8 LoRA")]


def lines(ax: plt.Axes, label: bool) -> None:
    ax.axhspan(*NEAR, color="#999999", alpha=0.15, lw=0, label='"near the line" band' if label else None)
    ax.axhline(FAIL, color="#444444", ls="--", lw=1, label="fail line 1.61" if label else None)
    ax.axhline(PASS, color="#444444", ls=":", lw=1.2, label="pass line 1.40" if label else None)
    ax.axhline(1.0, color="#999999", lw=0.8, label="W₀ itself (ratio 1)" if label else None)


def panel(ax: plt.Axes, states: list, title: str) -> None:
    n = len(states)
    for j, (key, col, name) in enumerate(states):
        st = S["b"][key]
        for i, b in enumerate(BLOCKS):
            x = i + (j - (n - 1) / 2) * 0.17
            ax.scatter([x] * 3, st["ratios"][b], s=14, color=COLOR[col], alpha=0.55, lw=0)
            ax.scatter([x], [st["median"][b]], s=70, marker="D", color=COLOR[col], edgecolor="black", lw=0.6,
                       label=f"{name}: median of 3 starts ({st['fails']} of 4 blocks fail)" if i == 0 else None)
    lines(ax, label=ax.get_subplotspec().colspan.start == 0)
    ax.set_xticks(range(4), [f"rows {b}-{int(b) + 127}" for b in BLOCKS])
    ax.set_title(title, fontsize=10)
    ax.set_yscale("log")
    ax.set_yticks([1, 1.4, 1.61, 2, 3, 5, 8], ["1", "1.40", "1.61", "2", "3", "5", "8"])
    ax.minorticks_off()
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)


def blocks() -> None:
    fig, (l, r) = plt.subplots(1, 2, figsize=(12, 6.4), sharey=True)
    panel(l, TRAINED, "after teacher training (small points: adversary starts 0, 1, 2)")
    panel(r, NOTHING, "do-nothing: the fine-tune's δW left in Δ (the check's reference that must fail)")
    l.set_ylabel("adversarial KL over W₀'s, same block and start (log scale)")
    fig.suptitle("On four new blocks of Pile rows, which states does the check fail?")
    fig.tight_layout()
    fig.savefig(FIG / "blocks.png", dpi=150)


def hard_block() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for j, (key, col, name) in enumerate(TRAINED[1:]):
        y = S["a"][key]["hard_block"]
        x = [s + (j - 1) * 0.12 for s in range(3)]
        ax.plot(x, y, "-o", color=COLOR[col], label=f"{name}, after teacher training")
    lines(ax, label=True)
    w0 = [A["kl"][f"W0|128|{s}"] for s in range(3)]
    ax.set_xticks(range(3), [f"start {s}\nW₀'s own KL {w:.2f}" for s, w in enumerate(w0)])
    ax.set_ylabel("adversarial KL over W₀'s, rows 128-255")
    ax.set_title("Does the SGD fail on rows 128-255 survive two more adversary starts?", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "hard_block.png", dpi=150)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    blocks()
    hard_block()


if __name__ == "__main__":
    main()
