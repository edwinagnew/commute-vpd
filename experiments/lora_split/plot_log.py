"""The research log's five figures for E14, redrawn from saved results with plain labels (no model, no new run).

  log_read.png        read side: the share of each update's input-weighted energy on what the subcomponents read,
                      over a random update's, against the step; full fine-tune and the 24-matrix LoRA, pooled over
                      the 20 non-saturated matrices (table[u][step].pooled_ns).
  log_write.png       write side: the same for what the subcomponents write.
  log_one_matrix.png  read side on the single matrix of the one-matrix LoRAs (layer 2's MLP output): the five
                      one-matrix LoRAs (ranks 1 to 32, two seeds at rank 8) beside the fine-tune's and the 24-matrix
                      LoRA's reading on that same matrix (table[u][step].per_matrix), at steps 10, 30 and 100.
  log_check.png       the adversarial check on Pile, as a ratio to the original model, for do-nothing and the
                      least-squares split of both updates at steps 10, 100 and 1000 (part2), with the pass line
                      1 + 3 x the original model's spread over three adversary seeds (W0.pgd_floor).
  log_rescaling.png   the rescaling share over its random reference against the step, pooled over all 24 matrices
                      (table[u][step].pooled_all, the pooling the prediction used).

Reads results/lora_split.json.
Usage (repo root):  python experiments/lora_split/plot_log.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FT, LORA, MUTED, INK = "#2a78d6", "#eb6834", "#9a9a9a", "#333333"
UPDATES = (("ft", "full fine-tune (Adam, all 24 matrices)", FT),
           ("lora24", "rank-8 LoRA on all 24 matrices", LORA))
SITE = "h.2.mlp.down_proj"
SITE_RUNS = (("site_r1_s0", "rank 1"), ("site_r4_s0", "rank 4"), ("site_r8_s0", "rank 8"),
             ("site_r8_s1", "rank 8, seed 2"), ("site_r32_s0", "rank 32"))


def side(p: dict[str, float], which: str) -> float:
    """Energy inside the subcomponents' read (or write) side over a random update's."""
    other = "out_in" if which == "read" else "in_out"
    return (p["in_in"] + p[other]) / (p["null_in_in"] + p["null_" + other])


def steps(out: dict[str, Any], u: str) -> list[str]:
    return sorted(out["table"][u], key=int)


def style(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)


def random_line(ax: Axes, x: float = 1.05, va: str = "bottom") -> None:
    ax.axhline(1, color=INK, linewidth=0.9, linestyle="--")
    ax.text(x, 1.0, "random update of the same size", fontsize=8, color=INK, va=va)


def save(fig: Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def fig_side(out: dict[str, Any], which: str, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for u, label, col in UPDATES:
        ss = steps(out, u)
        ys = [side(out["table"][u][s]["pooled_ns"], which) for s in ss]
        ax.plot([int(s) for s in ss], ys, "o-", color=col, label=label)
        ax.text(1000 * 1.12, ys[-1], f"{ys[-1]:.2f}", color=col, fontsize=8.5, va="center")
    random_line(ax)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 1600)
    ax.set_ylim(0.75, 1.5)
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel(f"share of the update's effect on what the\nsubcomponents {which}, over a random update's")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax)
    save(fig, name)


def fig_one_matrix(out: dict[str, Any]) -> None:
    cols = [(k, lab, MUTED) for k, lab in SITE_RUNS]
    cols += [("ft", "full fine-tune, read on this matrix", FT), ("lora24", "24-matrix LoRA, read on this matrix", LORA)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.12
    for i, s in enumerate(("10", "30", "100")):
        for j, (k, lab, col) in enumerate(cols):
            rec = out["table"][k][s]["per_matrix"][SITE]
            v = side(rec, "read")
            x = i + (j - (len(cols) - 1) / 2) * w
            ax.bar(x, v - 1, w * 0.9, bottom=1, color=col, alpha=0.55 if col == MUTED else 0.9,
                   label=(lab if col != MUTED else "one-matrix LoRAs: rank 1, 4, 8, 8 (second seed), 32") if i == 0 and j >= 4 else None)
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
            if col == MUTED:
                ax.text(x, 1.01, lab.replace("rank ", "r").replace(", seed 2", "\ns2"), ha="center", fontsize=6.5, va="bottom")
    random_line(ax, x=-0.55, va="top")
    ax.set_xticks(range(3), ["step 10", "step 30", "step 100"])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0.9, 1.7)
    ax.set_ylabel("share of the update's effect on what the\nsubcomponents read, over a random update's")
    ax.set_title("On one matrix, LoRAs of rank 1 to 32 read about the same, close to the fine-tune on that matrix", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    save(fig, "log_one_matrix.png")


def fig_check(out: dict[str, Any]) -> None:
    line = 1 + 3 * out["W0"]["pgd_floor"]
    arms = (("nothing", "left in Δ (do-nothing)", ""), ("lift", "least-squares split", "//"))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.19
    for i, s in enumerate(("10", "100", "1000")):
        j = 0
        for u, label, col in UPDATES:
            for arm, name, hatch in arms:
                v = out["part2"][u][s][arm]["pile"]["pgd_kl"]
                x = i + (j - 1.5) * w
                ax.bar(x, v - 1, w * 0.9, bottom=1, color=col, hatch=hatch, edgecolor="white",
                       label=f"{label.split(' (')[0]}, {name}" if i == 0 else None)
                ax.text(x, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
                j += 1
    ax.axhline(line, color=INK, linewidth=1, linestyle=":")
    ax.text(1.5, line + 0.05, f"pass line {line:.2f}", fontsize=8, ha="center")
    ax.axhline(1, color=INK, linewidth=0.8)
    ax.set_xticks(range(3), ["step 10", "step 100", "step 1000"])
    ax.set_ylim(0.9, 5.8)
    ax.set_ylabel("adversarial damage on Pile text,\nover the original decomposition's")
    ax.set_title("The fine-tune fails the adversarial check whether left in Δ or split, at most steps; the LoRA passes", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax)
    save(fig, "log_check.png")


def fig_rescaling(out: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for u, label, col in UPDATES:
        ss = steps(out, u)
        ys = [out["table"][u][s]["pooled_all"]["rescaling"] / out["table"][u][s]["pooled_all"]["rescaling_null"] for s in ss]
        ax.plot([int(s) for s in ss], ys, "o-", color=col, label=label)
        ax.text(1000 * 1.12, ys[-1], f"{ys[-1]:.1f}", color=col, fontsize=8.5, va="center")
    random_line(ax, va="top")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 1600)
    ax.set_ylim(0, 10.5)
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("share of the update a rescaling of the\nsubcomponents explains, over a random update's")
    ax.set_title("A growing part of the fine-tune is a rescaling of its subcomponents; the LoRA falls below random", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax)
    save(fig, "log_rescaling.png")


def main() -> None:
    out = json.loads((HERE / "results" / "lora_split.json").read_text())
    FIG.mkdir(exist_ok=True)
    fig_side(out, "read", "Read side: the fine-tune reads about 1.3 times random from step 1; the LoRA sits lower", "log_read.png")
    fig_side(out, "write", "Write side: both near random to step 30; then the fine-tune rises to 1.34, the LoRA falls below",
             "log_write.png")
    fig_one_matrix(out)
    fig_check(out)
    fig_rescaling(out)
    print(f"wrote {FIG}/log_*.png")


if __name__ == "__main__":
    main()
