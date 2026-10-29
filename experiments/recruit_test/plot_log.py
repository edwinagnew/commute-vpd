"""The research log's three figures for E6, redrawn from saved results with plain labels (no model, no runs).

  log_overlap.png   how much each update's switched-on set overlaps the fine-tune's (Jaccard), with each set's size,
                    so a small set cannot pass for a different one.
  log_share.png     each update's re-switching share: the part of its gain on its own target that the original
                    pieces get from Γ's new importance values alone; the random change has no gain to share.
  log_recovery.png  the part of that switching effect kept when only the switched-on set takes Γ's new values,
                    beside a random set of the same size.

Reads results/recruit.json (jaccard at threshold 0.3; arms.*.routes.shares.r, routes.gain, n_on, recovery,
recovery_random_control).
Usage (repo root):  python experiments/recruit_test/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from plot import HERE, OTHER, SAME, jaccard

MUTED = "#b8b8b8"
# arm -> (label, teaches "o" like the fine-tune), top to bottom
ARMS = {"ft_s1": ("the same fine-tune, second seed", True), "lora_r8": ("LoRA, rank 8, one matrix", True),
        "pullback_adam": ("component training", True), "null_a": ('fine-tune to "<|endoftext|>"', False),
        "null_b": ('fine-tune to " http"', False)}


def style(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend(fig: Any) -> None:
    handles = [Patch(color=SAME, label='teaches "o" after the emoticon trigger, like the fine-tune'),
               Patch(color=OTHER, label="teaches a different token after the same trigger")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5)


def save(fig: Any, name: str) -> None:
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / name, dpi=150)
    plt.close(fig)


def fig_overlap(res: dict[str, Any]) -> None:
    J, arms = res["jaccard"]["0.3"], res["arms"]
    names = list(ARMS)[::-1]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    vals = [jaccard(J, "ft_s0", n) for n in names]
    ax.barh(range(len(names)), vals, color=[SAME if ARMS[n][1] else OTHER for n in names])
    for i, (n, v) in enumerate(zip(names, vals, strict=True)):
        ax.text(v + 0.01, i, f"{v:.2f}   ({arms[n]['n_on']['0.3']} pieces switched on)", va="center", fontsize=8.5)
    ax.set_yticks(range(len(names)), [ARMS[n][0] for n in names], fontsize=9)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel(f"overlap with the fine-tune's switched-on set ({arms['ft_s0']['n_on']['0.3']} pieces)\n"
                  "0 = no piece in common, 1 = the same set")
    ax.set_title('Updates that teach "o" switch on mostly the fine-tune\'s pieces;\n'
                 "updates teaching another token switch on almost none of them", fontsize=10)
    style(ax)
    legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "log_overlap.png")


def fig_share(res: dict[str, Any]) -> None:
    arms = res["arms"]
    names = ["random_dw", *list(ARMS)[::-1], "ft_s0"]
    labels = {**{n: ARMS[n][0] for n in ARMS}, "ft_s0": "the fine-tune", "random_dw": "random change, same size"}
    fig, ax = plt.subplots(figsize=(9, 4.1))
    for i, n in enumerate(names):
        gain = arms[n]["routes"]["gain"]
        if n == "random_dw":
            ax.text(0.01, i, f"no gain ({gain:+.2f} nats), nothing to share", va="center", fontsize=8.5, color="0.35")
            continue
        r = arms[n]["routes"]["shares"]["r"]
        ax.barh(i, r, color=OTHER if n.startswith("null") else SAME)
        ax.text(r + 0.01, i, f"{r:.2f}   (gain {gain:.0f} nats)", va="center", fontsize=8.5)
    ax.set_yticks(range(len(names)), [labels[n] for n in names], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("re-switching share: part of the update's gain on its own target that the\n"
                  "original pieces get from Γ's new importance values alone (0 none, 1 all)")
    ax.set_title("Γ's switching alone gives about half or more of every update's gain,\n"
                 "whichever token it teaches", fontsize=10)
    style(ax)
    legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "log_share.png")


def fig_recovery(res: dict[str, Any]) -> None:
    arms = res["arms"]
    names = [*list(ARMS)[::-1], "ft_s0"]
    labels = {**{n: ARMS[n][0] for n in ARMS}, "ft_s0": "the fine-tune"}
    fig, ax = plt.subplots(figsize=(9, 4.1))
    h = 0.38
    for i, n in enumerate(names):
        rec, rnd = arms[n]["recovery"], arms[n]["recovery_random_control"]
        ax.barh(i + h / 2, rec, h, color=OTHER if n.startswith("null") else SAME)
        ax.barh(i - h / 2, rnd, h, color=MUTED)
        ax.text(rec + 0.01, i + h / 2, f"{rec:.2f}", va="center", fontsize=8.5)
        ax.text(rnd + 0.01, i - h / 2, f"{rnd:.2f}", va="center", fontsize=8, color="0.35")
    ax.set_yticks(range(len(names)), [labels[n] for n in names], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("part of the switching effect kept when only this set takes Γ's new values\n"
                  "(every other piece keeps its original value; 0 none, 1 all)")
    ax.set_title("The switched-on pieces alone keep only part of the switching effect (0.40 for the fine-tune);\n"
                 "random sets of their size keep almost none", fontsize=10)
    style(ax)
    handles = [Patch(color=SAME, label="the update's switched-on set (blue: teaches \"o\"; orange: another token)"),
               Patch(color=MUTED, label="a random set of the same size")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "log_recovery.png")


def main() -> None:
    res = json.loads((HERE / "results" / "recruit.json").read_text())
    fig_overlap(res)
    fig_share(res)
    fig_recovery(res)


if __name__ == "__main__":
    main()
