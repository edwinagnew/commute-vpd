"""Figure for the recruit test (E6), from results/recruit.json.

  overlap.png  Is the set of subcomponents Γ switches on specific to the target, and is the re-switching share? Left:
               each update's switched-on set against the fine-tune's (seed 0), as Jaccard overlap. Right: each
               update's re-switching share of the gain on its own target.

Usage (repo root):  uv run python experiments/recruit_test/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
SAME, OTHER = "#2a78d6", "#eb6834"
# arm -> (label, teaches the same target as the reference fine-tune)
ARMS = {"ft_s1": ("fine-tune, second seed", True), "lora_r8": ("LoRA, rank 8", True),
        "pullback_adam": ("component training", True), "null_a": ('fine-tune to "<|endoftext|>"', False),
        "null_b": ('fine-tune to " http"', False)}


def jaccard(J: dict[str, float], a: str, b: str) -> float:
    return J.get(f"{a}|{b}", J.get(f"{b}|{a}", float("nan")))


def main() -> None:
    res = json.loads((HERE / "results" / "recruit.json").read_text())
    J, arms = res["jaccard"]["0.3"], res["arms"]
    names = list(ARMS)[::-1]
    colors = [SAME if ARMS[a][1] else OTHER for a in names]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    a.barh(names, [jaccard(J, "ft_s0", n) for n in names], color=colors)
    a.set_xlim(0, 1)
    a.set_xlabel("overlap with the fine-tune's switched-on set (Jaccard)")
    a.set_title("Which subcomponents Γ switches on:\nspecific to the target", fontsize=10)
    b.barh(names, [arms[n]["routes"]["shares"]["r"] for n in names], color=colors)
    b.set_xlim(0, 1)
    b.set_xlabel("re-switching share of the gain on the update's own target")
    b.set_title("How much of the gain runs through switching:\nthe same for every target", fontsize=10)
    a.set_yticks(range(len(names)), [ARMS[n][0] for n in names])
    for ax in (a, b):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [Patch(color=SAME, label='teaches "o" at the emoticon trigger, like the reference'),
               Patch(color=OTHER, label="teaches a different token at the same trigger")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "overlap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
