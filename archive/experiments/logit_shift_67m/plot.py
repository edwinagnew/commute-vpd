"""E83's figure: the whole update's effect on the 67M's logits under each kind of mask, in the style of Qwen's
experiments/qwen_open_points/figures/logit_shift.png. Usage: python experiments/logit_shift_67m/plot.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, GRID = "#2b2a27", "#e4e3df"
GREY, BLUE = "#8a8984", "#2a78d6"
BARS = (("Random masks, random subset\n(how VPD trains)", "random_subset"), ("Random masks,\nall 24 matrices", "random_all"),
        ("Fixed masks,\nall 24 matrices", "fixed_all"))


def amps(update: str) -> list[tuple[str, float, str]]:
    read = [(name, json.loads((RES / update / f"{key}.json").read_text())["amp"], BLUE) for name, key in BARS]
    return [("No masks", 1.0, GREY), *read]


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), dpi=170, sharey=True)
    for ax, update, name in zip(axes, ("adam", "lora"), ("Adam fine-tune", "Rank-8 LoRA"), strict=True):
        rows = amps(update)
        ax.barh(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows], height=0.55)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=11)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Size of the update's effect on the logits (no masks = 1)", fontsize=10)
        ax.set_title(name, fontsize=11, color=INK)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("On the 67M too, the decomposition's masks shrink a real update's effect on the logits,\nto between a seventh and a little over half", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "logit_shift.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
