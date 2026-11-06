"""E104's figure from results/table.json (CPU): one panel per carried row. Each panel puts RL, its shuffled-reward null
and own-samples SFT at steps 10 and 100 (one point per seed) on the same axes as E94's template and TOFU full update
(dashed lines, seed means) and, on the think panel, the three scaled copies. A number that reads no call is left out.

  python experiments/qwen_d3_gsm8k/plot.py   ->  figures/rows.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rules

FIG = rules.HERE / "figures"
COLOUR = {"R": "#2a6fdb", "N": "#6b6b6b", "F": "#e07b00"}
LABEL = {"R": "RL", "N": "shuffled-reward null", "F": "own-samples SFT"}
TITLE = {"outside": "outside share (lower: more inside the subcomponents)",
         "placement": "placement excess over the Gaussian null",
         "think": "think fold T (overlap with think mode / chance)"}


def anchor_lines(ax, nums: dict, number: str) -> None:
    for key, style in (("template", "--"), ("tofu_full", ":")):
        m = rules.mean([rules.value(nums, n, number) for n in rules.ANCHORS[key]])
        if m is not None:
            ax.axhline(m, color="k", ls=style, lw=1, label=f"E94 {key.replace('_', ' ')}")


def panel(ax, nums: dict, number: str) -> None:
    for i, (arm, seeds) in enumerate(rules.ARMS.items()):
        for j, s in enumerate(rules.STEPS):
            ys = [rules.value(nums, f"{a}_{s}", number) for a in seeds]
            xs = [j * 4 + i + 0.12 * k for k in range(len(ys))]
            pts = [(x, y) for x, y in zip(xs, ys, strict=True) if y is not None]
            if pts:
                ax.scatter(*zip(*pts, strict=True), color=COLOUR[arm], label=LABEL[arm] if j == 0 else None, zorder=3)
    if number == "think":
        for k, name in enumerate(rules.COPIES):
            y = rules.value(nums, name, "think")
            if y is not None:
                ax.scatter([8.5 + k], [y], marker="s", facecolors="none", edgecolors="k", zorder=3)
                ax.annotate(name, (8.5 + k, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    anchor_lines(ax, nums, number)
    ax.set_xticks([1, 5] + ([9.5] if number == "think" else []),
                  ["step 10", "step 100"] + (["scaled copies"] if number == "think" else []))
    ax.set_title(TITLE[number], fontsize=9)


def main() -> None:
    nums = json.loads((rules.RESULTS / "table.json").read_text())["numbers"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, number in zip(axes, rules.NUMBERS, strict=True):
        panel(ax, nums, number)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    out = Path(FIG / "rows.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
