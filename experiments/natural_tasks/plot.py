"""E124's figure from results/stage0.json and E112's stage0.json, no model.

  figures/p0_blocks.png  per block, the Pile-trained gender probe's accuracy on held-out-name task windows at the
                         prediction position (all, female names, male names), the Hewitt-Liang control probe's, the
                         0.90 bar, chance, and E112's capitalisation probe on its own task as the reference that passed

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/natural_tasks/plot.py
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
E112 = HERE.parent / "case_label" / "results" / "stage0.json"
BLUE, ORANGE, GREEN, GREY, INK, GRID = "#2a78d6", "#eb6834", "#2e9e6b", "#7a7a75", "#52514e", "#e4e3df"
PURPLE = "#8c6bb1"
BAR = 0.90  # E112's possession accuracy bar, experiments/case_label/results/predictions.json .rules.P0.accuracy


def main() -> None:
    P0 = json.loads((RES / "stage0.json").read_text())["P0"]
    ref = json.loads(E112.read_text())["P0"]["acc_case"]
    blocks = list(range(len(P0["acc_case"])))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.axvspan(blocks[-1] - 0.5, blocks[-1] + 0.5, color=GRID, alpha=0.6, lw=0)
    ax.text(blocks[-1], 0.33, "last block:\nnot counted", ha="center", fontsize=8, color=INK)
    ax.plot(blocks, ref, color=PURPLE, lw=1.2, ls="-.", marker="s", ms=4,
            label="reference: capitalisation probe on the case-label task (passed at block 1)")  # fmt: skip
    ax.plot(blocks, P0["acc_case"], color=BLUE, lw=2.2, marker="o", label="gender probe, all held-out windows")
    ax.plot(blocks, P0["acc_case_by_gender"]["female"], color=ORANGE, lw=1.2, ls="--", marker="o", ms=3,
            label="gender probe, female names")  # fmt: skip
    ax.plot(blocks, P0["acc_case_by_gender"]["male"], color=GREEN, lw=1.2, ls="--", marker="o", ms=3,
            label="gender probe, male names")  # fmt: skip
    ax.plot(blocks, P0["acc_control"], color=GREY, lw=1.6, marker="o", label="control probe: random bit per name")
    for b, a in zip(blocks, P0["acc_case"], strict=True):
        dy = 0.035 if b == 0 else -0.06  # under the point, except block 0 where the control line sits under it
        ax.text(b + 0.1, a + dy, str(Decimal(str(a)).quantize(Decimal("0.01"), ROUND_HALF_UP)), color=BLUE, fontsize=8,
                bbox={"fc": "white", "ec": "none", "pad": 0.5})  # fmt: skip
    ax.axhline(BAR, color=INK, ls="--", lw=1)
    ax.text(-0.45, BAR + 0.01, "possession bar 0.90", va="bottom", color=INK, fontsize=8)
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.text(-0.45, 0.505, "chance", va="bottom", color=GREY, fontsize=8)
    ax.set(xticks=blocks, xticklabels=[f"block {b}" for b in blocks], xlim=(-0.5, blocks[-1] + 0.5), ylim=(0.3, 1.32),
           ylabel="accuracy on held-out names at the pronoun position",
           title="Does the small model hold the name's gender where it picks he or she?\nNot linearly at 0.90 before its last block")  # fmt: skip
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "p0_blocks.png", dpi=150)


if __name__ == "__main__":
    main()
