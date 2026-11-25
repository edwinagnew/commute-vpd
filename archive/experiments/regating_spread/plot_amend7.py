"""The room-to-move figure for regating_spread (E36, Amendment 7), from the results JSON only (no model).

Every number is a share of Γ's switching recovered at every held-out position of the lowercase-text fine-tune (R):
0 = the original model's importance values everywhere, 1 = Γ's values on the fine-tuned model. Every rule adds its
predicted change to the position's own original values and clips to [0, 1]; all are fitted on 262,144 positions.
  amend7.png  recovery at steps 10, 100 and 300 for the 8-nearest rule on the original importance values (the
              reference), the two value-bin rules (each subcomponent's change from its own original value alone, 16
              equal-width or 16 equal-count bins), the straight line per subcomponent (the bin rules' positive control),
              and the 4-token text summary with 32 neighbours; B1's bar (the nearest rule minus 0.3) marked at step 300

Usage (repo root):  uv run python experiments/regating_spread/plot_amend7.py [--results DIR] [--figures DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MUTED, GRID = "#52514e", "#d9d8d4"
STEPS = ("10", "100", "300")
GAP = 0.3
RULES = (("near_g0", "8 nearest positions in all the original values", "#1baf7a", "o-"),
         ("text4_k32", "32 nearest positions in the text of the last 4 tokens", "#e8a33d", "o-"),
         ("bin16w", "each value alone, 16 equal-width bins", "#7b52c4", "s-"),
         ("bin16q", "each value alone, 16 equal-count bins (fits worse than the line at steps 100, 300)", "#b58ae0",
          "s--"),
         ("fix", "each value alone, a straight line (control)", "#d1603d", "^:"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    a = ap.parse_args()
    st = json.loads((a.results / "regating_spread_amend7.json").read_text())["states"]
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    x = range(len(STEPS))
    for key, label, color, style in RULES:
        ax.plot(x, [st[s][f"R_{key}"] for s in STEPS], style, color=color, label=label)
    bar = st[STEPS[-1]]["R_near_g0"] - GAP
    ax.plot([len(STEPS) - 1.15, len(STEPS) - 0.85], [bar, bar], color=MUTED, lw=2)
    ax.text(len(STEPS) - 1.17, bar, "filed bar: nearest rule − 0.3", fontsize=7, color=MUTED, ha="right", va="center")
    ax.set_xticks(list(x), [f"step {s}" for s in STEPS])
    ax.set_xlim(-0.3, len(STEPS) - 0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8, ls="-.")
    ax.text(-0.3, 1.01, "Γ's own values on the fine-tuned model", fontsize=7, color=MUTED, va="bottom")
    ax.text(-0.3, -0.01, "original values everywhere", fontsize=7, color=MUTED, va="top")
    ax.set_ylim(-0.12, 1.08)
    ax.set_ylabel("share of the switching effect recovered")
    ax.set_title("Is the original values' edge only each value's room to move?\n"
                 "A function of each value alone recovers far less than the nearest positions", fontsize=10)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=8, frameon=False, loc="lower center", ncol=1)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    a.figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figures / "amend7.png", dpi=150)
    print(f"wrote {a.figures}/amend7.png")


if __name__ == "__main__":
    main()
