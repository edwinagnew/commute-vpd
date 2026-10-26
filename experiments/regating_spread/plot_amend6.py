"""The whole-preceding-text figure for regating_spread (E36, Amendment 6), from the results JSONs only (no model).

Every number is a share of Γ's switching recovered at every held-out position of the lowercase-text fine-tune (R):
0 = the original model's importance values everywhere, 1 = Γ's values on the fine-tuned model. Each series is the
8-nearest rule on 262,144 fitted positions, in one space.
  amend6.png  left: recovery at steps 10, 100 and 300 for the original importance values (the reference), the 4-token
              text summary, the bag of every preceding token, and the bag joined with the 4-token summary, with
              Amendment 4's one change per current token as a dotted line and the 0.014 noise floor as a band around
              the 4-token summary; right: the same four spaces at step 300 by position in the row

Usage (repo root):  uv run python experiments/regating_spread/plot_amend6.py [--results DIR] [--figures DIR]
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
SPACES = (("near_g0", "original importance values", "#1baf7a"), ("text4", "text of the last 4 tokens", "#e8a33d"),
          ("text_full", "whole preceding text joined with the last 4", "#c2417a"),
          ("prefix_bag", "bag of every preceding token", "#8a6f4e"))
STEPS = ("10", "100", "300")
BUCKETS = (("0-32", "0 to 31"), ("32-128", "32 to 127"), ("128-511", "128 to 511"))
NOISE = 0.014


def references(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8, ls="-.")
    ax.text(ax.get_xlim()[0], 1.01, "Γ's own values on the fine-tuned model", fontsize=7, color=MUTED, va="bottom")
    ax.text(ax.get_xlim()[0], -0.01, "original values everywhere", fontsize=7, color=MUTED, va="top")
    ax.set_ylim(-0.12, 1.08)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    a = ap.parse_args()
    a6 = json.loads((a.results / "regating_spread_amend6.json").read_text())["states"]
    a4 = json.loads((a.results / "regating_spread_amend4.json").read_text())["states"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    x = range(len(STEPS))
    t4 = [a6[s]["R_text4"] for s in STEPS]
    left.fill_between(x, [v - NOISE for v in t4], [v + NOISE for v in t4], color=SPACES[1][2], alpha=0.2, lw=0,
                      label="noise floor around the last 4 tokens")
    for key, label, color in SPACES:
        left.plot(x, [a6[s][f"R_{key}"] for s in STEPS], "o-", color=color, label=label)
    left.plot(x, [a4[s]["R_token"] for s in STEPS], ":", color="#2a78d6",
              label="one change per current token (Amendment 4)")
    left.set_xticks(list(x), [f"step {s}" for s in STEPS])
    left.set_ylabel("share of the switching effect recovered")
    left.set_title("By training step, all positions", fontsize=9.5)
    w = 0.2
    for j, (key, label, color) in enumerate(SPACES):
        right.bar([i + (j - 1.5) * w for i in range(len(BUCKETS))],
                  [a6["300"][f"R_{key}_by_bucket"][b] for b, _ in BUCKETS], w, color=color, label=label)
    right.set_xticks(range(len(BUCKETS)), [t for _, t in BUCKETS])
    right.set_xlabel("position in the row")
    right.set_title("Step 300, by position in the row", fontsize=9.5)
    for ax in (left, right):
        references(ax)
    fig.legend(*left.get_legend_handles_labels(), fontsize=8, frameon=False, loc="lower center", ncol=3)
    fig.suptitle("Adding the whole preceding text to the last 4 tokens does not bring the text up to the model's view",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    a.figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figures / "amend6.png", dpi=150)
    print(f"wrote {a.figures}/amend6.png")


if __name__ == "__main__":
    main()
