"""The sampling and space figure for regating_spread (E36, Amendments 1 to 4), from the results JSONs only (no model).

Every number is a share of Γ's switching recovered at every held-out position of the lowercase-text fine-tune (R):
0 = the original model's importance values everywhere, 1 = Γ's values on the fine-tuned model.
  sampling.png  the nearest rule on the original importance values against the number of fitted positions (16,384,
                65,536, 262,144), one line per read step; at 262,144 the same rule on the attention inputs Γ reads
                (diamonds) and on a 16-token text summary the model did not produce (triangles);
                the token rule and the rule per position in the row, refitted on 262,144, as dotted and dashed lines

Usage (repo root):  uv run python experiments/regating_spread/plot_sampling.py [--results DIR] [--figures DIR]
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
STEPS = (("10", "#e87ba4"), ("100", "#2a78d6"), ("300", "#1baf7a"))
NS = (16384, 65536, 262144)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    a = ap.parse_args()
    a2 = json.loads((a.results / "regating_spread_amend2.json").read_text())["states"]
    a4 = json.loads((a.results / "regating_spread_amend4.json").read_text())["states"]
    a5 = json.loads((a.results / "regating_spread_amend5.json").read_text())["states"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for s, color in STEPS:
        ax.plot(NS, [a2[s]["R_near16"], a2[s]["R_near65"], a4[s]["R_near_g0"]], "o-", color=color, label=f"step {s}")
        ax.plot([NS[-1]], [a4[s]["R_near_resid"]], "D", color=color, mec="black", mew=0.5, ms=7)
        ax.plot([NS[-1]], [a5[s]["R_near_text16"]], "^", color=color, mec="black", mew=0.5, ms=7)
        ax.axhline(a4[s]["R_token"], color=color, ls=":", lw=1)
        ax.axhline(a4[s]["R_position"], color=color, ls="--", lw=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8, ls="-.")
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS, ["16,384", "65,536", "262,144"])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("fitted positions the nearest rule draws on")
    ax.set_ylabel("share of the switching effect recovered")
    ax.plot([], [], "o-", color=MUTED, label="8 nearest, original importance values")
    ax.plot([], [], "D", color=MUTED, mec="black", mew=0.5, label="8 nearest, attention inputs Γ reads")
    ax.plot([], [], "^", color=MUTED, mec="black", mew=0.5, label="8 nearest, text of 16 tokens")
    ax.plot([], [], ":", color=MUTED, label="one change per current token (262,144)")
    ax.plot([], [], "--", color=MUTED, label="one change per position in the row (262,144)")
    ax.legend(fontsize=7.5, frameon=False, loc="center right")
    ax.set_title("Positions that look alike to the original model get alike changes", fontsize=10)
    fig.tight_layout()
    a.figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figures / "sampling.png", dpi=150)
    print(f"wrote {a.figures}/sampling.png")


if __name__ == "__main__":
    main()
