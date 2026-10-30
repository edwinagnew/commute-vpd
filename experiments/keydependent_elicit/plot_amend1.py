"""E54 Amendment 1's one figure, from results/amend1.json (no model).

  figures/a1_units.png  for the elicit map and the matched cipher, two seeds each: r (log-odds), S in nats, and S_lo
                        (S read in log-odds, r's unit), with each reading's gap, larger seed range and the
                        filed minimum gap of 0.2 written above, and the 0.3 cap (filed for S in nats) drawn

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/keydependent_elicit/plot_amend1.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLOR = {"cipher": "#eb6834", "elicit": "#1f9e6e"}
INK, GRID = "#52514e", "#e4e3df"
READINGS = [("r", "re-gating share r\n(log-odds)"), ("S", "steering share S\n(nats, as filed)"),
            ("S_lo", "steering share S\n(log-odds, r's unit)")]
MIN_GAP = 0.2  # E54's separation rule, predictions_amend1.json thresholds.min_gap
KINDS = [("elicit", "elicit map"), ("cipher", "cipher, matched size")]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def main() -> None:
    st: dict[str, Any] = json.loads((HERE / "results" / "amend1.json").read_text())["states"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for x, (key, _) in enumerate(READINGS):
        means, ranges = {}, {}
        for k, (kind, label) in enumerate(KINDS):
            ys = [st[f"{kind}_s{s}"][key] for s in (0, 1)]
            means[kind], ranges[kind] = mean(ys), max(ys) - min(ys)
            xs = [x + 0.18 * (k - 0.5) + 0.05 * (i - 0.5) for i in range(2)]
            ax.scatter(xs, ys, s=36, color=COLOR[kind], zorder=3, label=label if x == 0 else None)
        top = max(st[f"{kind}_s{s}"][key] for kind, _ in KINDS for s in (0, 1))
        note = (f"gap {means['elicit'] - means['cipher']:.2f}\nseed range {max(ranges.values()):.2f}\n"
                f"separates if gap > range and >= {MIN_GAP}")
        ax.annotate(note, (x, max(top, 0.3)), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom",
                    fontsize=7, color=INK)
    ax.plot([0.5, len(READINGS) - 0.5], [0.3, 0.3], color=INK, lw=0.8, ls="--")
    ax.annotate("cap 0.3, filed for nats", (len(READINGS) - 0.5, 0.3), xytext=(-2, -10), textcoords="offset points",
                ha="right", fontsize=7, color=INK)
    ax.set_xticks(range(len(READINGS)), [r[1] for r in READINGS], fontsize=8)
    ax.set_xlim(-0.5, len(READINGS) - 0.5)
    ax.set_ylim(-0.08, 1.12)
    ax.axhline(0, color=INK, lw=0.7)
    ax.axhline(1, color=INK, lw=0.7, ls="--")
    for y, label in ((0, "original model reads 0"), (1, "fine-tuned model reads 1")):
        ax.annotate(label, (0, y), xycoords=("axes fraction", "data"), xytext=(3, 3),
                    textcoords="offset points", fontsize=7, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.9))
    ax.set_title("Read in the same unit as r, does steering still fail to separate the pair?", fontsize=10,
                 color=INK, loc="left")
    ax.set_ylabel("share of the fine-tune's gain", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "a1_units.png", dpi=150)


if __name__ == "__main__":
    main()
